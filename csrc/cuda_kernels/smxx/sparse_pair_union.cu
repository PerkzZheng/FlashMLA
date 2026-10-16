#include "sparse_pair_union.h"

#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <climits>

namespace {

constexpr int THREADS = 256;

struct Segment {
    uint32_t index;
    uint32_t membership;
};

struct SegmentOr {
    __device__ Segment operator()(Segment left, Segment right) const {
        return {right.index, right.membership | (left.index == right.index ? left.membership : 0)};
    }
};

__device__ int load_index(const SparsePairUnionParams& params, int row, int position) {
    int length = params.lengths ? params.lengths[row] : params.topk;
    int index = position < params.topk && position < length ?
                params.indices[(int64_t)row * params.topk + position] : -1;
    return index >= 0 && index < params.kv_bound ? index : INT_MAX;
}

template<int ITEMS>
__global__ void pair_radix_union(__grid_constant__ const SparsePairUnionParams params) {
    using Sort = cub::BlockRadixSort<uint32_t, THREADS, ITEMS>;
    using Scan = cub::BlockScan<Segment, THREADS>;
    using Count = cub::BlockScan<int, THREADS>;
    __shared__ union {
        typename Sort::TempStorage sort;
        typename Scan::TempStorage scan;
        typename Count::TempStorage count;
    } temporary;
    __shared__ uint32_t sorted[THREADS * ITEMS];
    int* output = params.packed + (int64_t)blockIdx.x * params.capacity;
    uint32_t keys[ITEMS];
    #pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
        int candidate = threadIdx.x * ITEMS + item;
        int query = candidate / params.topk;
        int index = query < 2 ? load_index(params, blockIdx.x * 2 + query, candidate % params.topk) : INT_MAX;
        keys[item] = index == INT_MAX ? UINT32_MAX : ((uint32_t)index << 2) | (1u << query);
    }
    Sort(temporary.sort).Sort(keys, 2, 34 - __clz((uint32_t)params.kv_bound));
    __syncthreads();
    Segment segments[ITEMS];
    #pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
        sorted[threadIdx.x * ITEMS + item] = keys[item] >> 2;
        segments[item] = {keys[item] >> 2, keys[item] & 3};
    }
    __syncthreads();
    Scan(temporary.scan).InclusiveScan(segments, segments, SegmentOr{});
    __syncthreads();
    bool emit[ITEMS];
    int count = 0;
    #pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
        int rank = threadIdx.x * ITEMS + item;
        emit[item] = segments[item].index < (uint32_t)params.kv_bound &&
                     (rank + 1 == THREADS * ITEMS || sorted[rank + 1] != segments[item].index);
        count += emit[item];
    }
    int offset, total;
    Count(temporary.count).ExclusiveSum(count, offset, total);
    #pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
        if (emit[item]) output[offset++] = (segments[item].index << 2) | segments[item].membership;
    }
    for (int position = total + threadIdx.x; position < params.capacity; position += THREADS)
        output[position] = -1;
    if (threadIdx.x == 0) params.union_lengths[blockIdx.x] = total;
}

template<int ITEMS>
__global__ void pair_sort_rows(__grid_constant__ const SparsePairUnionParams params) {
    using Sort = cub::BlockRadixSort<int, THREADS, ITEMS>;
    __shared__ typename Sort::TempStorage temporary;
    int keys[ITEMS];
    #pragma unroll
    for (int item = 0; item < ITEMS; ++item)
        keys[item] = load_index(params, blockIdx.x, threadIdx.x * ITEMS + item);
    Sort(temporary).Sort(keys, 0, 32 - __clz((uint32_t)params.kv_bound));
    #pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
        int position = threadIdx.x * ITEMS + item;
        params.scratch[(int64_t)blockIdx.x * params.sorted_capacity + position] = keys[item];
    }
}

template<int ITEMS>
__global__ void pair_merge_union(__grid_constant__ const SparsePairUnionParams params) {
    __shared__ int rows[2][THREADS * ITEMS];
    using Scan = cub::BlockScan<int, THREADS>;
    __shared__ typename Scan::TempStorage temporary;
    int* output = params.packed + (int64_t)blockIdx.x * params.capacity;
    #pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
        int position = item * THREADS + threadIdx.x;
        rows[0][position] = params.scratch[(int64_t)(blockIdx.x * 2) * params.sorted_capacity + position];
        rows[1][position] = params.scratch[(int64_t)(blockIdx.x * 2 + 1) * params.sorted_capacity + position];
    }
    __syncthreads();
    int begin = threadIdx.x * 2 * params.topk / THREADS;
    int end = (threadIdx.x + 1) * 2 * params.topk / THREADS;
    int lower = max(0, begin - params.topk);
    int upper = min(begin, params.topk);
    while (lower < upper) {
        int first = (lower + upper) / 2;
        int second = begin - first;
        if (second > 0 && first < params.topk && rows[1][second - 1] >= rows[0][first])
            lower = first + 1;
        else upper = first;
    }
    int first = lower;
    int second = begin - lower;
    int remaining = end - begin;
    if (remaining > 0 && first > 0 && second < params.topk && rows[0][first - 1] == rows[1][second]) {
        ++second;
        --remaining;
    }
    uint32_t entries[2 * ITEMS];
    int count = 0;
    #pragma unroll
    for (int entry = 0; entry < 2 * ITEMS; ++entry) {
        entries[entry] = UINT32_MAX;
        if (remaining > 0) {
            int left = first < params.topk ? rows[0][first] : INT_MAX;
            int right = second < params.topk ? rows[1][second] : INT_MAX;
            bool in_first = left <= right;
            bool in_second = right <= left;
            bool consume_second = in_second && remaining > (int)in_first;
            int index = min(left, right);
            if (index != INT_MAX) {
                entries[entry] = ((uint32_t)index << 2) | (uint32_t)in_first | ((uint32_t)in_second << 1);
                ++count;
            }
            first += in_first;
            second += consume_second;
            remaining -= (int)in_first + (int)consume_second;
        }
    }
    int offset, total;
    Scan(temporary).ExclusiveSum(count, offset, total);
    #pragma unroll
    for (int entry = 0; entry < 2 * ITEMS; ++entry)
        if (entry < count) output[offset + entry] = entries[entry];
    for (int position = total + threadIdx.x; position < params.capacity; position += THREADS)
        output[position] = -1;
    if (threadIdx.x == 0) params.union_lengths[blockIdx.x] = total;
}

__global__ void pair_bitmap_union(__grid_constant__ const SparsePairUnionParams params, int domain) {
    extern __shared__ uint32_t bitmap[];
    using Scan = cub::BlockScan<int, THREADS>;
    __shared__ typename Scan::TempStorage temporary;
    int words = (domain + 31) / 32;
    int base = params.request_stride ? (blockIdx.x / (params.queries / 2)) * params.request_stride : 0;
    int* output = params.packed + (int64_t)blockIdx.x * params.capacity;
    for (int position = threadIdx.x; position < 2 * words; position += THREADS) bitmap[position] = 0;
    __syncthreads();
    int first_length = params.lengths ? params.lengths[blockIdx.x * 2] : params.topk;
    int second_length = params.lengths ? params.lengths[blockIdx.x * 2 + 1] : params.topk;
    const int* first_row = params.indices + (int64_t)(blockIdx.x * 2) * params.topk;
    const int* second_row = first_row + params.topk;
    for (int position = threadIdx.x; position < params.topk; position += THREADS) {
        uint32_t first = position < first_length ? (uint32_t)first_row[position] - (uint32_t)base : UINT32_MAX;
        uint32_t second = position < second_length ? (uint32_t)second_row[position] - (uint32_t)base : UINT32_MAX;
        if (first < (uint32_t)domain)
            atomicOr(bitmap + first / 32, 1u << (first % 32));
        if (second < (uint32_t)domain)
            atomicOr(bitmap + words + second / 32, 1u << (second % 32));
    }
    __syncthreads();
    int word_begin = threadIdx.x * words / THREADS;
    int word_end = (threadIdx.x + 1) * words / THREADS;
    int count = 0;
    for (int word = word_begin; word < word_end; ++word)
        count += __popc(bitmap[word] | bitmap[words + word]);
    int offset, total;
    Scan(temporary).ExclusiveSum(count, offset, total);
    for (int word = word_begin; word < word_end; ++word) {
        uint32_t left = bitmap[word];
        uint32_t right = bitmap[words + word];
        uint32_t remaining = left | right;
        while (remaining) {
            int bit = __ffs(remaining) - 1;
            int membership = ((left >> bit) & 1) | (((right >> bit) & 1) << 1);
            output[offset++] = ((uint32_t)(base + word * 32 + bit) << 2) | membership;
            remaining &= remaining - 1;
        }
    }
    for (int position = total + threadIdx.x; position < params.capacity; position += THREADS)
        output[position] = -1;
    if (threadIdx.x == 0) params.union_lengths[blockIdx.x] = total;
}

template<int ITEMS>
void launch(const SparsePairUnionParams& params) {
    int groups = params.batch * params.queries / 2;
    if (params.algorithm == 0) {
        if (params.topk <= 128)
            pair_radix_union<1><<<groups, THREADS, 0, params.stream>>>(params);
        else
            pair_radix_union<2 * ITEMS><<<groups, THREADS, 0, params.stream>>>(params);
    } else {
        pair_sort_rows<ITEMS><<<groups * 2, THREADS, 0, params.stream>>>(params);
        pair_merge_union<ITEMS><<<groups, THREADS, 0, params.stream>>>(params);
    }
}

}

void run_sparse_pair_union(const SparsePairUnionParams& params) {
    if (params.algorithm == 2) {
        int domain = params.request_stride ? params.request_stride : params.kv_bound;
        pair_bitmap_union<<<params.batch * params.queries / 2, THREADS,
                            2 * ((domain + 31) / 32) * sizeof(uint32_t), params.stream>>>(params, domain);
    } else if (params.sorted_capacity == 256) launch<1>(params);
    else if (params.sorted_capacity == 512) launch<2>(params);
    else if (params.sorted_capacity == 1024) launch<4>(params);
    else if (params.sorted_capacity == 2048) launch<8>(params);
    else if (params.sorted_capacity == 4096) launch<16>(params);
}
