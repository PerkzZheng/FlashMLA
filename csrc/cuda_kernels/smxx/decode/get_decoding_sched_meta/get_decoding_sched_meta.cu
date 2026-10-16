/*
Sparse Attention Decoding — Tile Scheduler Metadata Generation (GPU Architecture-Independent)

Computes the work partitioning metadata for split-KV decoding. A single warp
greedily assigns KV blocks from all batch requests to num_sm_parts SM partitions,
producing a DecodingSchedMeta descriptor per partition and a cumulative split
count array (num_splits_ptr). Supports optional per-request topk_length and
extra KV cache (extra_topk or extra_topk_length).

Template parameters:
  HAVE_TOPK_LEN      — Whether per-request topk_length is provided
  HAVE_EXTRA_TOPK    — Whether extra KV cache is enabled
  HAVE_EXTRA_TOPK_LEN — Whether per-request extra_topk_length is provided

Grid: [1, 1, 1]
Block: [32, 1, 1]

I/O: See GetDecodeSchedMetaParams in csrc/params.h
*/
#include "get_decoding_sched_meta.h"

#include <kerutils/kerutils.cuh>
#include <cub/block/block_scan.cuh>

namespace smxx::decode {

template<bool HAVE_TOPK_LEN, bool HAVE_EXTRA_TOPK, bool HAVE_EXTRA_TOPK_LEN>
__global__ void __launch_bounds__(32, 1, 1)
get_mla_metadata_kernel(__grid_constant__ const GetDecodeSchedMetaParams params) {
    DecodingSchedMeta *tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr;
    int *num_splits_ptr = params.num_splits_ptr;
    int batch_size = params.b;
    int block_size_topk = params.block_size_topk;
    int fixed_overhead_num_blocks = params.fixed_overhead_num_blocks;
    int num_sm_parts = params.num_sm_parts;

    extern __shared__ int shared_mem[];
    int* __restrict__ num_blocks_shared = shared_mem; // [batch_size]
    int* __restrict__ num_splits_shared = shared_mem + batch_size; // [batch_size+1]
    int* __restrict__ seqlens_k_shared = shared_mem + batch_size*2+1; // [batch_size]
    int* __restrict__ last_block_idx_shared = shared_mem + batch_size*3+1; // [batch_size], inclusive

    int total_num_blocks = 0;
    for (int i = threadIdx.x; i < batch_size; i += 32) {
        int cur_s_k = HAVE_TOPK_LEN ? __ldg(params.topk_length + i) : params.topk;
        if (cur_s_k == 0) cur_s_k = 1;  // Ensure the main loop will never be empty
        if constexpr (HAVE_EXTRA_TOPK) {
            cur_s_k = ku::ceil(cur_s_k, block_size_topk);
            cur_s_k += HAVE_EXTRA_TOPK_LEN ? __ldg(params.extra_topk_length + i) : params.extra_topk;
        }

        seqlens_k_shared[i] = cur_s_k;
        int cur_last_block_idx = (int)((uint32_t)(cur_s_k-1) / (uint32_t)block_size_topk); // Inclusive
        int num_blocks = cur_last_block_idx + 1;
        total_num_blocks += num_blocks + fixed_overhead_num_blocks;
        num_blocks_shared[i] = num_blocks;
        last_block_idx_shared[i] = cur_last_block_idx;
    }
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        total_num_blocks += __shfl_xor_sync(uint32_t(-1), total_num_blocks, offset);
    }
    __syncwarp();

    if (cute::elect_one_sync()) {
        int payload = (int)ku::ceil_div((uint32_t)total_num_blocks, (uint32_t)num_sm_parts) + fixed_overhead_num_blocks;

        int now_req_idx = 0;        // The first unfinished (not all blocks are allocated to some SM part) request
        int now_block = 0;          // The first unfinished block index of the current request
        int now_n_split_idx = 0;    // The number of splits of the current request
        int cum_num_splits = 0;     // The summation of splits of all previous requests
        num_splits_shared[0] = 0;
        for (int i = 0; i < num_sm_parts; ++i) {
            DecodingSchedMeta cur_meta;
            cur_meta.begin_req_idx = now_req_idx;
            cur_meta.begin_block_idx = now_block;
            cur_meta.begin_split_idx = now_n_split_idx;
            cur_meta.is_first_req_splitted = now_block != 0;
            int remain_payload = payload;
            while (now_req_idx < batch_size) {
                int num_blocks = num_blocks_shared[now_req_idx];
                int now_remain_blocks = num_blocks - now_block;
                if (remain_payload >= now_remain_blocks + fixed_overhead_num_blocks) {
                    // Consume the entire request
                    // Update `cum_num_splits` and `now_req_idx`, and jump to the next request
                    cum_num_splits += now_n_split_idx + 1;
                    num_splits_shared[now_req_idx + 1] = cum_num_splits;
                    remain_payload -= now_remain_blocks + fixed_overhead_num_blocks;
                    ++now_req_idx;
                    now_block = 0;
                    now_n_split_idx = 0;
                } else {
                    if (remain_payload - fixed_overhead_num_blocks > 0) {
                        // Consume part of the request
                        now_block += remain_payload - fixed_overhead_num_blocks;
                        ++now_n_split_idx;
                        remain_payload = 0;
                    } else {
                        // Skip
                    }
                    break;
                }
            }
            cur_meta.end_req_idx = now_block > 0 ? now_req_idx : now_req_idx - 1;
            cur_meta.end_block_idx = now_block > 0 ? now_block : last_block_idx_shared[now_req_idx-1] + 1;
            cur_meta.is_last_req_splitted = cur_meta.end_block_idx != last_block_idx_shared[cur_meta.end_req_idx] + 1;
            if (cur_meta.begin_req_idx == cur_meta.end_req_idx) {
                cur_meta.is_first_req_splitted = cur_meta.is_last_req_splitted = cur_meta.is_first_req_splitted || cur_meta.is_last_req_splitted;
            }
            #ifdef KERUTILS_ENABLE_SM100A
                KU_STG_256(tile_scheduler_metadata_ptr+i, &cur_meta, "evict_normal", "evict_normal");
            #else
                tile_scheduler_metadata_ptr[i] = cur_meta;
            #endif
        }
        KU_TRAP_ONLY_DEVICE_ASSERT(now_req_idx == batch_size && now_block == 0 && now_n_split_idx == 0);
    }
    __syncwarp();

    for (int i = threadIdx.x; i <= batch_size; i += 32) {
        num_splits_ptr[i] = num_splits_shared[i];
    }
}


void run_get_decoding_sched_meta_kernel(const GetDecodeSchedMetaParams& params) {
    int smem_size = (params.b * 4 + 1) * sizeof(int);
    auto kernel = 
        params.topk_length != nullptr ?
            (params.extra_topk != 0 ?
                (params.extra_topk_length != nullptr ?
                    get_mla_metadata_kernel<true, true, true> :
                    get_mla_metadata_kernel<true, true, false>
                ) :
                get_mla_metadata_kernel<true, false, false>
            ) :
            (params.extra_topk != 0 ?
                (params.extra_topk_length != nullptr ?
                    get_mla_metadata_kernel<false, true, true> :
                    get_mla_metadata_kernel<false, true, false>
                ) :
                get_mla_metadata_kernel<false, false, false>
            );
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    kernel<<<1, 32, smem_size, params.stream>>>(params);
    KU_CHECK_KERNEL_LAUNCH();
}


__global__ void __launch_bounds__(256)
get_pair_mla_metadata_kernel(__grid_constant__ const GetDecodeSchedMetaParams params) {
    using Scan = cub::BlockScan<int, 256>;
    __shared__ typename Scan::TempStorage temporary;
    __shared__ int block_offsets[257];
    __shared__ int first_parts[256];
    __shared__ int split_counts[256];
    int group = threadIdx.x;
    int blocks = 0;
    if (group < params.b) {
        int length = params.topk_length ? params.topk_length[group] : params.topk;
        blocks = (max(length, 1) + params.block_size_topk - 1) / params.block_size_topk;
    }
    int block_begin, total_blocks;
    Scan(temporary).ExclusiveSum(blocks, block_begin, total_blocks);
    if (group < params.b) block_offsets[group] = block_begin;
    if (group == 0) block_offsets[params.b] = total_blocks;
    __syncthreads();

    int minimum_blocks = max(params.fixed_overhead_num_blocks, 1);
    int useful_parts = max(params.b, (total_blocks + minimum_blocks - 1) / minimum_blocks);
    int active_parts = min(useful_parts, params.num_sm_parts);
    int count = 0;
    if (group < params.b) {
        int first_part = ((int64_t)(block_begin + 1) * active_parts - 1) / total_blocks;
        int last_part = ((int64_t)(block_begin + blocks) * active_parts - 1) / total_blocks;
        count = last_part - first_part + 1;
        first_parts[group] = first_part;
        split_counts[group] = count;
    }
    int split_begin, total_splits;
    Scan(temporary).ExclusiveSum(count, split_begin, total_splits);
    if (group < params.b) params.num_splits_ptr[group] = split_begin;
    if (group == 0) params.num_splits_ptr[params.b] = total_splits;
    __syncthreads();

    int part = threadIdx.x;
    if (part >= params.num_sm_parts) return;
    DecodingSchedMeta metadata{};
    if (part >= active_parts) {
        metadata.begin_req_idx = params.b;
        metadata.end_req_idx = params.b - 1;
    } else {
        int begin = (int64_t)total_blocks * part / active_parts;
        int end = (int64_t)total_blocks * (part + 1) / active_parts;
        auto locate_group = [&](int block) {
            int lower = 0, upper = params.b;
            while (lower + 1 < upper) {
                int middle = (lower + upper) / 2;
                if (block_offsets[middle] <= block) lower = middle;
                else upper = middle;
            }
            return lower;
        };
        int first_group = locate_group(begin);
        int last_group = locate_group(end - 1);
        metadata.begin_req_idx = first_group;
        metadata.end_req_idx = last_group;
        metadata.begin_block_idx = begin - block_offsets[first_group];
        metadata.end_block_idx = end - block_offsets[last_group];
        metadata.begin_split_idx = part - first_parts[first_group];
        metadata.is_first_req_splitted = split_counts[first_group] > 1;
        metadata.is_last_req_splitted = split_counts[last_group] > 1;
    }
    params.tile_scheduler_metadata_ptr[part] = metadata;
}

void run_get_pair_decoding_sched_meta_kernel(const GetDecodeSchedMetaParams& params) {
    if (params.b > 256 || params.num_sm_parts > 256) {
        run_get_decoding_sched_meta_kernel(params);
        return;
    }
    get_pair_mla_metadata_kernel<<<1, 256, 0, params.stream>>>(params);
    KU_CUDA_CHECK(cudaGetLastError());
}


}
