#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

struct SparsePairUnionParams {
    const int* indices;
    const int* lengths;
    int* packed;
    int* union_lengths;
    int* scratch;
    int batch;
    int queries;
    int topk;
    int capacity;
    int sorted_capacity;
    int kv_bound;
    int request_stride;
    int algorithm;
    cudaStream_t stream;
};

void run_sparse_pair_union(const SparsePairUnionParams& params);
