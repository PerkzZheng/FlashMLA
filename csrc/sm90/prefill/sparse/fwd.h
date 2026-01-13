#pragma once

#include "params.h"

namespace sm90 {

constexpr int SPLIT_KV_BLOCK_SIZE = 128;

void run_fwd_kernel(const SparsePrefillParams& params);

// Split-KV version of run_fwd_kernel
// This function launches the split-kv kernel which processes partitions of KV blocks
// 
// After calling this kernel, use run_flash_mla_combine_kernel with DecodingParams set as:
//   - b = s_q
//   - h_k = 1
//   - q_seq_per_hk = h_q
//   - o_batch_stride = h_q * d_v
//   - o_row_stride = d_v
//   - o_head_stride = 0 (not used since h_k=1)
//   - softmax_lse_ptr, softmax_lseaccum_ptr, oaccum_ptr from SparsePrefillParams
//
// The same get_mla_metadata_kernel can be used with:
//   - batch_size = s_q
//   - topk = params.topk (for sparse attention)
void run_fwd_splitkv_kernel(const SparsePrefillParams& params);

}
