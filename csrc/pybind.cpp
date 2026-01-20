// Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/flash_api.cpp
/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/fast_math.h>

#include "params.h"
#include "smxx/get_mla_metadata.h"
#include "smxx/mla_combine.h"
#include "sm90/decode/dense/splitkv_mla.h"
#include "sm90/decode/sparse_fp8/splitkv_mla.h"
#include "sm90/prefill/sparse/fwd.h"
#include "sm90/prefill/sparse/convert_indice.h"
#include "sm100/decode/sparse_fp8/splitkv_mla.h"
#include "sm100/prefill/dense/interface.h"
#include "sm100/prefill/sparse/fwd.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

struct Arch {
    int major;
    int minor;

    bool is_sm90() const {
        return major == 9 && minor == 0;
    }

    bool is_sm100() const {
        return major == 10;
    }

    void assert_is_supported() const {
        TORCH_CHECK(is_sm90() || is_sm100(), "Only SM90 and SM100 are supported");
    }
};

// DecodingAttnImplMeta - A struct to hold metadata for Decoding Attention Implementation (i.e. SM90 Dense BF16, SM90 Sparse FP8, etc.)
struct DecodingAttnImplMeta {
    int num_sm_parts;
    int fixed_overhead_num_blocks;
    int k_block_size;
};

DecodingAttnImplMeta get_attn_impl_meta(
    Arch arch,
    int sm_count,
    int num_q_tokens_per_head_k,
    int h_k,
    std::optional<int> h_q_,
    bool is_fp8_kvcache,
    bool is_sparse_attn
) {
    if (arch.is_sm90()) {
        if (is_sparse_attn) {
            if (is_fp8_kvcache) {
                TORCH_CHECK(h_q_.has_value());
                int h_q = h_q_.value();
                TORCH_CHECK(h_q % h_k == 0);
                int s_q = num_q_tokens_per_head_k * h_k / h_q;
                // FP8 + Sparse MLA
                return {
                    std::max((sm_count/2) / h_k / (cutlass::ceil_div(h_q/h_k, 2*64) * s_q), 1),
                    5,
                    64
                };
            } else {
                // Sparse BF16 MLA
                // TORCH_CHECK(false, "Sparse BF16 MLA is not supported on SM90");
                TORCH_CHECK(h_q_.has_value());
                int h_q = h_q_.value();
                TORCH_CHECK(h_q % h_k == 0);
                int s_q = num_q_tokens_per_head_k * h_k / h_q;
                // FP8 + Sparse MLA
                return {
                    std::max((sm_count) / h_k / (cutlass::ceil_div(h_q/h_k, 64) * s_q), 1),
                    5,
                    // Note we use 128 for bf16 kv cache kernel.
                    sm90::SPLIT_KV_BLOCK_SIZE
                };
            }
        } else {
            if (is_fp8_kvcache) {
                // Dense FP8 MLA
                TORCH_CHECK(false, "Dense FP8 MLA is not supported on SM90");
            } else {
                // Dense BF16 MLA
                return {
                    std::max(sm_count / h_k / cutlass::ceil_div(num_q_tokens_per_head_k, 64), 1),
                    5,
                    64
                };
            }
        }
    } else if (arch.is_sm100()) {
        if (is_sparse_attn) {
            if (is_fp8_kvcache) {
                TORCH_CHECK(h_q_.has_value());
                int h_q = h_q_.value();
                TORCH_CHECK(h_q % h_k == 0);
                int s_q = num_q_tokens_per_head_k * h_k / h_q;
                // FP8 + Sparse MLA
                return {
                    std::max(sm_count / h_k / (cutlass::ceil_div(h_q/h_k, 64) * s_q), 1),
                    5,
                    64
                };
            } else {
                // Sparse BF16 MLA
                TORCH_CHECK(false, "Sparse BF16 MLA is not supported on SM100");
            }
        } else {
            if (is_fp8_kvcache) {
                // FP8 MLA
                TORCH_CHECK(false, "FP8 Dence MLA is not supported on SM100");
            } else {
                // Normal BF16 MLA
                TORCH_CHECK(false, "BF16 Dence MLA is not supported on SM100");
            }
        }
    } else {
        TORCH_CHECK(false, "Unsupported GPU architecture");
    }
}


std::vector<at::Tensor>
get_mla_decoding_metadata(
    at::Tensor &seqlens_k,
    const int num_q_tokens_per_head_k,
    const int h_k,
    const std::optional<int> h_q,
    const bool is_fp8_kvcache,
    const std::optional<int> topk
) {
    bool is_sparse_attn = topk.has_value();
    CHECK_DEVICE(seqlens_k);
    TORCH_CHECK(seqlens_k.is_contiguous());
    TORCH_CHECK(seqlens_k.dtype() == torch::kInt32);
    if (is_sparse_attn)
        TORCH_CHECK(h_q.has_value(), "num_heads_q must be provided when topk is provided");

    int batch_size = seqlens_k.size(0);
    int *seqlens_k_ptr = seqlens_k.data_ptr<int>();
    auto options = seqlens_k.options();

    auto dprops = at::cuda::getCurrentDeviceProperties();
    int sm_count = dprops->multiProcessorCount;
    Arch arch = {dprops->major, dprops->minor};
    arch.assert_is_supported();
    // fixme
    DecodingAttnImplMeta attn_impl_meta = get_attn_impl_meta(arch, sm_count, num_q_tokens_per_head_k, h_k, h_q, is_fp8_kvcache, is_sparse_attn);

    auto tile_scheduler_metadata = torch::empty({attn_impl_meta.num_sm_parts, TileSchedulerMetaDataSize}, options);
    auto num_splits = torch::empty({batch_size + 1}, options);
    int *tile_scheduler_metadata_ptr = tile_scheduler_metadata.data_ptr<int>();
    int *num_splits_ptr = num_splits.data_ptr<int>();

    at::cuda::CUDAGuard device_guard{(char)seqlens_k.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    GetDecodingMetadataParams params = {};
    params.seqlens_k_ptr = seqlens_k_ptr;
    params.tile_scheduler_metadata_ptr = tile_scheduler_metadata_ptr;
    params.num_splits_ptr = num_splits_ptr;
    params.batch_size = batch_size;
    params.block_size_n = attn_impl_meta.k_block_size;
    params.fixed_overhead_num_blocks = attn_impl_meta.fixed_overhead_num_blocks;
    params.num_sm_parts = attn_impl_meta.num_sm_parts;
    params.topk = is_sparse_attn ? topk.value() : -1;
    run_get_mla_metadata_kernel(params, stream);

    return {tile_scheduler_metadata, num_splits};
}

std::vector<at::Tensor>
fwd_kvcache_mla(
    at::Tensor &q,                               // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor &kcache,                    // num_blocks x page_block_size x num_heads_k x head_size (when is_fp8 is False) or num_blocks x num_heads_k x (page_block_size*656) (when is_fp8 is True)
    const int head_size_v,
    const at::Tensor &seqlens_k,                 // batch_size
    const at::Tensor &block_table,               // batch_size x max_num_blocks_per_seq
    const float softmax_scale,
    bool is_causal,
    const at::Tensor &tile_scheduler_metadata,   // num_sm_parts x TileSchedulerMetaDataSize
    const at::Tensor &num_splits,                // batch_size + 1
    const bool &is_fp8,
    const std::optional<at::Tensor> &indices     // None, or batch_size x seqlen_q x topk
) {
    bool is_sparse_attn = indices.has_value();
    int topk = is_sparse_attn ? indices->size(-1) : -1;

    // Check the architecture
    auto dprops = at::cuda::getCurrentDeviceProperties();
    Arch arch = {dprops->major, dprops->minor};
    arch.assert_is_supported();

    // Check data types
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kBFloat16 || q_dtype == torch::kHalf);
    
    if (!is_fp8) {
        TORCH_CHECK(kcache.dtype() == q_dtype, "query and key must have the same dtype");
    } else {
        TORCH_CHECK(kcache.dtype() == torch::kFloat8_e4m3fn || kcache.dtype() == torch::kInt8 || kcache.dtype() == torch::kUInt8, "key must have dtype fp8_e4m3fn or int8 or uint8");
    }
    TORCH_CHECK(seqlens_k.dtype() == torch::kInt32, "seqlens_k must have dtype int32");
    TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
    TORCH_CHECK(tile_scheduler_metadata.dtype() == torch::kInt32, "tile_scheduler_metadata must have dtype int32");
    TORCH_CHECK(num_splits.dtype() == torch::kInt32, "num_splits must have dtype int32");
    TORCH_CHECK(!is_sparse_attn || indices->dtype() == torch::kInt32, "indices must have dtype int32");

    // Check device
    CHECK_DEVICE(q);
    CHECK_DEVICE(kcache);
    CHECK_DEVICE(seqlens_k);
    CHECK_DEVICE(block_table);
    CHECK_DEVICE(tile_scheduler_metadata);
    CHECK_DEVICE(num_splits);
    if (is_sparse_attn) CHECK_DEVICE(indices.value());

    // Check layout
    TORCH_CHECK(q.stride(-1) == 1, "q must have contiguous last dimension");
    TORCH_CHECK(kcache.stride(-1) == 1, "kcache must have contiguous last dimension");
    CHECK_CONTIGUOUS(seqlens_k);
    TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
    CHECK_CONTIGUOUS(tile_scheduler_metadata);
    CHECK_CONTIGUOUS(num_splits);
    TORCH_CHECK(!is_sparse_attn || indices->stride(-1) == 1, "indices must have contiguous last dimension");

    const auto sizes = q.sizes();
    const int batch_size = sizes[0];
    const int seqlen_q_ori = sizes[1];
    const int num_heads_q = sizes[2];
    const int head_size_k = sizes[3];
    TORCH_CHECK(head_size_k == 576, "Only head_size_k == 576 is supported");
    TORCH_CHECK(head_size_v == 512, "Only head_size_v == 576 is supported");

    const int max_num_blocks_per_seq = block_table.size(1);
    const int num_blocks = kcache.size(0);
    const int page_block_size = kcache.size(1);
    const int num_heads_k = kcache.size(2);
    TORCH_CHECK(page_block_size == 64, "Currently page_block_size must be 64");
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(num_heads_q % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (seqlen_q_ori == 1) { is_causal = false; }

    const int num_q_heads_per_hk = num_heads_q / num_heads_k;
    const int q_seq_per_hk = seqlen_q_ori * num_q_heads_per_hk;
    const int num_heads = num_heads_k;
    q = q.view({batch_size, seqlen_q_ori, num_heads_k, num_q_heads_per_hk, head_size_k}).transpose(2, 3)
            .reshape({batch_size, q_seq_per_hk, num_heads, head_size_k});

    CHECK_SHAPE(q, batch_size, q_seq_per_hk, num_heads, head_size_k);
    if (!is_fp8) {
        CHECK_SHAPE(kcache, num_blocks, page_block_size, num_heads_k, head_size_k);
    } else {
        int bytes_per_token = 512 + 64*2 + (512/128)*4;
        CHECK_SHAPE(kcache, num_blocks, page_block_size, num_heads_k, bytes_per_token);
        TORCH_CHECK(num_heads_k == 1, "Currently the number of k heads must be 1 when is_fp8_kvcache is True");
        TORCH_CHECK(kcache.stride(1) == bytes_per_token, "The whole block must be contiguous when is_fp8_cache is True");
    }
    CHECK_SHAPE(seqlens_k, batch_size);
    CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
    TORCH_CHECK(tile_scheduler_metadata.size(1) == TileSchedulerMetaDataSize);
    CHECK_SHAPE(num_splits, batch_size+1);
    if (is_sparse_attn) CHECK_SHAPE(indices.value(), batch_size, seqlen_q_ori, topk);

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();
    at::Tensor out = torch::empty({batch_size, q_seq_per_hk, num_heads, head_size_v}, opts);
    at::Tensor softmax_lse = torch::empty({batch_size, num_heads, q_seq_per_hk}, opts.dtype(at::kFloat));
    CHECK_CONTIGUOUS(softmax_lse);

    DecodingParams params = {};
    // Set the sizes.
    params.b = batch_size;
    params.s_q = seqlen_q_ori;
    params.q_seq_per_hk = q_seq_per_hk;
    params.seqlens_k_ptr = seqlens_k.data_ptr<int>();
    params.h_q = num_heads_q;
    params.h_k = num_heads_k;
    params.num_blocks = num_blocks;
    params.q_head_per_hk = num_q_heads_per_hk;
    params.is_causal = is_causal;
    params.d = head_size_k;
    params.d_v = head_size_v;
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = float(softmax_scale * M_LOG2E);
    params.topk = topk;
    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = kcache.data_ptr();
    params.o_ptr = out.data_ptr();
    params.indices_ptr = is_sparse_attn ? indices->data_ptr<int>() : nullptr;
    params.softmax_lse_ptr = softmax_lse.data_ptr();
    // All stride are in elements, not bytes.
    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = kcache.stride(0);
    params.o_batch_stride = out.stride(0);
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = kcache.stride(1);
    params.o_row_stride = out.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = kcache.stride(2);
    params.o_head_stride = out.stride(-2);
    params.indices_batch_stride = is_sparse_attn ? indices->stride(0) : 0;
    params.indices_row_stride = is_sparse_attn ? indices->stride(1) : 0;

    params.block_table = block_table.data_ptr<int>();
    params.block_table_batch_stride = block_table.stride(0);
    params.page_block_size = page_block_size;
    
    params.tile_scheduler_metadata_ptr = tile_scheduler_metadata.data_ptr<int>();
    params.num_sm_parts = tile_scheduler_metadata.size(0);
    params.num_splits_ptr = num_splits.data_ptr<int>();

    const int total_num_splits = batch_size + params.num_sm_parts;
    at::Tensor softmax_lse_accum = torch::empty({total_num_splits, num_heads, q_seq_per_hk}, opts.dtype(at::kFloat));
    at::Tensor out_accum = torch::empty({total_num_splits, num_heads, q_seq_per_hk, head_size_v}, opts.dtype(at::kFloat));
    CHECK_CONTIGUOUS(softmax_lse_accum);
    CHECK_CONTIGUOUS(out_accum);
    params.total_num_splits = total_num_splits;
    params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
    params.oaccum_ptr = out_accum.data_ptr();

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(head_size_k == 576);

    if (q_dtype == torch::kHalf) {
#ifdef FLASH_MLA_DISABLE_FP16
        TORCH_CHECK(false, "FlashMLA is compiled with -DFLASH_MLA_DISABLE_FP16. Please remove this flag from your environment and re-compile FlashMLA.");
#endif
    }

    if (arch.is_sm90()) {
        if (is_sparse_attn) {
            if (is_fp8) {
                TORCH_CHECK(q_dtype == torch::kBFloat16, "Sparse FP8 MLA only supports BFloat16 on SM90");
                sm90::run_flash_splitkv_mla_fp8_sparse_kernel(params, stream);
            } else {
                TORCH_CHECK(false, "Only FP8 kvcahe is supported for sparse MLA on SM90");
            }
        } else {
            if (is_fp8) {
                TORCH_CHECK(false, "Dense FP8 MLA is not supported on SM90");
            } else {
                if (q_dtype == torch::kBFloat16) {
                    sm90::run_flash_splitkv_mla_kernel<cutlass::bfloat16_t>(params, stream);
                } else if (q_dtype == torch::kHalf) {
#ifndef FLASH_MLA_DISABLE_FP16
                    sm90::run_flash_splitkv_mla_kernel<cutlass::half_t>(params, stream);
#endif
                } else {
                    TORCH_CHECK(false, "Unsupported dtype for dense MLA on SM90");
                }
            }
        }
    } else if (arch.is_sm100()) {
        TORCH_CHECK(is_fp8 && is_sparse_attn, "Only FP8 + Sparse attention is supported on SM100");
        sm100::run_flash_splitkv_mla_fp8_sparse_kernel(params, stream);
    } else {
        TORCH_CHECK(false, "Unsupported GPU architecture");
    }

    if (q_dtype == torch::kBFloat16) {
        run_flash_mla_combine_kernel<cutlass::bfloat16_t, DecodingParams>(params, stream);
    } else if (q_dtype == torch::kHalf) {
#ifndef FLASH_MLA_DISABLE_FP16
        run_flash_mla_combine_kernel<cutlass::half_t, DecodingParams>(params, stream);
#endif
    } else {
        TORCH_CHECK(false, "Unsupported tensor dtype for query");
    }

    out = out.view({batch_size, seqlen_q_ori, num_q_heads_per_hk, num_heads_k, head_size_v}).transpose(2, 3)
            .reshape({batch_size, seqlen_q_ori, num_heads_q, head_size_v});
    softmax_lse = softmax_lse.view({batch_size, num_heads_k, seqlen_q_ori, num_q_heads_per_hk}).transpose(2, 3)
            .reshape({batch_size, num_heads_q, seqlen_q_ori});

    return {out, softmax_lse};
}


inline int int64_stride_to_int(int64_t orig_stride) {
    if (orig_stride > std::numeric_limits<int>::max()) {
        TORCH_CHECK(false, "[Sparse TopK Attention] Stride exceeds int32 limit: ", orig_stride);
    }
    return static_cast<int>(orig_stride);
}

std::vector<at::Tensor> sparse_prefill_fwd(
    const at::Tensor &q,
    const at::Tensor &kv,
    const at::Tensor &indices,
    const at::Tensor &win_kv,
    const at::Tensor &win_indices,
    const std::optional<at::Tensor> &sinks,
    float sm_scale,
    int d_v
) {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm90 = dprops->major == 9;
    bool is_sm100 = dprops->major == 10;
    TORCH_CHECK(is_sm90 || is_sm100, "Sparse Attention Forward Kernel (sparse_prefill_fwd) is only supported on SM90 or SM100 architectures");

    CHECK_DEVICE(q);
    CHECK_DEVICE(kv);
    CHECK_DEVICE(indices);

    TORCH_CHECK(q.dtype() == torch::kBFloat16);
    TORCH_CHECK(kv.dtype() == torch::kBFloat16);
    TORCH_CHECK(indices.dtype() == torch::kInt32);

    int s_q = q.size(0);
    int s_kv = kv.size(0);
    int win_s_kv = win_kv.size(0);
    int h_q = q.size(1);
    int h_kv = kv.size(1);
    int d_qk = q.size(2);
    int topk = indices.size(2);

    CHECK_SHAPE(q, s_q, h_q, d_qk);
    CHECK_SHAPE(kv, s_kv, h_kv, d_qk);
    CHECK_SHAPE(indices, s_q, h_kv, topk);
    CHECK_SHAPE(win_kv, win_s_kv, h_kv, d_qk);
    // Fix me: hardcode the window size to 128
    CHECK_SHAPE(win_indices, s_q, h_kv, 128);
    // It can be none when there are no sinks
    if (sinks.has_value()) {
        TORCH_CHECK(sinks.value().dtype() == torch::kFloat32);
        CHECK_SHAPE(sinks.value(), h_q);
    }
    float *sinks_ptr = sinks.has_value() ? sinks.value().data_ptr<float>() : nullptr;

    TORCH_CHECK(q.stride(-1) == 1);
    TORCH_CHECK(kv.stride(-1) == 1);
    TORCH_CHECK(indices.stride(-1) == 1);

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto opts = q.options();
    at::Tensor out = torch::empty({s_q, h_q, d_v}, opts);
    CHECK_CONTIGUOUS(out);
    
    at::Tensor buf_attn_score, max_logits, lse, p_sum;
    max_logits = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat));
    lse = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat));
    CHECK_CONTIGUOUS(max_logits);
    CHECK_CONTIGUOUS(lse);

    SparsePrefillParams params = {
        s_q, s_kv, win_s_kv, h_q, h_kv, d_qk, d_v, topk,
        sm_scale, sm_scale * 1.44269504f,

        (cutlass::bfloat16_t*)q.data_ptr(),
        (cutlass::bfloat16_t*)kv.data_ptr(),
        (cutlass::bfloat16_t*)win_kv.data_ptr(),
        (int*)indices.data_ptr(),
        (int*)win_indices.data_ptr(),
        nullptr,
        nullptr,

        sinks_ptr,

        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)),
        int64_stride_to_int(kv.stride(0)), int64_stride_to_int(kv.stride(1)),
        int64_stride_to_int(win_kv.stride(0)), int64_stride_to_int(win_kv.stride(1)),
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),

        (cutlass::bfloat16_t*)out.data_ptr(),
        (float*)max_logits.data_ptr(),
        (float*)lse.data_ptr(),

        // Split-KV related tensors
        nullptr,
        0,
        nullptr,
        0,
        nullptr,
        nullptr,

        // Combine kernel compatibility fields
        s_q,
        h_q / h_kv,
        1,
        nullptr,
        nullptr,
        (SparsePrefillParams::index_t)(s_q * d_v),
        (SparsePrefillParams::index_t)(d_v),
        (SparsePrefillParams::index_t)(d_v),

        at::cuda::getCurrentCUDAStream().stream()
    };

    if (is_sm90) {
        sm90::run_fwd_kernel(params);
    } else if (is_sm100) {
        sm100::run_fwd_kernel(params);
    } else {
        TORCH_CHECK(false, "Unknown architecture");
    }

    return {out, max_logits, lse};
}

at::Tensor swa_sparse_fwd(
    const at::Tensor &q,
    const at::Tensor &win_kvcache,
    const at::Tensor &win_block_ids,
    const at::Tensor &win_topk_ids,
    const std::optional<at::Tensor> &compress_kvcache,
    const std::optional<at::Tensor> &compress_block_ids,
    const std::optional<at::Tensor> &compress_topk_ids,
    const at::Tensor &cu_seqlens_q,
    const std::optional<at::Tensor> &sinks,
    const std::optional<at::Tensor> &prefill_topk_length,   // [s_q] - per-query topk length for prefill, optional. this should include swa_size.
    const std::optional<at::Tensor> &decode_topk_length,   // [num_batch] - per-request topk length, optional. this should include swa_size.
    const std::optional<at::Tensor> &tile_scheduler_metadata,   // num_sm_parts x TileSchedulerMetaDataSize
    const std::optional<at::Tensor> &num_splits,                // batch_size + 1
    const std::optional<at::Tensor> &output = std::nullopt,
    const bool is_decode = false,
    float sm_scale = 1.0
) {

    CHECK_DEVICE(q);
    CHECK_DEVICE(win_kvcache);
    CHECK_DEVICE(win_block_ids);
    CHECK_DEVICE(cu_seqlens_q);

    TORCH_CHECK(q.dtype() == torch::kBFloat16);
    TORCH_CHECK(win_kvcache.dtype() == torch::kBFloat16);
    TORCH_CHECK(win_block_ids.dtype() == torch::kInt32);
    TORCH_CHECK(win_topk_ids.dtype() == torch::kInt32);
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32);

    // Check ndims
    TORCH_CHECK(q.dim() == 3);
    TORCH_CHECK(win_kvcache.dim() == 4);
    TORCH_CHECK(win_block_ids.dim() == 2);
    TORCH_CHECK(win_topk_ids.dim() == 2);
    TORCH_CHECK(cu_seqlens_q.dim() == 1);

    // There can be empty sinks.
    if (sinks.has_value()) {
        CHECK_DEVICE(sinks.value());
        TORCH_CHECK(sinks.value().dtype() == torch::kFloat32);
    }

    // If it is decode, then tile_scheduler_metadata and num_splits must be provided.
    if (is_decode) {
        TORCH_CHECK(tile_scheduler_metadata.has_value(), "tile_scheduler_metadata must be provided when is_decode is True");
        TORCH_CHECK(num_splits.has_value(), "num_splits must be provided when is_decode is True");
        CHECK_DEVICE(tile_scheduler_metadata.value());
        CHECK_DEVICE(num_splits.value());
        TORCH_CHECK(tile_scheduler_metadata.value().dtype() == torch::kInt32);
        TORCH_CHECK(num_splits.value().dtype() == torch::kInt32);
        CHECK_CONTIGUOUS(tile_scheduler_metadata.value());
        CHECK_CONTIGUOUS(num_splits.value());
    }

    // There can be empty compress_kvcache, compress_block_ids, compress_topk_ids
    bool has_compress_kvcache = compress_kvcache.has_value();
    if (has_compress_kvcache) {
        CHECK_DEVICE(compress_kvcache.value());
        CHECK_DEVICE(compress_block_ids.value());
        CHECK_DEVICE(compress_topk_ids.value());
        TORCH_CHECK(compress_kvcache.value().dtype() == torch::kBFloat16);
        TORCH_CHECK(compress_block_ids.value().dtype() == torch::kInt32);
        TORCH_CHECK(compress_topk_ids.value().dtype() == torch::kInt32);
        TORCH_CHECK(compress_kvcache.value().dim() == 4);
        TORCH_CHECK(compress_block_ids.value().dim() == 2);
        TORCH_CHECK(compress_topk_ids.value().dim() == 2);
    }

    // The device.
    int device = at::cuda::current_device();
    // Get number of SMs from device attributes
    int num_sms;
    cudaError_t res = cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
    if (res != cudaSuccess) {
        TORCH_CHECK(false, "Failed to get number of SMs from device attributes");
    }

    // paged kv block size.
    int block_size = win_kvcache.size(1);
    TORCH_CHECK(win_kvcache.size(1) == block_size);

    // Max number of window blocks.
    int max_win_blocks = win_block_ids.size(1);
    // Max number of compress blocks.
    int max_compress_blocks = has_compress_kvcache ? compress_block_ids.value().size(1) : 0;

    // batch size.
    int b = win_block_ids.size(0);
    // Q [num_total_tokens_q, num_head_q, num_dim]
    int total_s_q = q.size(0);
    // SWA KV [num_win_blocks, block_size, num_head_kv, num_dim]
    int win_s_kv = win_kvcache.size(0) * win_kvcache.size(1);
    // Compress KV [num_compress_blocks, block_size, num_head_kv, num_dim]
    int s_kv = has_compress_kvcache ? compress_kvcache.value().size(0) * compress_kvcache.value().size(1) : 0;
    int h_q = q.size(1);
    int h_kv = win_kvcache.size(2);
    int d_qk = q.size(2);
    int swa_size = win_topk_ids.size(1);
    int topk = has_compress_kvcache ? compress_topk_ids.value().size(1) : 0;
    int d_v = win_kvcache.size(-1);
    // It can be none when there are no sinks
    if (sinks.has_value()) {
        CHECK_SHAPE(sinks.value(), h_q);
    }
    float *sinks_ptr = sinks.has_value() ? sinks.value().data_ptr<float>() : nullptr;

    // prefill_topk_length is optional - if not provided, use topk for all queries
    if (prefill_topk_length.has_value()) {
        CHECK_DEVICE(prefill_topk_length.value());
        TORCH_CHECK(prefill_topk_length.value().dtype() == torch::kInt32);
        CHECK_SHAPE(prefill_topk_length.value(), total_s_q);
        CHECK_CONTIGUOUS(prefill_topk_length.value());
    }
    int *prefill_topk_length_ptr = prefill_topk_length.has_value() ? prefill_topk_length.value().data_ptr<int>() : nullptr;

    // decode_topk_length is optional - if not provided, use topk for all requests
    if (decode_topk_length.has_value()) {
        CHECK_DEVICE(decode_topk_length.value());
        TORCH_CHECK(decode_topk_length.value().dtype() == torch::kInt32);
        CHECK_SHAPE(decode_topk_length.value(), b);
        CHECK_CONTIGUOUS(decode_topk_length.value());
    }
    int *decode_topk_length_ptr = decode_topk_length.has_value() ? decode_topk_length.value().data_ptr<int>() : nullptr;

    TORCH_CHECK(q.stride(-1) == 1);
    TORCH_CHECK(win_kvcache.stride(-1) == 1);
    TORCH_CHECK(win_topk_ids.stride(-1) == 1);
    if (has_compress_kvcache) {
        TORCH_CHECK(compress_kvcache.value().stride(-1) == 1);
        TORCH_CHECK(compress_block_ids.value().stride(-1) == 1);
        TORCH_CHECK(compress_topk_ids.value().stride(-1) == 1);
    }

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto opts = q.options();
    at::Tensor out;
    if (output.has_value()) {
        TORCH_CHECK(output.value().dtype() == torch::kBFloat16);
        CHECK_SHAPE(output.value(), total_s_q, h_q, d_v);
        out = output.value();
    } else {
        out = torch::empty({total_s_q, h_q, d_v}, opts);
    }
    CHECK_CONTIGUOUS(out);

    // Allocate buffers for compressed global indices with shape of [total_s_q, topk or win_topk_ids]
    at::Tensor swa_global_indices = torch::empty({total_s_q, swa_size}, opts.dtype(torch::kInt32));
    CHECK_CONTIGUOUS(swa_global_indices);
    at::Tensor compress_global_indices;
    if (has_compress_kvcache) {
        compress_global_indices = torch::empty({total_s_q, topk}, opts.dtype(torch::kInt32));
    } else {
        compress_global_indices = torch::empty({0}, opts.dtype(torch::kInt32));
    }
    CHECK_CONTIGUOUS(compress_global_indices);

    const int q_head_per_hk = h_q / h_kv;
    const int s_q = cutlass::ceil_div(total_s_q, b);
    const int max_s_q = is_decode ? s_q : total_s_q;
    const int q_seq_per_hk = s_q * q_head_per_hk;

    // Convert SWA and compress indices to global indices
    sm90::convert_swa_compress_to_global_ids(
        (int*)win_block_ids.data_ptr(),
        (int*)win_topk_ids.data_ptr(),
        (int*)swa_global_indices.data_ptr(),
        (int*)(has_compress_kvcache ? compress_block_ids.value().data_ptr() : nullptr),
        (int*)(has_compress_kvcache ? compress_topk_ids.value().data_ptr() : nullptr),
        (int*)compress_global_indices.data_ptr(),
        (int*)cu_seqlens_q.data_ptr(),
        total_s_q, b, max_s_q, swa_size, topk, block_size, max_win_blocks, max_compress_blocks, num_sms, at::cuda::getCurrentCUDAStream().stream()
    );

    // print swa_global_indices and compress_global_indices torch tensors.
    // printf("swa_global_indices \n");
    // for (int i = 0; i < s_q; i++)
    // {
    //     std::cout << swa_global_indices[i] << std::endl;
    // }

    // printf("topk_ids \n");
    // std::cout << compress_topk_ids << std::endl;
    // printf("compress_global_indices \n");
    // std::cout << compress_global_indices << std::endl;

    // at::Tensor buf_attn_score, max_logits, lse, p_sum;
    // max_logits = torch::empty({total_s_q, h_q}, opts.dtype(torch::kFloat));
    // lse = torch::empty({total_s_q, h_q}, opts.dtype(torch::kFloat));
    // CHECK_CONTIGUOUS(max_logits);
    // CHECK_CONTIGUOUS(lse);

    // Allocate split-KV accumulator tensors
    int num_sm_parts = 0;
    int total_num_splits = 0;
    at::Tensor softmax_lse_accum;
    at::Tensor out_accum;
    if (is_decode) {
        num_sm_parts = tile_scheduler_metadata.value().size(0);
        total_num_splits = b + num_sm_parts;
        softmax_lse_accum = torch::empty({total_num_splits, q_seq_per_hk}, opts.dtype(at::kFloat));
        out_accum = torch::empty({total_num_splits, q_seq_per_hk, d_v}, opts.dtype(at::kFloat));
        CHECK_CONTIGUOUS(softmax_lse_accum);
        CHECK_CONTIGUOUS(out_accum);
    } else {
        num_sm_parts = num_sms;
        softmax_lse_accum = torch::empty({0}, opts.dtype(at::kFloat));
        out_accum = torch::empty({0}, opts.dtype(at::kFloat));
    }

    // The stream
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    SparsePrefillParams params = {
        is_decode ? s_q : total_s_q, s_kv, win_s_kv, h_q, h_kv, d_qk, d_v, topk,
        sm_scale, sm_scale * 1.44269504f,

        (cutlass::bfloat16_t*)q.data_ptr(),
        (cutlass::bfloat16_t*)(has_compress_kvcache ? compress_kvcache.value().data_ptr() : nullptr),
        (cutlass::bfloat16_t*)win_kvcache.data_ptr(),
        (int*)compress_global_indices.data_ptr(),
        (int*)swa_global_indices.data_ptr(),
        prefill_topk_length_ptr,
        decode_topk_length_ptr,

        sinks_ptr,

        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)),
        has_compress_kvcache ? int64_stride_to_int(compress_kvcache.value().stride(1)) : 0,
        has_compress_kvcache ? int64_stride_to_int(compress_kvcache.value().stride(2)) : 0,
        int64_stride_to_int(win_kvcache.stride(1)), int64_stride_to_int(win_kvcache.stride(2)),
        has_compress_kvcache ? int64_stride_to_int(compress_global_indices.stride(0)) : 0,
        has_compress_kvcache ? int64_stride_to_int(compress_global_indices.stride(1)) : 0,
        
        (cutlass::bfloat16_t*)out.data_ptr(),
        (float*)nullptr,
        (float*)nullptr,

        // Split-KV related fields
        is_decode ? tile_scheduler_metadata.value().data_ptr<int>() : nullptr,   // tile_scheduler_metadata_ptr
        num_sm_parts,                              // num_sm_parts
        is_decode ? num_splits.value().data_ptr<int>() : nullptr,                // num_splits_ptr
        total_num_splits,                          // total_num_splits
        softmax_lse_accum.data_ptr<float>(),       // softmax_lseaccum_ptr
        out_accum.data_ptr<float>(),               // oaccum_ptr

        // Combine kernel compatibility fields
        b,                                         // b (single batch for combine)
        q_seq_per_hk,                              // q_seq_per_hk
        h_kv,                                      // h_k
        (void*)nullptr,                            // softmax_lse_ptr (output lse, set if needed)
        (void*)out.data_ptr(),                     // o_ptr
        (SparsePrefillParams::index_t)(q_seq_per_hk * d_v),  // o_batch_stride
        (SparsePrefillParams::index_t)(d_v),         // o_head_stride
        (SparsePrefillParams::index_t)(d_v),         // o_row_stride

        stream
    };

    // If it is decode, then use split-KV kernel. Otherwise, use normal kernel.
    if (is_decode) {
        sm90::run_fwd_splitkv_kernel(params);
        // Call mla_combine kernel afterwards to merge results from all partitions
        run_flash_mla_combine_kernel<cutlass::bfloat16_t, SparsePrefillParams>(params, stream);
    } else {
        sm90::run_fwd_kernel(params);
    }

    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA";
    m.def("get_mla_decoding_metadata", &get_mla_decoding_metadata);
    m.def("fwd_kvcache_mla", &fwd_kvcache_mla);
    m.def("dense_prefill_fwd", &FMHACutlassSM100FwdRun);
    m.def("dense_prefill_bwd", &FMHACutlassSM100BwdRun);
    m.def("sparse_prefill_fwd", &sparse_prefill_fwd);
    m.def("swa_sparse_fwd", &swa_sparse_fwd);
}
