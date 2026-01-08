#pragma once

#include <cuda_runtime.h>

namespace sm90 {

/**
 * Convert block-based indices (paged KV cache) to global indices for both SWA and compress KV
 * 
 * This function converts indices in the format:
 *   indices[i][j] = page_block_index * block_size + offset_within_block
 * to global indices:
 *   global_ids[i][j] = block_table[batch_idx][page_block_index] * block_size + offset_within_block
 * 
 * @param win_block_ids: [batch_size, max_win_blocks], int32 - Block table for sliding window KV cache
 * @param win_topk_ids: [num_tokens, win_topk], int32 - Encoded indices for window (page_block_idx * block_size + offset)
 * @param win_global_ids: [num_tokens, win_topk], int32 - Output global indices for window
 * @param compress_block_ids: [batch_size, max_compress_blocks], int32 - Block table for compress KV cache
 * @param compress_topk_ids: [num_tokens, compress_topk], int32 - Encoded indices for compress (page_block_idx * block_size + offset)
 * @param compress_global_ids: [num_tokens, compress_topk], int32 - Output global indices for compress
 * @param cu_seqlens: [batch_size + 1], int32 - Cumulative sequence lengths
 * @param num_tokens: Total number of query tokens
 * @param batch_size: Number of batches
 * @param win_topk: Number of window indices per token
 * @param compress_topk: Number of compress indices per token
 * @param block_size: Size of each block in the paged KV cache (typically 64)
 * @param max_win_blocks: Maximum number of window blocks per batch
 * @param max_compress_blocks: Maximum number of compress blocks per batch
 * @param num_sms: Number of SMs
 * @param stream: CUDA stream
 */
void convert_swa_compress_to_global_ids(
    const int* win_block_ids,
    const int* win_topk_ids,
    int* win_global_ids,
    const int* compress_block_ids,
    const int* compress_topk_ids,
    int* compress_global_ids,
    const int* cu_seqlens,
    int num_tokens,
    int batch_size,
    int win_topk,
    int compress_topk,
    int block_size,
    int max_win_blocks,
    int max_compress_blocks,
    int num_sms,
    cudaStream_t stream
);

} // namespace sm90

