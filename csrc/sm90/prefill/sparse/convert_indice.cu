#include "convert_indice.h"
#include "utils.h"
#include <cstdio>

namespace sm90 {

/**
 * Fused CUDA kernel to convert both SWA and compress indices to global indices
 * 
 * This kernel processes both sliding window (SWA) and compress KV indices in a single launch.
 * The first num_tokens * win_topk threads process SWA indices, and the remaining threads
 * process compress indices.
 */

__global__ void convert_swa_compress_fused_kernel(
    const int* __restrict__ win_block_ids,
    const int* __restrict__ win_topk_ids,
    int* __restrict__ win_global_ids,
    const int* __restrict__ compress_block_ids,
    const int* __restrict__ compress_topk_ids,
    int* __restrict__ compress_global_ids,
    const int* __restrict__ cu_seqlens,
    int num_tokens,
    int batch_size,
    int win_topk,
    int compress_topk,
    int block_size_log2,
    int max_win_blocks,
    int max_compress_blocks
) {
    // Batch index.
    int batch_idx = blockIdx.y;

    // Return early if the batch is out of range
    if (batch_idx >= batch_size) {
        return;
    }

    cudaGridDependencySynchronize();

    // The token_offset is the offset of the token in the batch.
    int token_offset = cu_seqlens[batch_idx];
    // The seq_len_q.
    int seq_len_q = cu_seqlens[batch_idx + 1] - token_offset;

    // The block size.
    int block_size = 1 << block_size_log2;

    // Q for loop:
    for (int q_idx = blockIdx.x; q_idx < seq_len_q; q_idx += gridDim.x) {

        // KV for loop:
        for (int index_pos = threadIdx.x * 4; index_pos < win_topk + compress_topk; index_pos += (blockDim.x * 4)) {

            // Load four indices at a time.
            int4 topk_ids = make_int4(-1, -1, -1, -1);
            const int* block_table;
            int max_num_blocks;

            // The offset of the ids.
            int64_t ids_offset = 0;
            
            if (index_pos < win_topk) {
                ids_offset = (token_offset + q_idx) * int64_t(win_topk) + index_pos;
                topk_ids = *reinterpret_cast<const int4*>(win_topk_ids + ids_offset);
                block_table = win_block_ids;
                max_num_blocks = max_win_blocks;
            } else {
                ids_offset = (token_offset + q_idx) * int64_t(compress_topk) + index_pos - win_topk;
                // No guarantee that the compress_topk_ids is divisible by 4, so we need to load one by one.
                if (index_pos - win_topk < compress_topk) {
                    topk_ids.x = compress_topk_ids[ids_offset];
                }
                if (index_pos - win_topk + 1 < compress_topk) {
                    topk_ids.y = compress_topk_ids[ids_offset + 1];
                }
                if (index_pos - win_topk + 2 < compress_topk) {
                    topk_ids.z = compress_topk_ids[ids_offset + 2];
                }
                if (index_pos - win_topk + 3 < compress_topk) {
                    topk_ids.w = compress_topk_ids[ids_offset + 3];
                }
                block_table = compress_block_ids;
                max_num_blocks = max_compress_blocks;
            }
            
            // Handle invalid indices (marked as -1) and convert to global IDs
            int4 global_ids_vec = make_int4(-1, -1, -1, -1);
            
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int topk_id = reinterpret_cast<const int*>(&topk_ids)[i];
                if (topk_id >= 0) {
                    // Extract page block index and within-block offset from encoded_index
                    int page_block_index = topk_id >> block_size_log2;
                    int offset_within_block = topk_id & (block_size - 1);

                    // Check bounds
                    if (page_block_index < max_num_blocks) {
                        // Look up the physical block position from the block table
                        int physical_block_position = block_table[batch_idx * int64_t(max_num_blocks) + page_block_index];
                        
                        // Handle invalid block table entries and compute global index
                        if (physical_block_position >= 0) {
                            reinterpret_cast<int*>(&global_ids_vec)[i] = physical_block_position * block_size + offset_within_block;
                        }
                    }
                }
            }
            
            // Store the global ids.
            if (index_pos < win_topk) {
                reinterpret_cast<int4*>(win_global_ids + ids_offset)[0] = global_ids_vec;
            } else {
                // No guarantee that the compress_global_ids is divisible by 4, so we need to store one by one.
                if (index_pos - win_topk < compress_topk) {
                    compress_global_ids[ids_offset] = global_ids_vec.x;
                }
                if (index_pos - win_topk + 1 < compress_topk) {
                    compress_global_ids[ids_offset + 1] = global_ids_vec.y;
                }
                if (index_pos - win_topk + 2 < compress_topk) {
                    compress_global_ids[ids_offset + 2] = global_ids_vec.z;
                }
                if (index_pos - win_topk + 3 < compress_topk) {
                    compress_global_ids[ids_offset + 3] = global_ids_vec.w;
                }
            }
        }
    }

    cudaTriggerProgrammaticLaunchCompletion();
}

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
) {
    // Fixed threads per block for now.
    int threads_per_block = 256;

    // Each thread converts four indices at a time.
    dim3 grid((num_sms * 8) / batch_size, batch_size);
    
    // Assert block size is power of 2.
    FLASH_ASSERT((block_size & (block_size - 1)) == 0);
    // block_size_log2 - use bit scan instead of log2 for efficiency
    int block_size_log2 = __builtin_ctz(block_size);

    // Assert win_topk is divisible by 4.
    FLASH_ASSERT(win_topk % 4 == 0);

    // Use cudaLaunchKernelEx to enable PDL (Programmatic Dependent Launch)
    cudaLaunchAttribute kernel_attributes[1];
    kernel_attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    kernel_attributes[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t kernel_config = {
        grid,
        dim3(threads_per_block, 1, 1),
        0,  // dynamicSmemBytes
        stream,
        kernel_attributes,
        1
    };
    cudaLaunchKernelEx(&kernel_config, convert_swa_compress_fused_kernel,
        win_block_ids,
        win_topk_ids,
        win_global_ids,
        compress_block_ids,
        compress_topk_ids,
        compress_global_ids,
        cu_seqlens,
        num_tokens,
        batch_size,
        win_topk,
        compress_topk,
        block_size_log2,
        max_win_blocks,
        max_compress_blocks
    );
}

} // namespace sm90
