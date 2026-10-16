#include "common.h"
#include "cuda_kernels/smxx/sparse_pair_union.h"

static void sparse_pair_union_out(
    const Tensor& indices, const std::optional<Tensor>& lengths,
    const Tensor& packed, const Tensor& union_lengths, const Tensor& scratch,
    int64_t kv_bound, int64_t request_stride, int64_t algorithm
) {
    KU_CHECK_NDIM(indices, 3);
    int64_t batch = indices.size(0), queries = indices.size(1), topk = indices.size(2);
    STD_TORCH_CHECK(batch > 0 && queries > 0 && queries % 2 == 0,
                    "Expected nonempty even-length query sequences");
    STD_TORCH_CHECK(batch <= INT32_MAX / queries, "Too many query rows");
    STD_TORCH_CHECK(topk > 0 && topk <= 4096, "Pair union supports top-k 1..4096");
    STD_TORCH_CHECK(kv_bound > 0 && kv_bound < (1 << 29), "Invalid KV index bound");
    STD_TORCH_CHECK(algorithm >= 0 && algorithm <= 2, "Unknown pair union algorithm");
    STD_TORCH_CHECK(request_stride >= 0 && request_stride <= INT32_MAX, "Invalid request stride");
    if (algorithm == 2) {
        int64_t domain = request_stride ? request_stride : kv_bound;
        STD_TORCH_CHECK(domain <= 65536, "Bitmap union requires a domain of at most 65536 tokens");
        STD_TORCH_CHECK(!request_stride || batch <= kv_bound / request_stride, "Request domains exceed KV bound");
    }
    int capacity = ((2 * topk + 63) / 64) * 64;
    int sorted_capacity = 256;
    while (sorted_capacity < topk) sorted_capacity *= 2;
    for (const auto& tensor : {indices, packed, union_lengths, scratch}) {
        KU_CHECK_DEVICE(tensor);
        KU_CHECK_DTYPE(tensor, ScalarType::Int);
        KU_CHECK_CONTIGUOUS(tensor);
        STD_TORCH_CHECK(tensor.get_device_index() == indices.get_device_index(), "All union tensors must use the same device");
    }
    KU_CHECK_DEVICE(lengths);
    KU_CHECK_DTYPE(lengths, ScalarType::Int);
    KU_CHECK_CONTIGUOUS(lengths);
    KU_CHECK_SHAPE(lengths, batch, queries);
    if (lengths) STD_TORCH_CHECK(lengths->get_device_index() == indices.get_device_index(), "Lengths must use the indices device");
    KU_CHECK_SHAPE(packed, batch * queries / 2, 1, capacity);
    KU_CHECK_SHAPE(union_lengths, batch * queries / 2);
    if (algorithm == 1) KU_CHECK_SHAPE(scratch, batch, queries, sorted_capacity);
    torch::stable::accelerator::DeviceGuard device_guard(indices.get_device_index());
    SparsePairUnionParams params{
        indices.const_data_ptr<int>(), ku::get_optional_tensor_ptr<int>(lengths),
        packed.mutable_data_ptr<int>(), union_lengths.mutable_data_ptr<int>(), scratch.mutable_data_ptr<int>(),
        static_cast<int>(batch), static_cast<int>(queries), static_cast<int>(topk), capacity, sorted_capacity,
        static_cast<int>(kv_bound), static_cast<int>(request_stride), static_cast<int>(algorithm), get_current_stream()
    };
    run_sparse_pair_union(params);
    cudaError_t status = cudaGetLastError();
    STD_TORCH_CHECK(status == cudaSuccess, "Sparse pair union launch failed: ", cudaGetErrorString(status));
}

STABLE_TORCH_LIBRARY_IMPL(flash_mla, CUDA, module) {
    module.impl("sparse_pair_union_out", TORCH_BOX(&sparse_pair_union_out));
}
