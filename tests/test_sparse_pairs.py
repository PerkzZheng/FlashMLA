import pytest
import torch

import flash_mla
import lib
import quant


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="Paired attention requires Blackwell",
)


def make_indices(batch, queries, topk, domain, pattern="random"):
    rows = lib._randperm_batch(
        batch * queries, torch.full((batch * queries,), domain, dtype=torch.int32), topk, [-1]
    ).view(batch, queries, topk)
    if pattern == "identical":
        rows[:, 1::2] = rows[:, ::2]
    elif pattern == "disjoint":
        rows[:, ::2] = torch.arange(topk, dtype=torch.int32)
        rows[:, 1::2] = torch.arange(topk, 2 * topk, dtype=torch.int32)
    elif pattern == "empty":
        rows.fill_(-1)
    rows += torch.arange(batch, dtype=torch.int32).view(-1, 1, 1) * domain
    if pattern == "empty":
        rows.fill_(-1)
    return rows.cuda()


def assert_metadata(indices, metadata, lengths=None):
    batch, queries, topk = indices.shape
    source = indices.cpu()
    limits = lengths.cpu() if lengths is not None else torch.full((batch, queries), topk)
    packed = metadata.packed.cpu().view(batch, queries // 2, -1)
    counts = metadata.lengths.cpu().view(batch, queries // 2)
    for request in range(batch):
        for pair in range(queries // 2):
            expected = {}
            for query in range(2):
                row = pair * 2 + query
                for index in source[request, row, :max(0, min(topk, int(limits[request, row])))].tolist():
                    if 0 <= index < metadata.kv_bound:
                        expected[index] = expected.get(index, 0) | (1 << query)
            entries = packed[request, pair, :counts[request, pair]].to(torch.int64)
            decoded = {int(entry >> 2): int(entry & 3) for entry in entries}
            assert decoded == expected
            assert len(entries) == len(expected)
            assert entries.tolist() == sorted(entries.tolist())
            assert (packed[request, pair, counts[request, pair]:] == -1).all()


@pytest.mark.parametrize("algorithm", ["sort", "merge", "bitmap"])
@pytest.mark.parametrize("topk", [1, 31, 64, 128, 576, 2048, 4096])
@pytest.mark.parametrize("pattern", ["random", "identical", "disjoint", "empty"])
def test_pair_metadata(algorithm, topk, pattern):
    torch.manual_seed(177)
    indices = make_indices(2, 8, topk, 65536, pattern)
    lengths = torch.full((2, 8), topk, dtype=torch.int32, device="cuda")
    lengths[0, 0] = 0
    lengths[1, 3] = topk // 2
    metadata = flash_mla.prepare_sparse_pairs(indices, 2 * 65536, algorithm, 65536, lengths)
    assert_metadata(indices, metadata, lengths)


def reference(q, kv, indices, lengths=None, sink=None):
    batch, queries, heads, dimension = q.shape
    outputs, normalizers = [], []
    for request in range(batch):
        for query in range(queries):
            selected = indices[request, query]
            if lengths is not None:
                selected = selected[:max(0, min(indices.shape[-1], int(lengths[request, query])))]
            selected = selected[(selected >= 0) & (selected < kv.shape[0])]
            if selected.numel() == 0:
                outputs.append(torch.zeros((heads, dimension), device=q.device))
                normalizers.append(torch.full((heads,), float("inf"), device=q.device))
                continue
            values = kv[selected.long()].view(-1, dimension).float()
            logits = q[request, query].float() @ values.T * dimension ** -0.5
            normalizer = logits.logsumexp(-1)
            denominator = torch.logaddexp(normalizer, sink) if sink is not None else normalizer
            outputs.append((logits - denominator[:, None]).exp() @ values)
            normalizers.append(normalizer)
    return torch.stack(outputs).view_as(q), torch.stack(normalizers).view(batch, queries, heads)


@pytest.mark.parametrize("algorithm", ["sort", "merge", "bitmap"])
@pytest.mark.parametrize("mode", ["prefill", "decode", "decode_no_split"])
@pytest.mark.parametrize("pattern", ["random", "identical", "disjoint", "empty"])
@pytest.mark.parametrize("topk", [64, 512, 2048])
def test_pair_attention(algorithm, mode, pattern, topk):
    torch.manual_seed(891)
    batch, queries, domain = 2, 8, max(2048, 2 * topk)
    q = torch.randn((batch, queries, 64, 512), dtype=torch.bfloat16, device="cuda")
    kv = torch.randn((batch * domain, 1, 512), dtype=torch.bfloat16, device="cuda") * 0.2
    indices = make_indices(batch, queries, topk, domain, pattern)
    lengths = torch.full((batch, queries), topk, dtype=torch.int32, device="cuda")
    lengths[0, 0] = 0
    lengths[0, 1] = 1
    lengths[1, 2] = topk // 2
    sink = torch.randn((64,), device="cuda", dtype=torch.float32)
    sink[0], sink[1] = float("inf"), float("-inf")
    metadata = flash_mla.prepare_sparse_pairs(indices, batch * domain, algorithm, domain, lengths)
    if mode == "prefill":
        output, _, lse = flash_mla.flash_mla_sparse_fwd_paired(q, kv, metadata, attn_sink=sink)
    else:
        cache = quant.quantize_k_cache(kv.view(-1, 64, 1, 512), quant.KVCacheLayout.V41_FP8Sparse)
        kv = quant.dequantize_k_cache(cache, quant.KVCacheLayout.V41_FP8Sparse).view(-1, 1, 512)
        output, lse = flash_mla.flash_mla_with_kvcache_paired(
            q, cache, metadata, attn_sink=sink, force_no_split=mode == "decode_no_split"
        )
        lse = lse.transpose(1, 2)
    expected, expected_lse = reference(q, kv, indices, lengths, sink)
    torch.testing.assert_close(output.float(), expected, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(lse, expected_lse, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("algorithm", ["sort", "merge", "bitmap"])
@pytest.mark.parametrize("mode", ["prefill", "decode", "decode_no_split"])
@pytest.mark.parametrize("topk", [512, 2048])
def test_pair_graph_replay_changes(algorithm, mode, topk):
    torch.manual_seed(934)
    q = torch.randn((1, 8, 64, 512), dtype=torch.bfloat16, device="cuda")
    domain = max(2048, 2 * topk)
    kv = torch.randn((domain, 1, 512), dtype=torch.bfloat16, device="cuda") * 0.2
    indices = make_indices(1, 8, topk, domain, "identical")
    lengths = torch.full((1, 8), topk, dtype=torch.int32, device="cuda")
    metadata = flash_mla.prepare_sparse_pairs(indices, domain, algorithm, lengths=lengths)
    cache = quant.quantize_k_cache(kv.view(-1, 64, 1, 512), quant.KVCacheLayout.V41_FP8Sparse)
    if mode != "prefill":
        kv = quant.dequantize_k_cache(cache, quant.KVCacheLayout.V41_FP8Sparse).view(-1, 1, 512)

    def run():
        metadata.update(indices, lengths)
        if mode == "prefill":
            output, _, lse = flash_mla.flash_mla_sparse_fwd_paired(q, kv, metadata)
            return output, lse
        output, lse = flash_mla.flash_mla_with_kvcache_paired(
            q, cache, metadata, force_no_split=mode == "decode_no_split"
        )
        return output, lse.transpose(1, 2)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, lse = run()
    for pattern in ("disjoint", "empty", "random", "identical"):
        indices.copy_(make_indices(1, 8, topk, domain, pattern))
        lengths[0, 0] = 0
        graph.replay()
        assert_metadata(indices, metadata, lengths)
        expected, expected_lse = reference(q, kv, indices, lengths)
        torch.testing.assert_close(output.float(), expected, atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(lse, expected_lse, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("domain", [1, 31, 32, 33, 511, 65535, 65536])
@pytest.mark.parametrize("topk", [1, 31, 128, 4096])
def test_bitmap_compaction_boundaries(domain, topk):
    indices = torch.full((1, 2, topk), -1, dtype=torch.int32, device="cuda")
    count = min(domain, topk)
    indices[0, 0, :count] = torch.arange(count, device="cuda", dtype=torch.int32)
    indices[0, 1, :count] = torch.arange(domain - count, domain, device="cuda", dtype=torch.int32)
    metadata = flash_mla.prepare_sparse_pairs(indices, domain, "bitmap")
    assert_metadata(indices, metadata)
    indices.fill_(-1)
    metadata.update(indices)
    assert_metadata(indices, metadata)


@pytest.mark.parametrize("groups", [1, 3, 17, 75, 76, 129])
@pytest.mark.parametrize("pattern", ["ragged", "empty"])
def test_pair_scheduler_boundaries(groups, pattern):
    torch.manual_seed(814)
    indices = make_indices(1, groups * 2, 2048, 8192)
    lengths = torch.randint(0, 2049, (1, groups * 2), dtype=torch.int32, device="cuda")
    lengths[:, ::3] = 0
    if pattern == "empty":
        lengths.zero_()
    metadata = flash_mla.prepare_sparse_pairs(indices, 8192, "bitmap", lengths=lengths)
    query = torch.randn((1, groups * 2, 64, 512), device="cuda", dtype=torch.bfloat16)
    original = torch.randn((128, 64, 1, 512), device="cuda", dtype=torch.bfloat16) * 0.1
    cache = quant.quantize_k_cache(original, quant.KVCacheLayout.V41_FP8Sparse)
    observed = flash_mla.flash_mla_with_kvcache_paired(query, cache, metadata)
    expected = flash_mla.flash_mla_with_kvcache_paired(query, cache, metadata, split_policy="legacy")
    torch.testing.assert_close(observed[0], expected[0], atol=3e-4, rtol=2e-2)
    torch.testing.assert_close(observed[1], expected[1], atol=2e-4, rtol=2e-4)


def test_pair_input_guards():
    with pytest.raises(ValueError, match="even"):
        flash_mla.prepare_sparse_pairs(torch.zeros((1, 3, 64), dtype=torch.int32, device="cuda"), 64)
    with pytest.raises(ValueError, match="algorithm"):
        flash_mla.prepare_sparse_pairs(torch.zeros((1, 2, 64), dtype=torch.int32, device="cuda"), 64, "unknown")
    with pytest.raises(RuntimeError, match="domain"):
        flash_mla.prepare_sparse_pairs(torch.zeros((1, 2, 64), dtype=torch.int32, device="cuda"), 65537, "bitmap")


@pytest.mark.parametrize("algorithm", ["sort", "merge", "bitmap"])
@pytest.mark.parametrize("topk", [1, 31, 65])
@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_unaligned_topk_and_clamped_lengths(algorithm, topk, mode):
    torch.manual_seed(471)
    indices = make_indices(2, 4, topk, 1024)
    lengths = torch.full((2, 4), topk, dtype=torch.int32, device="cuda")
    lengths[0, 0] = -3
    lengths[1, 1] = topk + 17
    query = torch.randn((2, 4, 64, 512), dtype=torch.bfloat16, device="cuda")
    values = torch.randn((2048, 1, 512), dtype=torch.bfloat16, device="cuda") * 0.1
    metadata = flash_mla.prepare_sparse_pairs(indices, 2048, algorithm, 1024, lengths)
    assert_metadata(indices, metadata, lengths)
    if mode == "prefill":
        output, _, lse = flash_mla.flash_mla_sparse_fwd_paired(query, values, metadata)
    else:
        cache = quant.quantize_k_cache(values.view(-1, 64, 1, 512), quant.KVCacheLayout.V41_FP8Sparse)
        values = quant.dequantize_k_cache(cache, quant.KVCacheLayout.V41_FP8Sparse).view(-1, 1, 512)
        output, lse = flash_mla.flash_mla_with_kvcache_paired(query, cache, metadata)
        lse = lse.transpose(1, 2)
    expected, expected_lse = reference(query, values, indices, lengths)
    torch.testing.assert_close(output.float(), expected, atol=3e-4, rtol=2e-2)
    torch.testing.assert_close(lse, expected_lse, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("algorithm", ["sort", "merge", "bitmap"])
def test_invalid_indices(algorithm):
    indices = make_indices(1, 8, 64, 1024)
    indices[:, :, :5] = torch.tensor([-2147483648, -123456, -1, 1024, 2147483647], device="cuda")
    metadata = flash_mla.prepare_sparse_pairs(indices, 1024, algorithm)
    assert_metadata(indices, metadata)


def test_large_pair_batch_auto_policy():
    groups = 4097
    indices = torch.full((1, groups * 2, 2048), -1, device="cuda", dtype=torch.int32)
    indices[..., 0] = 0
    metadata = flash_mla.prepare_sparse_pairs(indices, 64, "bitmap")
    query = torch.zeros((1, groups * 2, 64, 512), device="cuda", dtype=torch.bfloat16)
    original = torch.full((2, 64, 1, 512), 0.125, device="cuda", dtype=torch.bfloat16)
    cache = quant.quantize_k_cache(original, quant.KVCacheLayout.V41_FP8Sparse)
    output, lse = flash_mla.flash_mla_with_kvcache_paired(query, cache, metadata)
    assert torch.all(output == 0.125)
    torch.testing.assert_close(lse, torch.zeros_like(lse), atol=1e-6, rtol=0)
    with pytest.raises(ValueError, match="force_no_split"):
        flash_mla.flash_mla_with_kvcache_paired(query, cache, metadata, split_policy="legacy")
    with pytest.raises(ValueError, match="split_policy"):
        flash_mla.flash_mla_with_kvcache_paired(query, cache, metadata, split_policy="invalid")


@pytest.fixture
def bf16_defaults():
    original = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        yield
    finally:
        torch.set_default_dtype(original)


@pytest.mark.parametrize("heads", [64, 128])
@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_ungrouped_regression(heads, mode, bf16_defaults):
    import kernelkit as kk
    import ref

    with torch.device("cuda"):
        if mode == "decode":
            parameters = lib.RawTestParamForDecode(
                4, heads, 3, 1, 2048, True, 576, d_qk=512,
                have_topk_length=True, seed=99, num_runs=0,
            ).to_test_param()
            testcase = lib.generate_testcase_for_decode(parameters)
            scheduler, _ = flash_mla.get_mla_metadata()
            output, lse = lib.run_flash_mla_decode(parameters, testcase, scheduler, None)
            expected, expected_lse = ref.ref_sparse_attn_decode(parameters, testcase)
        else:
            parameters = lib.TestParam(8, 2048, 576, h_q=heads, have_attn_sink=True, have_topk_length=True, seed=99)
            testcase = lib.generate_testcase(parameters)
            output, _, lse = flash_mla.flash_mla_sparse_fwd(
                testcase.q, testcase.kv, testcase.indices, testcase.sm_scale,
                attn_sink=testcase.attn_sink, topk_length=testcase.topk_length,
            )
            expected, _, _, expected_lse = ref.ref_sparse_attn_fwd(parameters, testcase)
        assert kk.check_is_allclose("output", output, expected, abs_tol=1e-3, rel_tol=1/128, cos_diff_tol=1e-5)
        assert kk.check_is_allclose("lse", lse, expected_lse, abs_tol=1e-5, rel_tol=1/128, cos_diff_tol=1e-5)


@pytest.mark.parametrize("algorithm", ["sort", "merge"])
def test_paired_shuffled_physical_pages(algorithm, bf16_defaults):
    import ref

    with torch.device("cuda"):
        parameters = lib.RawTestParamForDecode(
            4, 64, 8, 1, 2048, True, 128, d_qk=512, block_size=61,
            have_topk_length=True, seed=314, num_runs=0,
        ).to_test_param()
        testcase = lib.generate_testcase_for_decode(parameters)
        cache = testcase.kv_scope.get_kvcache_for_flash_mla()
        indices = testcase.kv_scope.indices_in_kvcache.contiguous()
        lengths = testcase.kv_scope.topk_length[:, None].expand(4, 8).contiguous()
        metadata = flash_mla.prepare_sparse_pairs(indices, cache.shape[0] * cache.shape[1], algorithm, lengths=lengths)
        output, lse = flash_mla.flash_mla_with_kvcache_paired(
            testcase.q.contiguous(), cache, metadata, testcase.sm_scale, testcase.attn_sink
        )
        expected, expected_lse = ref.ref_sparse_attn_decode(parameters, testcase)
        torch.testing.assert_close(output, expected, atol=3e-4, rtol=2e-2)
        torch.testing.assert_close(lse, expected_lse, atol=2e-4, rtol=2e-4)
