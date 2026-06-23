import unittest

import torch

import flash_mla
import lib
from lib import RawTestParamForDecode


class SparsePrefillVarlenClcTest(unittest.TestCase):
    def _set_up_cuda(self):
        torch.cuda.set_device(0)
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("sparse_prefill_varlen_clc is SM100-only")

        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.bfloat16)
        torch.set_float32_matmul_precision("high")

    def _assert_matches_decode_path(self, raw: RawTestParamForDecode, contiguous_inputs: bool):
        p = raw.to_test_param()
        t = lib.generate_testcase_for_decode(p)
        if contiguous_inputs:
            t.q = t.q.contiguous()
            t.kv_scope.indices_in_kvcache = t.kv_scope.indices_in_kvcache.contiguous()

        tile_scheduler_metadata, _ = flash_mla.get_mla_metadata()
        out_old, lse_old = lib.run_flash_mla_decode(p, t, tile_scheduler_metadata, None)

        q_flat = t.q.reshape(p.decode.b * p.s_q, p.h_q, p.d_qk)
        indices_flat = t.kv_scope.indices_in_kvcache.reshape(p.decode.b * p.s_q, p.topk)
        out_new, lse_new = flash_mla.flash_mla_sparse_prefill_varlen_clc(
            q_flat,
            t.kv_scope.get_kvcache_for_flash_mla(),
            indices_flat,
            head_dim_v=p.d_v,
            softmax_scale=t.sm_scale,
            attn_sink=t.attn_sink,
        )

        out_old_flat = out_old.reshape(p.decode.b * p.s_q, p.h_q, p.d_v)
        lse_old_flat = lse_old.transpose(1, 2).reshape(p.decode.b * p.s_q, p.h_q)
        self.assertTrue(torch.allclose(out_old_flat.float(), out_new.float(), atol=1e-3, rtol=2.01 / 128))
        self.assertTrue(torch.allclose(lse_old_flat.float(), lse_new.float(), atol=1e-6, rtol=8.01 / 65536))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    @torch.inference_mode()
    def test_matches_decode_path(self):
        self._set_up_cuda()

        cases = [
            (
                "v32_small_attn_sink_noncontiguous",
                RawTestParamForDecode(
                    b=4,
                    h_q=64,
                    s_q=3,
                    h_kv=1,
                    s_kv=1024,
                    is_varlen=True,
                    topk=64,
                    d_qk=576,
                    d_v=512,
                    enable_attn_sink=True,
                    check_correctness=True,
                    num_runs=0,
                    seed=20260622,
                ),
                False,
            ),
            (
                "v32_topk2048_no_attn_sink",
                RawTestParamForDecode(
                    b=2,
                    h_q=64,
                    s_q=64,
                    h_kv=1,
                    s_kv=2048,
                    is_varlen=True,
                    topk=2048,
                    d_qk=576,
                    d_v=512,
                    enable_attn_sink=False,
                    check_correctness=True,
                    num_runs=0,
                    seed=20260623,
                ),
                True,
            ),
            (
                "v32_total_q_512_topk2048",
                RawTestParamForDecode(
                    b=8,
                    h_q=64,
                    s_q=64,
                    h_kv=1,
                    s_kv=2048,
                    is_varlen=True,
                    topk=2048,
                    d_qk=576,
                    d_v=512,
                    enable_attn_sink=True,
                    check_correctness=True,
                    num_runs=0,
                    seed=20260625,
                ),
                True,
            ),
            (
                "model1_dqk512_attn_sink",
                RawTestParamForDecode(
                    b=4,
                    h_q=64,
                    s_q=5,
                    h_kv=1,
                    s_kv=2048,
                    is_varlen=True,
                    topk=128,
                    d_qk=512,
                    d_v=512,
                    enable_attn_sink=True,
                    check_correctness=True,
                    num_runs=0,
                    seed=20260626,
                ),
                False,
            ),
        ]

        for name, raw, contiguous_inputs in cases:
            with self.subTest(name=name):
                self._assert_matches_decode_path(raw, contiguous_inputs)
                torch.cuda.empty_cache()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    @torch.inference_mode()
    def test_rejects_unsupported_topk(self):
        self._set_up_cuda()

        q = torch.empty((1, 64, 576), dtype=torch.bfloat16)
        k_cache = torch.empty((1, 64, 1, 656), dtype=torch.float8_e4m3fn)
        indices = torch.empty((1, 96), dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, "multiple of 64"):
            flash_mla.flash_mla_sparse_prefill_varlen_clc(q, k_cache, indices)


if __name__ == "__main__":
    unittest.main()
