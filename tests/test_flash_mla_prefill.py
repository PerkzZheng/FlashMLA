import math
import time
from typing import Tuple
import random
import dataclasses

import torch
import triton

from flash_mla import flash_mla_sparse_fwd, index_attention_with_kvcache_bf16
from lib import check_is_allclose

@dataclasses.dataclass
class TestParam:
    b: int
    s_q: int
    s_kv: int
    topk: int
    win_s_kv: int = 128
    h_q: int = 64
    h_kv: int = 1
    d_qk: int = 512
    d_v: int = 512
    block_size: int = 64
    seed: int = 0
    has_sinks: bool = False
    check_correctness: bool = True
    benchmark: bool = True

@dataclasses.dataclass
class Testcase:
    t: TestParam
    cu_seqlens_q: torch.Tensor
    q: torch.Tensor
    kv: torch.Tensor
    indices: torch.Tensor
    paged_kv: torch.Tensor
    paged_block_ids: torch.Tensor
    win_kv: torch.Tensor
    win_indices: torch.Tensor
    win_paged_kv: torch.Tensor
    win_paged_block_ids: torch.Tensor
    sinks: torch.Tensor

def generate_testcase(t: TestParam) -> Testcase:
    torch.manual_seed(t.seed)
    torch.cuda.manual_seed(t.seed)
    random.seed(t.seed)
    q = torch.randn((t.b, t.s_q, t.h_q, t.d_qk), dtype=torch.bfloat16) / 10
    kv = torch.randn((t.b, t.s_kv, t.h_kv, t.d_qk), dtype=torch.bfloat16) / 10
    win_kv = torch.randn((t.b, t.win_s_kv, t.h_kv, t.d_qk), dtype=torch.bfloat16) / 10

    q.clamp_(-10, 10)
    kv.clamp_(-10, 10)

    indices = None
    if t.topk > 0:
        indices = torch.full((t.b, t.s_q, t.h_kv, t.topk), t.s_kv, dtype=torch.int32)
        for b in range(t.b):
            for s in range(t.s_q):
                for h in range(t.h_kv):
                    # NOTE We use the following method to generate indices so that most indices lies within [s_kv-20000, s_kv), which is more realistic for sparse attention
                    near_mask = torch.randint(0, 32, (min(t.topk, t.s_kv),)) < 31
                    cur_indices = torch.randperm(t.s_kv)[:t.topk]
                    cur_indices[near_mask] = torch.randint(max(0, t.s_kv - 20000), t.s_kv - 1, (near_mask.sum().item(),))
                    if len(cur_indices) < t.topk:
                        cur_indices = torch.cat([cur_indices, torch.full((t.topk - len(cur_indices),), 2147480000)])
                    cur_indices = cur_indices[torch.randperm(t.topk)]
                    indices[b, s, h] = cur_indices
        indices = indices.to(q.device)

    # Arrange indices in the window from [0, win_s_kv) and expand it to [s_q, win_s_kv]
    win_indices = torch.arange(t.win_s_kv, dtype=torch.int32, device=q.device).unsqueeze(0).expand(t.b, t.s_q, t.h_kv, -1).contiguous()
    
    # Create paged kv.
    paged_kv = None
    paged_block_ids = None
    if t.topk > 0:
        max_num_blocks_per_seq = math.ceil(t.s_kv / t.block_size)
        paged_kv = torch.zeros((t.b, max_num_blocks_per_seq, t.block_size, t.h_kv, t.d_qk), dtype=torch.bfloat16, device=q.device)
        # Fill paged kv with kv.
        for b in range(t.b):
            for s in range(t.s_kv):
                block_id = s // t.block_size
                paged_kv[b, block_id, s % t.block_size] = kv[b, s, 0, :]
        # Reshape paged kv to [b * s_kv, h_kv, d_qk]
        paged_kv = paged_kv.view(t.b * max_num_blocks_per_seq, t.block_size, t.h_kv, t.d_qk)
        # Create paged block ids.
        paged_block_ids = torch.arange(t.b * max_num_blocks_per_seq, dtype=torch.int32, device=q.device).view(t.b, -1).contiguous()
        

    # Create win paged kv.
    max_num_win_blocks_per_seq = math.ceil(t.win_s_kv / t.block_size)
    win_paged_kv = torch.zeros((t.b, max_num_win_blocks_per_seq, t.block_size, t.h_kv, t.d_qk), dtype=torch.bfloat16, device=q.device)
    # Fill win paged kv with win kv.
    for b in range(t.b):
        for s in range(t.win_s_kv):
            block_id = s // t.block_size
            win_paged_kv[b, block_id, s % t.block_size] = win_kv[b, s, 0, :]
    # Reshape win paged kv to [b * s_q, h_kv, d_qk]
    win_paged_kv = win_paged_kv.view(t.b * max_num_win_blocks_per_seq, t.block_size, t.h_kv, t.d_qk)
    # Create win paged block ids.
    win_paged_block_ids = torch.arange(t.b * max_num_win_blocks_per_seq, dtype=torch.int32, device=q.device).view(t.b, -1).contiguous()

    # Cumulative seqlens of q.
    cu_seqlens_q = torch.zeros((t.b + 1,), dtype=torch.int32, device=q.device)
    for b in range(t.b):
        cu_seqlens_q[b + 1] = cu_seqlens_q[b] + t.s_q

    # Attention sinks with range of float [2, 3]
    if t.has_sinks:
        sinks = torch.rand(t.h_q, dtype=torch.float32, device=q.device) / 10
    else:
        sinks = None

    return Testcase(
        t=t,
        cu_seqlens_q=cu_seqlens_q,
        q=q,
        kv=kv,
        paged_kv=paged_kv,
        paged_block_ids=paged_block_ids,
        win_kv=win_kv,
        win_paged_kv=win_paged_kv,
        win_paged_block_ids=win_paged_block_ids,
        indices=indices,
        win_indices=win_indices,
        sinks=sinks
    )

def get_flop(p: TestParam) -> float:
    flop = 2 * sum([
        p.h_q * p.d_qk * (p.topk + p.win_s_kv),
        p.h_q * p.d_v * (p.topk + p.win_s_kv)
    ]) * p.b * p.s_q
    return flop

def reference_torch(p: TestParam, t: Testcase, sm_scale: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def log2sumexp2(a: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.logsumexp(a * math.log(2), dim=dim) * math.log2(math.e)
    output = torch.zeros((p.b, p.s_q, p.h_q, p.d_v), dtype=torch.bfloat16, device=t.q.device)
    for bi in range(p.b):
        indices = None
        win_indices = None
        assert p.topk > 0 or p.win_s_kv > 0
        if p.topk > 0:
            indices = t.indices[bi, :, 0, :]  # [s_q, topk]
            invalid_indices_mask = (indices < 0) | (indices >= p.s_kv)
        win_indices = None
        invalid_win_indices_mask = None
        if p.win_s_kv > 0:
            win_indices = t.win_indices[bi, :, 0, :]  # [s_q, win_s_kv]
            invalid_win_indices_mask = (win_indices < 0) | (win_indices >= p.win_s_kv)
        qs = t.q[bi, :, :, :].float()  # [s_q, h_q, d_qk]
        kvs = None
        if p.topk > 0:
            kvs = t.kv[bi, :, 0, :].float()  # [s_kv, d_qk]
            kvs = torch.index_select(kvs, 0, indices.masked_fill(invalid_indices_mask, 0).flatten()).view(p.s_q, p.topk, p.d_qk)  # [s_q, topk, d_qk]
        # concat win_kv
        if p.win_s_kv > 0:
            win_kvs = t.win_kv[bi, :, 0, :].float()  # [win_s_kv, d_qk]
            win_kvs = win_kvs.expand(p.s_q, p.win_s_kv, p.d_qk)
            if p.topk > 0:
                kvs = torch.cat([win_kvs, kvs], dim=1)  # [s_q, topk + win_s_kv, d_qk]
                invalid_indices_mask = torch.cat([invalid_win_indices_mask, invalid_indices_mask], dim=1)
            else:
                kvs = win_kvs
                invalid_indices_mask = invalid_win_indices_mask
        attn_score = qs @ kvs.transpose(1, 2)    # [s_q, h_q, topk + win_s_kv]
        attn_score.masked_fill_(invalid_indices_mask.unsqueeze(1), float('-inf'))
        attn_score *= sm_scale * math.log2(math.e)
        max_logits = torch.max(attn_score, dim=-1)[0]   # [s_q, h_q]
        if t.sinks is not None:
            attn_score = torch.concat([attn_score, t.sinks.view(p.h_q, 1).expand(p.s_q, p.h_q, 1)], dim=-1)
        lse = log2sumexp2(attn_score, dim=-1)   # [s_q, h_q]
        if t.sinks is not None:
            attn_score = attn_score[:, :, :-1]
        attn_score = torch.exp2(attn_score - lse.unsqueeze(-1))   # [s_q, h_q, topk]
        output[bi, :, :, :] = attn_score @ kvs[:, :, :p.d_v]
    return (max_logits, lse, output.view(-1, p.h_q, p.d_v))

@torch.inference_mode()
def run_test(p: TestParam) -> bool:
    print("================")
    print(f"Running on {p}")
    torch.cuda.empty_cache()

    t = generate_testcase(p)
    sm_scale = 1 / math.sqrt(p.d_qk)
    torch.cuda.synchronize()


    # def index_attention_with_kvcache_bf16(
    #         q: Tensor,
    #         win_kvcache: Tensor,
    #         win_block_ids: Tensor,
    #         win_topk_ids: Tensor,
    #         compress_kvcache: Tensor,
    #         compress_block_ids: Tensor,
    #         compress_topk_ids: Tensor,
    #         cu_seqlens_q: Tensor,
    #         sink_weight: Tensor,
    #         sm_scale: float,
    #         output: Tensor = None,
    #     ) -> Tensor:

    def run_ans():
        # return flash_mla_sparse_fwd(
        #     t.q.squeeze(0), t.kv.squeeze(0), t.indices.squeeze(0), t.win_kv.squeeze(0), t.win_indices.squeeze(0), t.sinks, sm_scale=sm_scale
        # )
        return index_attention_with_kvcache_bf16(
            t.q.view(-1, p.h_q, p.d_qk),
            t.win_paged_kv,
            t.win_paged_block_ids,
            t.win_indices.view(-1, p.win_s_kv),
            t.paged_kv if t.paged_kv is not None else None,
            t.paged_block_ids if t.paged_block_ids is not None else None,
            t.indices.view(-1, p.topk) if t.indices is not None else None,
            t.cu_seqlens_q,
            t.sinks,
            sm_scale=sm_scale
        )

    ans_out = run_ans()
    torch.cuda.synchronize()

    if p.benchmark:
        flop = get_flop(p)
        prefill_ans_time: float = triton.testing.do_bench(run_ans, warmup=10, rep=20) / 1000  # type: ignore
        prefill_flops = flop / prefill_ans_time / 1e12
        print(f"Prefill:  {prefill_ans_time * 1e6:4.0f} us, {prefill_flops:.3f} TFlops")

    if p.check_correctness:
        torch.cuda.synchronize()
        ref_max_logits, ref_lse, ref_out = reference_torch(p, t, sm_scale)
        torch.cuda.synchronize()

        is_correct = True
        is_correct &= check_is_allclose("out", ans_out, ref_out, abs_tol=8e-4, rel_tol=2.01 / 128, cos_diff_tol=7e-6)
        # is_correct &= check_is_allclose("max_logits", ans_max_logits, ref_max_logits, abs_tol=1e-6, rel_tol=2.01 / 65536)
        # is_correct &= check_is_allclose("lse", ans_lse, ref_lse, abs_tol=1e-6, rel_tol=2.01 / 65536)

        return is_correct
    else:
        return True


if __name__ == '__main__':
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')

    correctness_cases = [
        # Regular shapes
        TestParam(1, s_q, s_kv, topk, has_sinks=has_sinks, h_q=64, benchmark=False)
        for s_kv, topk in [
            # Regular shapes
            (128, 128),
            (256, 256),
            (512, 512),

            # Irregular shapes
            (592, 128),
            (1840, 256),
            (1592, 384),
            (1521, 512),

            # Irregular shapes with OOB TopK
            (95, 128),
            (153, 256),
            (114, 384),
        ]
        for s_q in [
            1, 62
        ]
        for has_sinks in [True, False]
    ]

    ds_new_cases = [
        TestParam(8, s, s, topk, h_q=h_q, benchmark=False)
            for s in [592, 1840, 1592, 1521, 95, 153, 114]
            for topk in [0, 512, 812]
            for h_q in [16, 64]
    ]

    corner_cases = [
        # In these cases, some blocks may not have any valid topk indices
        TestParam(1, s_q, s_kv, topk, h_q=64, benchmark=False)
        for s_kv, topk in [
            (32, 2048),
            (64, 8192)
        ]
        for s_q in [1, 1024]
    ]

    performance_cases = [
        TestParam(1, s_q, s_kv, topk, h_q=64)
        for s_q in [4096]
        for s_kv in [4096]
        for topk in [512, 1024]
    ]

    testcases = correctness_cases + ds_new_cases + corner_cases + performance_cases

    failed_cases = []
    for test in testcases:
        if test.benchmark:
            time.sleep(0.2)
        is_correct = run_test(test)
        if not is_correct:
            failed_cases.append(test)

    if len(failed_cases) > 0:
        print(f"\033[31m\033[1m{len(failed_cases)} / {len(testcases)} cases failed:\033[0m")
        for case in failed_cases:
            print(f"    {case}")
    else:
        print(f"\033[32m\033[1mAll {len(testcases)} cases passed!\033[0m")
