"""Cold-L2 CUDA-graph comparison of original and paired sparse attention.

Run from the repository root with PYTHONPATH=.:tests. FP8 mode exercises
the decode kernels for both short (SQ8) and long (prefill) query sequences.
BF16 mode additionally exercises the sparse-prefill entry point.
"""

import argparse
import ctypes
import ctypes.util
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch

import flash_mla
from flash_mla._build_info import EXTENSION
import lib
import quant


CUDA_RUNTIME = ctypes.CDLL(ctypes.util.find_library("cudart") or "libcudart.so")
CUDA_RUNTIME.cudaEventRecordWithFlags.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
CUDA_RUNTIME.cudaEventRecordWithFlags.restype = ctypes.c_int


def record_external(event):
    status = CUDA_RUNTIME.cudaEventRecordWithFlags(event.cuda_event, torch.cuda.current_stream().cuda_stream, 1)
    if status:
        raise RuntimeError(f"cudaEventRecordWithFlags failed with CUDA error {status}")


def make_indices(batch, queries, topk, domain, overlap):
    result = torch.empty((batch, queries, topk), dtype=torch.int32, device="cuda")
    flat = result.view(-1, topk)
    with torch.device("cuda"):
        for begin in range(0, batch * queries, 128):
            count = min(128, batch * queries - begin)
            if overlap < 0:
                rows = lib._randperm_batch(count, torch.full((count,), domain, dtype=torch.int32), topk, [-1])
            else:
                shared = round(topk * overlap)
                candidates = lib._randperm_batch(
                    count // 2, torch.full((count // 2,), domain, dtype=torch.int32), 2 * topk - shared, [-1]
                )
                rows = torch.empty((count, topk), dtype=torch.int32)
                rows[::2] = candidates[:, :topk]
                rows[1::2, :shared] = candidates[:, :shared]
                rows[1::2, shared:] = candidates[:, topk:]
            requests = torch.arange(begin, begin + count, dtype=torch.int32) // queries
            flat[begin:begin + count] = rows + requests[:, None] * domain
    return result


def statistics_us(samples):
    ordered = sorted(samples)
    return {
        "median_us": statistics.median(samples),
        "mean_us": statistics.mean(samples),
        "p10_us": ordered[int((len(ordered) - 1) * 0.1)],
        "p90_us": ordered[int((len(ordered) - 1) * 0.9)],
        "samples_us": samples,
    }


def measure(function, eviction, sample_rows, samples, warmups, extract=True):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            function()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    end.record()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        eviction.add_(1)
        record_external(start)
        outputs = function()
        record_external(end)
    timings = []
    for iteration in range(warmups + samples):
        graph.replay()
        torch.cuda.current_stream().synchronize()
        if iteration >= warmups:
            timings.append(start.elapsed_time(end) * 1000)
    selected = None
    if extract:
        output, lse = outputs
        selected = (
            output.reshape(-1, 64, 512).index_select(0, sample_rows).float().cpu(),
            lse.reshape(-1, 64).index_select(0, sample_rows).float().cpu(),
        )
    del outputs, graph
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    return statistics_us(timings), selected


def compare(actual, expected):
    torch.testing.assert_close(actual[0], expected[0], atol=3e-4, rtol=2e-2)
    torch.testing.assert_close(actual[1], expected[1], atol=2e-4, rtol=2e-4)
    difference = actual[0] - expected[0]
    return {
        "max_abs_error": difference.abs().max().item(),
        "relative_rmse": (difference.square().mean() / expected[0].square().mean()).sqrt().item(),
        "max_lse_error": (actual[1] - expected[1]).abs().max().item(),
    }


def run_case(args, batch, queries, overlap, eviction):
    torch.manual_seed(args.seed)
    print(f"Preparing B={batch} SQ={queries} K={args.topk} overlap={overlap}", flush=True)
    kv = torch.randn((batch * args.kv_length, 1, 512), dtype=torch.bfloat16, device="cuda")
    kv.mul_(0.1)
    if args.mode == "fp8":
        cache = quant.quantize_k_cache(kv.view(-1, 64, 1, 512), quant.KVCacheLayout.V41_FP8Sparse)
        del kv
    else:
        cache = kv
    q = torch.randn((batch, queries, 64, 512), dtype=torch.bfloat16, device="cuda")
    indices = make_indices(batch, queries, args.topk, args.kv_length, overlap)
    sample_rows = torch.linspace(0, batch * queries - 1, min(32, batch * queries), device="cuda").long()
    no_split = args.force_no_split or queries > 8
    scheduler, _ = flash_mla.get_mla_metadata()
    native = torch.ops.flash_mla

    def baseline():
        if args.mode == "bf16":
            output, _, lse = native.sparse_prefill_fwd(
                q.view(-1, 64, 512), cache, indices.view(-1, 1, args.topk), 512 ** -0.5,
                512, None, None,
            )
            return output, lse
        output, lse, scheduler.tile_scheduler_metadata, scheduler.num_splits = native.sparse_decode_fwd(
            q, cache, indices, None, None, scheduler.tile_scheduler_metadata, scheduler.num_splits,
            None, None, None, 512, 512 ** -0.5, no_split,
        )
        return output, lse.transpose(1, 2)

    base_timing, expected = measure(baseline, eviction, sample_rows, args.samples, args.warmups)
    algorithms = args.algorithms if args.order == "forward" else list(reversed(args.algorithms))
    for algorithm in algorithms:
        metadata = flash_mla.prepare_sparse_pairs(
            indices, batch * args.kv_length, algorithm, request_stride=args.kv_length
        )
        mean_union = metadata.lengths.float().mean().item()

        def union():
            native.sparse_pair_union_out(
                indices, None, metadata.packed, metadata.lengths, metadata.scratch,
                batch * args.kv_length, args.kv_length, {"sort": 0, "merge": 1, "bitmap": 2}[algorithm],
            )

        def attention():
            if args.mode == "bf16":
                output, _, lse = native.sparse_prefill_fwd(
                    q.view(-1, 128, 512), cache, metadata.packed, 512 ** -0.5,
                    512, None, metadata.lengths, True,
                )
                return output.view_as(q), lse.view(batch, queries, 64)
            output, lse, _, _ = native.sparse_decode_fwd(
                q.view(-1, 1, 128, 512), cache, metadata.packed, metadata.lengths,
                None, None, None, None, None, None, 512, 512 ** -0.5, no_split, True,
                args.paired_split_policy == "auto",
            )
            return output.view_as(q), lse.view(batch, queries, 64)

        def combined():
            union()
            return attention()

        union_timing, _ = measure(union, eviction, sample_rows, args.samples, args.warmups, extract=False)
        attention_timing, observed = measure(attention, eviction, sample_rows, args.samples, args.warmups)
        accuracy = compare(observed, expected)
        combined_timing, observed = measure(combined, eviction, sample_rows, args.samples, args.warmups)
        combined_accuracy = compare(observed, expected)
        record = {
            "mode": args.mode, "kv_format": "V4.1_FP8" if args.mode == "fp8" else "BF16", "batch": batch, "queries": queries, "kv_length": args.kv_length,
            "heads": 64, "dimension": 512, "topk": args.topk, "requested_overlap": overlap,
            "measured_overlap": (2 * args.topk - mean_union) / args.topk,
            "mean_union_length": mean_union, "algorithm": algorithm, "force_no_split": no_split,
            "paired_split_policy": args.paired_split_policy, "binary_sha256": args.binary_sha256,
            "order": args.order, "seed": args.seed, "baseline": base_timing, "union": union_timing,
            "attention": attention_timing, "combined": combined_timing, "accuracy": accuracy,
            "combined_accuracy": combined_accuracy,
            "speedup": base_timing["median_us"] / combined_timing["median_us"],
            "allocated_peak_bytes": torch.cuda.max_memory_allocated(),
        }
        with args.output.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(
            f"{algorithm:6s} B={batch} SQ={queries} overlap={record['measured_overlap']:.4f} "
            f"base={base_timing['median_us']:.3f} union={union_timing['median_us']:.3f} "
            f"attn={attention_timing['median_us']:.3f} e2e={combined_timing['median_us']:.3f} "
            f"speedup={record['speedup']:.3f}x", flush=True,
        )
        del metadata, observed
        gc.collect()
        torch.cuda.empty_cache()
    del expected, q, indices, cache
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 64, 256])
    parser.add_argument("--query-lengths", type=int, nargs="+", default=[8, 1024, 4096])
    parser.add_argument("--kv-length", type=int, default=65536)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--overlaps", type=float, nargs="+", default=[-1])
    parser.add_argument("--algorithms", nargs="+", choices=["sort", "merge", "bitmap"], default=["sort", "merge", "bitmap"])
    parser.add_argument("--mode", choices=["fp8", "bf16"], default="fp8")
    parser.add_argument("--force-no-split", action="store_true", help="Use non-split decode for both baseline and grouped SQ8 cases")
    parser.add_argument("--paired-split-policy", choices=["auto", "legacy"], default="auto")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--order", choices=["forward", "reverse"], default="forward")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1 or args.warmups < 0:
        parser.error("samples must be positive and warmups nonnegative")
    if any(batch < 1 for batch in args.batch_sizes):
        parser.error("batch sizes must be positive")
    if not 0 < args.topk <= 4096 or args.topk % 64:
        parser.error("this ungrouped-baseline benchmark requires top-k to be a multiple of 64 in 64..4096")
    if any(queries <= 0 or queries % 2 for queries in args.query_lengths):
        parser.error("query lengths must be positive and even")
    if any(overlap != -1 and not 0 <= overlap <= 1 for overlap in args.overlaps):
        parser.error("overlap must be -1 (independent random) or between zero and one")
    if args.kv_length % 64 or args.kv_length < 2 * args.topk:
        parser.error("KV length must be a multiple of 64 and at least twice top-k")
    if args.output.exists() or args.output.with_suffix(".environment.json").exists():
        parser.error("output already exists; use a fresh path to avoid mixing benchmark runs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    properties = torch.cuda.get_device_properties(0)
    eviction_bytes = max(512 * 1024**2, 4 * properties.L2_cache_size)
    eviction = torch.randint(-(2**30), 2**30, (eviction_bytes // 4,), dtype=torch.int32, device="cuda")
    environment = {
        "gpu": properties.name, "compute_capability": [properties.major, properties.minor],
        "sm_count": properties.multi_processor_count, "l2_bytes": properties.L2_cache_size,
        "eviction_bytes": eviction_bytes, "torch": torch.__version__, "cuda": torch.version.cuda,
        "job": os.getenv("SLURM_JOB_ID"), "node": os.getenv("SLURM_JOB_NODELIST"),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "timing": "CUDA graph: random-data eviction.add_(1), external start event, workload, external end event; eviction excluded",
    }
    compiler = Path(os.getenv("CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc"
    environment["cuda_home"] = str(compiler.parent.parent)
    environment["nvcc_version"] = subprocess.check_output([str(compiler), "--version"], text=True) if compiler.is_file() else None
    binary = Path(flash_mla.__file__).parent / EXTENSION
    args.binary_sha256 = hashlib.sha256(binary.read_bytes()).hexdigest()
    environment["kv_format"] = "V4.1 FP8 per-32 UE8M0" if args.mode == "fp8" else "BF16"
    environment["mma"] = "BF16 operands, FP32 accumulation"
    environment["binary_sha256"] = args.binary_sha256
    args.output.with_suffix(".environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    print(json.dumps(environment), flush=True)
    for overlap in args.overlaps:
        for queries in args.query_lengths:
            for batch in args.batch_sizes:
                torch.cuda.reset_peak_memory_stats()
                run_case(args, batch, queries, overlap, eviction)


if __name__ == "__main__":
    main()
