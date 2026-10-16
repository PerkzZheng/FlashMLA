# Experimental two-query sparse attention

This opt-in path combines **two adjacent query tokens, each with 64 heads** into
the existing SM100-family M128 **two-CTA** attention kernel. It does not change
automatic routing or enable grouping for existing calls.

## Supported inputs

- Blackwell SM100-family GPUs; BF16 Q, DQK=DV=512, one KV head.
- Contiguous Q `[batch, query_length, 64, 512]`, with positive even query length.
- Contiguous int32 selections `[batch, query_length, topk]`; top-k 1–4096.
- Physical KV index bound is positive and strictly less than `2**29`.
- Unique valid indices within each query row. Cross-query overlap is arbitrary.
- Negative and out-of-range indices are ignored. Optional per-query int32
  prefix lengths `[batch, query_length]` may also make individual rows empty.
- Decode: upstream V4.1 FP8 paged KV. Prefill: BF16 KV `[tokens, 1, 512]`.
- Optional 64-element FP32 attention sink. Extra KV caches, odd query lengths,
  D576 and non-contiguous Q are not supported by this experimental interface.

Pairing never crosses a batch/request boundary. A union entry packs the physical
token index in its upper 30 bits and two membership bits in its low bits. Each
CTA gathers its half of every union KV vector, but masks logits using only its
query's membership bit. Thus a query cannot attend to the other query's exclusive
selections. Padding is -1; empty rows retain FlashMLA's zero-output/+inf-LSE convention.

## Usage

```python
from flash_mla import prepare_sparse_pairs, flash_mla_with_kvcache_paired

metadata = prepare_sparse_pairs(indices, k_cache.shape[0] * k_cache.shape[1])
output, lse = flash_mla_with_kvcache_paired(q, k_cache, metadata)

metadata.update(new_indices)
output, lse = flash_mla_with_kvcache_paired(q, k_cache, metadata)
```

This port follows upstream's current V4.1 token-major cache layout (528 bytes/token,
per-32 UE8M0 scales); it does not restore the removed V4 cache layout. Historical
V4 measurements are not measurements of this build. BF16 Q and dequantized KV
feed BF16 MMA with FP32 accumulation. CUDA-only operators use the stable PyTorch
ABI; install PyTorch 2.10+ and CPython 3.10+.

For prefill-sized workloads using the FP8 **decode** kernels, pass
`force_no_split=True` to make the policy explicit. The paired API now defaults to
`split_policy="auto"`: it avoids split-KV when union capacity is at most 1280 or
when the independent pairs provide at least one two-CTA cluster per two SMs.
This also automatically avoids the small-batch scheduler's shared-memory scaling
for large query batches. Otherwise it uses a parallel, device-side planner.
`split_policy="legacy"` retains the earlier capacity threshold of 640 and its
serial planner for ablations. The benchmark applies upstream's `enable_batch_invariant=True` (non-split)
choice to grouped and ungrouped long-query inputs. The paired Python wrapper
maps `force_no_split=True` to this existing upstream control.

`flash_mla_sparse_fwd_paired(q, bf16_kv, metadata)` additionally exposes the BF16
prefill path. It returns output, maximum logits and LSE, reshaped to the original
batch/query dimensions.

### CUDA graphs

Allocate the workspace and warm up before capture. Capture **both**
`metadata.update(indices)` and the attention call when selections can change.
The workspace has fixed addresses and capacity; it rewrites counts and padding
on each invocation. Decode split scheduling is rebuilt inside each attention
call, not cached using stale union lengths. Do not share a workspace across
concurrent streams. An attention-only measurement deliberately excludes update;
it is not the end-to-end grouping cost.

The parallel planner partitions the actual padded union blocks across active SM
parts, limits tiny splits using the kernel's existing fixed-overhead estimate,
uses block scans for request and split prefixes, and derives each part's
request boundaries by binary search. Empty unions retain one masked block, and
unused SM parts are marked inactive. The intermediate-buffer bound and combine
format remain unchanged. Scheduling depends on current device lengths on every
replay, not on a CPU-read snapshot of their values.

## Union implementations

All algorithms emit the same ascending union and query-membership representation.

| Name | Method | Prerequisites / measured preparation |
| --- | --- | --- |
| `sort` | CUB radix sort, segmented membership OR, compacting scan | Unsorted physical indices; one kernel |
| `merge` | Per-row CUB sort followed by two-row merge-path and compacting scan | Unsorted unique input rows; **both sorting and merging are timed** |
| `bitmap` | Two shared-memory bitmaps, union popcount/scan, query-membership emission | Bounded contiguous index domain, at most 65536 tokens |

The implementations limit radix passes to the declared index bound, keep merge
temporaries in registers, traverse the two bitmap input rows directly without
per-candidate division, and initialize only the unused output tail. Every update
still writes the complete live prefix and padding, including after a union shrinks.

The merge-path design follows [DeepGEMM PR #432](https://github.com/deepseek-ai/DeepGEMM/pull/432),
specifically `scheduler/sm100_sparse_mqa_logits_metadata.cuh` at
`39d8c4cacc2c07c1fa9921c6c29c6a8a2da75359`. That source expects sorted selections.
FlashMLA's test generator instead returns indices in random-score order, so this
implementation includes sorting rather than treating it as free.

The radix-sort/segmented-OR and shared-map approaches are based on
[FlashInfer PR #5351](https://github.com/flashinfer-ai/flashinfer/pull/5351),
`q_token_kv_block_sparse_metadata.cuh` at
`81e51360be0f1a4c0dde5dd0a0e1a3329645ce68`. Here they operate on tokens rather than
sparse blocks. The map is specialized to exactly two queries: two bit planes
encode union and membership directly, avoiding a second membership lookup pass.
These are adapted algorithm comparisons, not timings of the original projects'
full metadata operators, schedulers, or their broader supported layouts.

For `bitmap`, `request_stride=65536` declares disjoint contiguous request caches:
request `r` may select `[r*65536, (r+1)*65536)`. A zero stride means all requests
share the entire declared KV domain. Do not use this mode on arbitrarily shuffled
physical pages unless the indices still satisfy that domain contract; use sort
or merge instead. The benchmark uses **disjoint** caches, not a shared-cache proxy.

## Reproduce

Use CUDA Toolkit **13.2.x** for comparable builds and benchmarks. Follow the
[benchmark quickstart](../benchmark/README.md) for toolkit selection, build,
correctness tests and executable examples. The existing conversion helpers
then use native FP8-to-BF16 instructions;
older supported toolchains widen through FP16 and FP32. This can materially
affect paired attention performance without changing the KV format. Compare
grouped and ungrouped kernels built with the same compiler, and report the
compiler version separately from PyTorch's CUDA runtime version.

```bash
PYTHONPATH=.:tests python -m pytest tests/test_sparse_pairs.py -q
PYTHONPATH=.:tests python benchmark/bench_sparse_pairs.py \
  --query-lengths 8 --overlaps -1 0.5 0.9 1 --output results/decode.jsonl
PYTHONPATH=.:tests python benchmark/bench_sparse_pairs.py \
  --query-lengths 1024 4096 --output results/prefill.jsonl
```

Repeat with `--order reverse` and a different output file to reverse algorithm
order. `--mode bf16` additionally benchmarks BF16 prefill. The default top-k is
2048, KV length 65536, and batches 1/4/64/256. Other supported top-k values can be
selected explicitly; this benchmark restricts top-k to multiples of 64 for the
ungrouped baseline's tile loads, while the paired API itself pads arbitrary top-k.
`--force-no-split` overrides split-KV for **both** short-query variants, allowing a
separate scheduling sensitivity study. `--paired-split-policy legacy` restores
the initial paired scheduling behavior without undoing union optimizations.
Each record includes its binary hash. The summary tool separates policies and binaries instead of merging
their samples. The auto heuristic selects split versus non-split paired kernels;
it never hides a grouping regression by silently running ungrouped attention.
The shape-based heuristic is not a universal optimum for highly ragged lengths;
the legacy policy remains available for workload-specific comparisons.

### Timing and interpretation

- Each sample replays one CUDA graph. Its first kernel reads and updates a
  random-initialized eviction buffer at least four times the GPU's L2 capacity.
- External CUDA event nodes bracket only the target workload; eviction is
  outside the timed interval. The benchmark records these nodes using the
  CUDA runtime's `cudaEventRecordWithFlags(..., cudaEventRecordExternal)`.
- Report union-only, attention-only (including dynamic split scheduling), and
  actual combined latency. Do **not** add separate medians to estimate combined
  latency: freshly produced metadata is naturally cached in the combined run.
- Event-node timing overhead is retained, not silently subtracted.
- Original decode's scheduler is reusable for static top-k. The paired path
  includes regeneration because its union lengths can change on graph replay.
- Independent random rows use FlashMLA's `_randperm_batch`, generated in bounded
  chunks outside timing. Controlled-overlap rows also use that generator but are
  labeled separately. These are synthetic tests, not recorded model selections.
- Report full raw samples, medians, p10/p90, exact environment and arguments.
  Benchmark correctness compares output/LSE at up to 32 query positions spread
  across the full allocation; dedicated tests additionally use FP32 references.

If a pair overlaps by fraction `r` of its top-k, its union has `(2-r)*topk`
entries. An M128 union tile computes both queries against that union, including
masked-out scores. Relative to two M64 queries, dot-product work therefore grows
by roughly `2-r`, while potential unique KV traffic falls by `r/2`. At top-k2048
in a 64K uniformly random domain, expected overlap is only 3.125%: a nearly
twofold compute increase is exchanged for just 1.56% fewer distinct KV tokens.
Grouping is not guaranteed to be faster, even before union construction cost.
