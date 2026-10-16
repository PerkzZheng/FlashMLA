# Paired-query sparse MLA benchmarks

These benchmarks compare two ordinary 64-head query tokens with an opt-in M128
two-CTA execution of the same two queries. Each query keeps its own selected KV
set; union metadata records both membership bits. Existing ungrouped APIs are
not automatically redirected to the paired implementation.

See [the paired-attention API guide](../docs/paired-sparse-attention.md) for
supported inputs, reusable metadata and CUDA-graph integration.

## Build and test

Run commands from the repository root, on a Blackwell SM100-family GPU. Use
CUDA Toolkit **13.2.x** for comparable FP8 results: this enables native
FP8-to-BF16 conversion. The compiler version and PyTorch CUDA runtime version
are different settings. This port uses PyTorch 2.10+ (stable ABI), CPython 3.10+,
and NVCC13.2.x. Validation uses PyTorch 2.10.0+cu130 with NVCC13.2.86; using that
runtime does not select the CUDA 13.0 compiler.

Set `CUDA_HOME` to your installed toolkit before importing PyTorch build tools:

```bash
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=4 NVCC_THREADS=2
git submodule update --init --recursive
python -m pip install ninja pytest
FLASH_MLA_SKIP_REG_SPILL_CHECK=1 python -m pip install --no-build-isolation -v -e .
export PYTHONPATH=.:tests:benchmark
export CUDA_VISIBLE_DEVICES=0
nvcc --version
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
python -m pytest tests/test_sparse_pairs.py -q
```

CUDA-enabled PyTorch must already be installed. The tests cover metadata,
independent per-query masks, empty/invalid selections, FP32 references,
changed-input graph replay, scheduling, V4.1 caches and ungrouped regressions.
Tests skip on unsupported GPUs; a skipped suite is not GPU validation.

The spill-check override above is upstream's existing build escape hatch. With
NVCC13.2.86, unmodified upstream `0f11e17` and this port both report the same
four H64 split-kernel resource entries with 16-byte stack frames (two format
variants across SM100a/SM103a). The port adds no spilling kernels. The normal
post-build gate rejects both builds, so validation uses this explicit override
and independently compares their `cuobjdump --dump-resource-usage` reports.
Do not interpret the override as a spill-free build or disable the check without
reviewing the compiler diagnostics. A full toolkit including `cuobjdump` is
needed to run the normal gate.

## Quick check: B200, SQ4, 60% overlap

This reproduces the most recent shape using DS V4.1 FP8 KV, 65536 KV tokens per
request, 64 query heads per token and head dimension 512. `60%` means
`intersection_size / topk`; the union contains approximately `1.4 * topk`
entries. The shared count is rounded to an integer and the actual overlap is
recorded. Q vectors remain independent even when KV sets overlap.

Use a fresh output directory: the benchmark appends records to its JSONL file.
Both passes below use the same seed and reverse the K execution order.

```bash
export PYTHONPATH=.:tests:benchmark CUDA_VISIBLE_DEVICES=0
RESULTS=/tmp/flashmla-b200-sq4-overlap60
mkdir "$RESULTS" || exit 1
for order in forward reverse; do
    topks="128 512 2048"
    if [ "$order" = reverse ]; then topks="2048 512 128"; fi
    for topk in $topks; do
        python benchmark/bench_sparse_pairs.py \
            --mode fp8 --batch-sizes 200 --query-lengths 4 \
            --kv-length 65536 --topk "$topk" --overlaps 0.6 \
            --algorithms bitmap --paired-split-policy auto \
            --warmups 30 --samples 100 --order "$order" \
            --output "$RESULTS/fp8-k${topk}-${order}.jsonl"
    done
done
python benchmark/summarize_sparse_pairs.py "$RESULTS"/*.jsonl \
    --output "$RESULTS/summary.md"
```

FP8 describes **KV storage**, not the MMA datatype. The current V4.1 cache has
512 FP8 values (including RoPE) with sixteen per-32 UE8M0 scales, stored
token-major in 528 bytes. Both attention variants use BF16 Q, dequantize KV
to BF16, and perform BF16 QK/PV MMA with FP32 accumulation. Historical V4
results used a different cache and older source; do not reuse their speedups
as measurements of this branch.

## Other shapes, overlap and union algorithms

The main entry point accepts arbitrary positive batch sizes, even query
lengths, and K multiples of 64 in `64..4096`. Its ungrouped comparator imposes
this K restriction; the paired metadata API itself supports `1..4096`.

```bash
python benchmark/bench_sparse_pairs.py \
    --mode fp8 --batch-sizes 1 4 64 256 --query-lengths 8 1024 \
    --kv-length 65536 --topk 512 --overlaps -1 0.5 0.7 0.9 1 \
    --algorithms sort merge bitmap --paired-split-policy auto \
    --warmups 20 --samples 50 --order forward \
    --output /tmp/flashmla-pairs-forward.jsonl
python benchmark/bench_sparse_pairs.py \
    --mode fp8 --batch-sizes 256 64 4 1 --query-lengths 1024 8 \
    --kv-length 65536 --topk 512 --overlaps 1 0.9 0.7 0.5 -1 \
    --algorithms sort merge bitmap --paired-split-policy auto \
    --warmups 20 --samples 50 --order reverse \
    --output /tmp/flashmla-pairs-reverse.jsonl
python benchmark/summarize_sparse_pairs.py \
    /tmp/flashmla-pairs-forward.jsonl /tmp/flashmla-pairs-reverse.jsonl \
    --output /tmp/flashmla-pairs-summary.md
```

`--order reverse` reverses algorithm order, not the user-supplied shape/overlap
lists; the example reverses those lists explicitly. `-1` generates independent
random selections. Nonnegative overlap uses a common core plus disjoint private
remainders. This is controlled synthetic overlap, not a measured model trace.

| Algorithm | Work charged to metadata timing | Input contract |
| --- | --- | --- |
| `sort` | Radix sort, segmented membership OR and compaction | Unsorted physical IDs |
| `merge` | Per-row sort, then merge-path union | Unsorted physical IDs; sorting is not free |
| `bitmap` | Two bit planes, OR/popcount/scan and emission | Contiguous domain of at most 65536 IDs per request |

The benchmark uses disjoint contiguous request caches, so bitmap is valid.
For shuffled physical pages, use a compatible sort/merge path or provide an
appropriate logical-ID mapping; do not declare an incorrect bitmap domain.
Top-k output sorted by score is not necessarily sorted by KV ID.

Use `--mode bf16` to compare BF16 KV through the sparse-prefill entry point,
including for short SQ. This changes the cache and attention kernel family;
it is not an isolated FP8 conversion-cost experiment. No quantization or
top-k/indexer generation is included in the timed region.

## Timing and interpretation

- Every sample is a CUDA-graph replay with a preceding random-data eviction
  touching at least four L2 capacities. Eviction is inside the graph but outside
  the timed external-event interval. Event overhead is retained.
- `baseline` measures ungrouped attention. `union` measures metadata alone.
  `attention` measures paired attention using prepared metadata, including any
  split planning/combine. `combined` directly measures union plus attention.
- E2E speedup is `baseline / combined`; values below 1 are regressions. Do not
  reconstruct E2E by summing separate component medians: cache state and graph
  boundaries differ. These are attention-pipeline, not whole-model, speedups.
- FP8 SQ greater than 8 sets upstream's `enable_batch_invariant=True` for
  **both** paths, disabling splitting. These prefill-sized measurements still
  exercise sparse-decode kernels. Short SQ retains primary dispatch unless
  `--force-no-split` explicitly overrides both paths. Compare
  absolute times as well as ratios; a slower forced baseline can inflate gains.
- The API does not silently fall back to ungrouped attention. Low overlap,
  larger K, small batches and union overhead can make grouping slower.
- Each timed path is checked against sampled ungrouped output/LSE at up to 32
  query positions. Large benchmarks are not exhaustive FP32-reference checks;
  run the correctness suite separately.

Raw JSONL retains all timing samples, p10/p90, accuracy metrics, measured
overlap, policy and peak allocated memory. The neighboring `.environment.json`
records GPU, source revision, compiler/runtime versions, arguments and binary
SHA256. Keep these together and compare repeated passes; near-1x differences
are not automatically statistically significant. GPU clocks are not locked.

Full B256/SQ4096 cases require substantial memory: the recorded FP8 campaign
peaked around 169 GiB. Reduce batch/query length on smaller devices and label
the changed shape; do not substitute a shared-cache proxy or silently chunk it.

## Larger campaigns and scope

- [Campaign instructions](staging/README.md): the complete FP8/BF16 matrix,
  provenance checks, restart rules and table generation.
This PR exposes G2 attention only. Standalone G4/G8 union research and local
profiling/conversion/cache-policy diagnostic hooks are not part of the
production implementation.

Keep generated results, compiled extensions, profiler traces and local cluster
launchers out of source commits. Record performance observations with their
exact binary and protocol rather than treating them as guarantees for a new
build or a different GPU.
