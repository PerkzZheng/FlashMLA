# Reproducible paired-attention campaigns

Start with [the benchmark quickstart](../README.md) and build the paired-query
extension using CTK13.2.x. All commands below run from the repository root.
They require the current feature build, not an unmodified FlashMLA installation.

## Complete attention matrix

`full_sweep.py` runs the established FP8/BF16 comparison: B1/4/64/256,
SQ8/1024/4096, K128/512/2048, independent random and 50%/90%/100% overlap,
three union algorithms, two reversed execution orders, and explicit short-SQ
non-split controls. It writes 1920 records when all cases complete.

Derive the expected binary identity from **your own build**; do not copy a hash
from a historical report. Record the source revision and any local diff as well.

```bash
export PYTHONPATH=.:tests:benchmark CUDA_VISIBLE_DEVICES=0
RESULTS=/tmp/flashmla-full-sweep
BINARY_SHA256=$(python -c 'import hashlib; from pathlib import Path; import flash_mla; from flash_mla._build_info import EXTENSION; print(hashlib.sha256((Path(flash_mla.__file__).parent / EXTENSION).read_bytes()).hexdigest())')
python benchmark/staging/full_sweep.py \
    --output-dir "$RESULTS" --expected-binary-sha256 "$BINARY_SHA256" --dry-run
python -m pytest tests/test_sparse_pairs.py -q
python benchmark/staging/full_sweep.py \
    --output-dir "$RESULTS" --expected-binary-sha256 "$BINARY_SHA256"
python benchmark/staging/summarize_full_sweep.py "$RESULTS"
```

Use a new result directory for changed binaries, settings or software stacks.
The manifest records every invocation. Completed outputs are verified before
being skipped; an incomplete JSONL file is rejected rather than blindly
appended. Preserve/move partial artifacts before rerunning that invocation.

The summarizer audits coverage, two-pass identity, compiler/binary provenance,
accuracy and memory use, then creates:

- `full-table.md` / `.csv`: absolute stage times, attention/E2E gains and useful
  TFLOPS using the same useful QK+PV numerator for both implementations.
- `union-algorithm-comparison.md` / `.csv`: side-by-side metadata comparisons.
- `speedup-matrices.md`: fixed-bitmap E2E comparisons, without selecting a
  different best algorithm for each cell.
- `audit.json` and repeatability outputs: coverage and variability diagnostics.

The full matrix requires a large-memory GPU; the B256/SQ4096 FP8 cases have
previously reached about 169 GiB. Start with the quickstart's small shape if
memory or runtime is limited. Never call a truncated campaign complete.

## Arbitrary overlap or shape

For B200/SQ4/60%, 70% overlap, or any other supported input, use
`benchmark/bench_sparse_pairs.py` and `benchmark/summarize_sparse_pairs.py` as
shown in the quickstart. This route does not require any historical saved
binary or hardcoded machine-specific identity.

The historical V4 campaigns and pinned-binary helpers are not part of this
branch. Current FP8 benchmarks use upstream's V4.1 cache format and stable
operator API. Regenerate measurements on the new build instead of attributing
old V4 results to the port.

## Reporting requirements

Report actual overlap as intersection/K, and distinguish unsorted sort+merge
from merge-only on ID-sorted inputs. Include metadata in attention-pipeline
claims, retain separate attention-only timings, and disclose forced non-split
controls. A top-k list ordered by score does not satisfy an ID-sorted contract.

Keep raw samples, environment files, stdout/stderr and completion markers with
the generated tables. Record NVCC separately from `torch.version.cuda`, the
GPU model and driver, and whether clocks were controlled. Saved profiler or
conversion/cache-policy diagnostic results are not new benchmark measurements
of the final PR build.
