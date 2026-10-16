import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grouped = defaultdict(lambda: defaultdict(list))
    for path in args.inputs:
        for line in path.read_text().splitlines():
            record = json.loads(line)
            key = (record["mode"], record["topk"], record["requested_overlap"], record["force_no_split"], record.get("paired_split_policy", "legacy"), record.get("binary_sha256", "original"), record["queries"], record["batch"])
            grouped[key][record["algorithm"]].append(record)
    lines = [
        "# Paired-token benchmark measurements", "",
        "Median of per-pass medians; microseconds. Speedup = ungrouped / grouped end-to-end.",
        "A value below 1 means grouping is slower. Overlap -1 denotes independently sampled random rows.",
        "Union latency includes row sorting for merge-path. Attention includes dynamic split scheduling.",
        "L2 eviction is inside each CUDA graph but outside the timed interval; event overhead is retained.", "",
        "| KV | Top-k | Overlap | Force no split | Pair policy | Binary | SQ | B | Algorithm | Passes | Baseline us | Union us | Attention us | E2E us | Speedup |",
        "| --- | ---: | ---: | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, algorithms in sorted(grouped.items()):
        mode, topk, overlap, no_split, policy, binary, queries, batch = key
        for algorithm, records in sorted(algorithms.items()):
            medians = {
                field: statistics.median(record[field]["median_us"] for record in records)
                for field in ("baseline", "union", "attention", "combined")
            }
            speedup = medians["baseline"] / medians["combined"]
            lines.append(
                f"| {mode} | {topk} | {overlap:g} | {no_split} | {policy} | {binary[:8]} | {queries} | {batch} | {algorithm} | {len(records)} | "
                f"{medians['baseline']:.2f} | {medians['union']:.2f} | {medians['attention']:.2f} | "
                f"{medians['combined']:.2f} | {speedup:.3f}x |"
            )
    lines.extend(["", "## Input artifacts", ""])
    lines.extend(f"- `{path}`" for path in args.inputs)
    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
