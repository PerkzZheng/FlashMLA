import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

from full_sweep import verify_records


def policy(row):
    if row["mode"] == "bf16":
        return "BF16 prefill"
    if row["queries"] > 8:
        return "both non-split"
    return "both non-split control" if row["force_no_split"] else "paired auto"


def case_key(row):
    return tuple(row[key] for key in ("mode", "queries", "batch", "topk", "requested_overlap", "force_no_split"))


def primary(row):
    return not (row["mode"] == "fp8" and row["queries"] == 8 and row["force_no_split"])


def aggregate(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[case_key(record) + (record["algorithm"],)].append(record)
    result = []
    for key, passes in sorted(grouped.items()):
        if len(passes) != 2 or {row["order"] for row in passes} != {"forward", "reverse"}:
            raise ValueError(f"Expected one forward and one reverse pass: {key}")
        mode, queries, batch, topk, overlap, no_split, algorithm = key
        row = {"mode": mode, "queries": queries, "batch": batch, "topk": topk,
               "requested_overlap": overlap, "force_no_split": no_split, "algorithm": algorithm,
               "measured_overlap": statistics.mean(item["measured_overlap"] for item in passes),
               "passes": len(passes)}
        for stage in ("baseline", "union", "attention", "combined"):
            for statistic in ("median", "p10", "p90"):
                row[f"{stage}_{statistic}_us"] = statistics.median(item[stage][f"{statistic}_us"] for item in passes)
        baseline, attention, combined = (row[f"{stage}_median_us"] for stage in ("baseline", "attention", "combined"))
        useful_flops = 4 * batch * queries * 64 * topk * 512
        row.update({"policy": policy(row), "attention_speedup": baseline / attention,
                    "e2e_speedup": baseline / combined, "baseline_tflops": useful_flops / baseline / 1e6,
                    "paired_useful_tflops": useful_flops / attention / 1e6})
        for metric in ("max_abs_error", "relative_rmse", "max_lse_error"):
            values = [item[field][metric] for item in passes for field in ("accuracy", "combined_accuracy")]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Nonfinite accuracy metric: {key}")
            row[metric] = max(values)
        row["allocated_peak_bytes"] = max(item["allocated_peak_bytes"] for item in passes)
        result.append(row)
    return result


def markdown_table(rows):
    lines = [
        "| KV | SQ | B | K | Overlap | Policy | Union | Ungrouped us | Union us | Paired attention us | Paired E2E us | Attention speedup | E2E speedup | Ungrouped TFLOPS | Paired useful TFLOPS |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        overlap = "random" if row["requested_overlap"] < 0 else f"{100 * row['requested_overlap']:g}%"
        cells = [row["mode"], str(row["queries"]), str(row["batch"]), str(row["topk"]), overlap, row["policy"], row["algorithm"]]
        cells += [f"{row[stage + '_median_us']:.2f}" for stage in ("baseline", "union", "attention", "combined")]
        cells += [f"{row[field]:.3f}x" for field in ("attention_speedup", "e2e_speedup")]
        cells += [f"{row[field]:.1f}" for field in ("baseline_tflops", "paired_useful_tflops")]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_outputs(directory, rows, audit):
    with (directory / "full-table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    introduction = [
        "# Complete CTK 13.2.x benchmark rerun", "",
        "GB300, KV65536/request, H64/token, D512; V4.1 FP8 or BF16 KV. Q vectors are independent.",
        "Cold L2 before every CUDA-graph sample; eviction is outside the timed region.",
        "Two pass medians; SQ8: 20 warmups/50 samples, SQ1024/4096: 10/30.",
        "Reverse pass reverses configuration, batch, query-length and union-algorithm order.",
        "E2E is measured directly, not obtained by adding separate component medians.",
        "Attention timing includes any split planning/combine, but excludes union preparation.",
        "TFLOPS count useful QK+PV operations (4 * B * SQ * 64 * K * 512), excluding softmax and metadata.",
        "Both paths use the same useful FLOP count; masked union arithmetic does not inflate paired TFLOPS.",
        "BF16 uses its sparse-prefill entry point, including at SQ8. No hidden ungrouped fallback.",
        "Current paired auto dispatch is used; explicit SQ8 non-split controls are labeled separately.",
        "A speedup below 1 is a regression. p10/p90 and correctness metrics are also in full-table.csv.",
        f"Binary SHA256: `{audit['binary_sha256']}`.",
        f"Validated {audit['records']} raw records, {audit['cases']} workload/policy cases, {audit['averaged_rows']} algorithm rows.", "",
    ]
    (directory / "full-table.md").write_text("\n".join(introduction + markdown_table(rows)) + "\n")
    bitmap = [row for row in rows if row["algorithm"] == "bitmap"]
    (directory / "bitmap-table.md").write_text("\n".join(introduction + markdown_table(bitmap)) + "\n")
    lines = introduction + ["## Primary bitmap E2E speedups", "", "Fixed algorithm throughout; no per-cell winner selection.", ""]
    lookup = {(row["mode"], row["queries"], row["topk"], row["batch"], row["requested_overlap"]): row
              for row in bitmap if primary(row)}
    for mode in ("fp8", "bf16"):
        for queries in (8, 1024, 4096):
            for topk in (128, 512, 2048):
                lines += [f"### {mode.upper()} SQ{queries} K{topk}", "",
                          "| Batch | Random E2E | 50% E2E | 90% E2E | 100% E2E | 100% attention |",
                          "| ---: | ---: | ---: | ---: | ---: | ---: |"]
                for batch in (1, 4, 64, 256):
                    selected = [lookup[mode, queries, topk, batch, overlap] for overlap in (-1, 0.5, 0.9, 1)]
                    cells = [str(batch)] + [f"{row['e2e_speedup']:.3f}x" for row in selected]
                    cells += [f"{selected[-1]['attention_speedup']:.3f}x"]
                    lines.append("| " + " | ".join(cells) + " |")
                lines.append("")
    lines += ["## Explicit non-split decode controls", ""]
    lines += markdown_table([row for row in bitmap if not primary(row)])
    (directory / "speedup-matrices.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    records = []
    environments = []
    for original in manifest["jobs"]:
        job = dict(original, output=str(directory / Path(original["output"]).name))
        if not Path(job["output"]).with_suffix(".done.json").exists():
            raise ValueError(f"Job not complete: {job['label']}")
        verify_records(job, manifest["binary_sha256"])
        records += [json.loads(line) for line in Path(job["output"]).read_text().splitlines()]
        environments.append(json.loads(Path(job["output"]).with_suffix(".environment.json").read_text()))
    if len(records) != manifest["expected_records"]:
        raise ValueError("Record count does not match manifest")
    rows = aggregate(records)
    cases = defaultdict(list)
    for row in rows:
        cases[case_key(row)].append(row)
    winners = Counter()
    ties = 0
    for case_rows in cases.values():
        if len(case_rows) != 3:
            raise ValueError("Missing union algorithm")
        minimum = min(row["union_median_us"] for row in case_rows)
        near_best = [row for row in case_rows if row["union_median_us"] <= minimum * 1.02]
        if len(near_best) == 1:
            winners[near_best[0]["algorithm"]] += 1
        else:
            ties += 1
    audit = {"records": len(records), "cases": len(cases), "averaged_rows": len(rows),
             "binary_sha256": manifest["binary_sha256"], "jobs": sorted({env['job'] for env in environments}),
             "nodes": sorted({env['node'] for env in environments}), "gpus": sorted({env['gpu'] for env in environments}),
             "nvcc_versions": sorted({env['nvcc_version'] for env in environments}),
             "torch_cuda_runtimes": sorted({env['cuda'] for env in environments}),
             "peak_allocated_gib": max(row["allocated_peak_bytes"] for row in rows) / 1024**3,
             "union_winners_2pct_tolerance": dict(winners), "union_ties_2pct_tolerance": ties,
             "max_accuracy": {metric: max(row[metric] for row in rows)
                              for metric in ("max_abs_error", "relative_rmse", "max_lse_error")}}
    (directory / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    write_outputs(directory, rows, audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
