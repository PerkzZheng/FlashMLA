import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def make_jobs(directory):
    jobs = []
    for order in ("forward", "reverse"):
        configurations = list(itertools.product(("fp8", "bf16"), (128, 512, 2048), (-1, 0.5, 0.9, 1)))
        if order == "reverse":
            configurations.reverse()
        for mode, topk, overlap in configurations:
            for queries, samples, warmups in (([8], 50, 20), ([1024, 4096], 30, 10)):
                label = f"{mode}-k{topk}-overlap{overlap:g}-q{queries[0]}-{order}"
                jobs.append(make_job(directory, label, mode, topk, overlap, queries, samples, warmups, order, False))
        controls = list(itertools.product((512, 2048), (-1, 0.5, 0.9, 1)))
        if order == "reverse":
            controls.reverse()
        for topk, overlap in controls:
            label = f"fp8-k{topk}-overlap{overlap:g}-q8-nosplit-{order}"
            jobs.append(make_job(directory, label, "fp8", topk, overlap, [8], 50, 20, order, True))
    return jobs


def make_job(directory, label, mode, topk, overlap, queries, samples, warmups, order, no_split):
    batches = [1, 4, 64, 256]
    if order == "reverse":
        batches.reverse()
        queries = list(reversed(queries))
    output = directory / (label + ".jsonl")
    command = [sys.executable, "benchmark/bench_sparse_pairs.py", "--mode", mode,
               "--topk", str(topk), "--overlaps", str(overlap),
               "--batch-sizes", *map(str, batches), "--query-lengths", *map(str, queries),
               "--algorithms", "sort", "merge", "bitmap", "--paired-split-policy", "auto",
               "--samples", str(samples), "--warmups", str(warmups), "--order", order,
               "--output", str(output)]
    if no_split:
        command.append("--force-no-split")
    return {"label": label, "mode": mode, "topk": topk, "overlap": overlap,
            "queries": queries, "batches": batches, "samples": samples, "warmups": warmups,
            "order": order, "force_no_split": no_split, "expected_records": len(batches) * len(queries) * 3,
            "output": str(output), "command": command}


def verify_records(job, binary_hash):
    path = Path(job["output"])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    expected = set(itertools.product(job["batches"], job["queries"], ("sort", "merge", "bitmap")))
    observed = {(row["batch"], row["queries"], row["algorithm"]) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError(f"Incomplete or duplicate records: {path}")
    for row in rows:
        if (row["binary_sha256"] != binary_hash or row["mode"] != job["mode"]
                or row["topk"] != job["topk"] or row["requested_overlap"] != job["overlap"]
                or row["order"] != job["order"] or row["paired_split_policy"] != "auto"
                or row["force_no_split"] != (job["force_no_split"] or row["queries"] > 8)):
            raise ValueError(f"Configuration mismatch: {path}")
        for stage in ("baseline", "union", "attention", "combined"):
            if len(row[stage]["samples_us"]) != job["samples"]:
                raise ValueError(f"Wrong sample count: {path}")
        if "accuracy" not in row or "combined_accuracy" not in row:
            raise ValueError(f"Missing correctness checks: {path}")
    environment = json.loads(path.with_suffix(".environment.json").read_text())
    if "release 13.2," not in environment["nvcc_version"] or environment["binary_sha256"] != binary_hash:
        raise ValueError(f"Wrong toolkit or binary: {path}")
    if environment["eviction_bytes"] < 4 * environment["l2_bytes"]:
        raise ValueError(f"Insufficient L2 eviction: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-binary-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    jobs = make_jobs(directory)
    manifest = {"jobs": jobs, "expected_records": sum(job["expected_records"] for job in jobs),
                "binary_sha256": args.expected_binary_sha256}
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing manifest differs; use a separate result directory")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.dry_run:
        print(f"{len(jobs)} jobs, {manifest['expected_records']} records", flush=True)
        return
    compiler = Path(os.environ["CUDA_HOME"]) / "bin/nvcc"
    version = subprocess.check_output([str(compiler), "--version"], text=True)
    if "release 13.2," not in version:
        raise ValueError("This sweep requires CTK 13.2.x")
    import flash_mla
    from flash_mla._build_info import EXTENSION
    binary_hash = hashlib.sha256((Path(flash_mla.__file__).parent / EXTENSION).read_bytes()).hexdigest()
    if binary_hash != args.expected_binary_sha256:
        raise ValueError("Unexpected extension binary; rebuild or correct provenance before timing")
    for position, job in enumerate(jobs, 1):
        output = Path(job["output"])
        done = output.with_suffix(".done.json")
        if done.exists():
            verify_records(job, binary_hash)
            print(f"SKIP {position}/{len(jobs)} {job['label']}", flush=True)
            continue
        if output.exists():
            raise ValueError(f"Partial output exists; preserve or move it before resuming: {output}")
        started = time.time()
        print(f"START {position}/{len(jobs)} {job['label']}", flush=True)
        with output.with_suffix(".log").open("w") as handle:
            subprocess.run(job["command"], stdout=handle, stderr=subprocess.STDOUT, check=True)
        verify_records(job, binary_hash)
        duration = time.time() - started
        done.write_text(json.dumps({"seconds": duration, "records": job["expected_records"]}) + "\n")
        print(f"DONE {position}/{len(jobs)} {job['label']} {duration:.1f}s", flush=True)
    print("FULL_SWEEP_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
