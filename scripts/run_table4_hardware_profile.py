#!/usr/bin/env python3
"""Reproduce Table 4: Impact of Cache-Aligned Memory Reordering on L3 Cache
and CPU Time (Sec. 4.4).

Builds a base HNSW index and a Phase-1-reordered HNSW index -- same
topology, same M/ef_construction, no shortcuts -- on identical data, then
uses the Linux `perf stat` tool to compare LLC-loads, LLC-load-misses, and
task-clock while searching the same test queries. This isolates Phase 1's
pure hardware effect: the number of distance computations is identical
between the two indices (same graph), only the physical memory layout of
the vectors differs.

Requires the Linux `perf` tool with sufficient permissions
(kernel.perf_event_paranoid <= 2, or CAP_PERFMON on the Python process).

Example:
    python scripts/run_table4_hardware_profile.py \\
        --datasets-dir ./dataset_benchmark \\
        --datasets LAION COCO \\
        --output-dir ./results/table4
"""
import argparse
import gc
import os
import shutil
import subprocess
import sys

import common
import h5py
import hnswlib
import numpy as np
import pandas as pd

import ours_utils


def check_perf_available():
    if shutil.which("perf") is None:
        sys.exit(
            "`perf` was not found on PATH. Install it, e.g.:\n"
            "  sudo apt install linux-tools-common linux-tools-$(uname -r)\n"
            "and re-run this script."
        )
    probe = subprocess.run(["perf", "stat", "-e", "task-clock", "true"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if probe.returncode != 0:
        sys.exit(
            "`perf stat` failed to run (this is usually a permissions issue "
            "in containers/CI). Try:\n"
            "  sudo sysctl -w kernel.perf_event_paranoid=1\n\n"
            f"perf stderr:\n{probe.stderr}"
        )


def build_indexes(dataset_key, dataset_file, args, out_dir):
    hdf5_path = os.path.join(args.datasets_dir, f"{dataset_file}.hdf5")
    if not os.path.exists(hdf5_path):
        print(f"  [skip] dataset file not found: {hdf5_path}")
        return None

    with h5py.File(hdf5_path, "r") as f:
        raw_db = np.array(f["train"]).astype(np.float32)
    num_elements, dim = raw_db.shape

    base_path = os.path.join(
        out_dir, common.base_index_filename(dataset_key, args.m, args.ef_construction))
    reord_path = base_path.replace("_base_", "_reordered_")

    if not os.path.exists(base_path):
        print("  Building base (unreordered) HNSW...")
        idx = hnswlib.Index(space="ip", dim=dim)
        idx.init_index(max_elements=num_elements, ef_construction=args.ef_construction, M=args.m)
        idx.add_items(raw_db, np.arange(num_elements))
        idx.save_index(base_path)
        del idx
        gc.collect()

    if not os.path.exists(reord_path):
        print("  Building Phase-1-reordered HNSW...")
        sorted_db, _, _ = ours_utils.reorder_dataset_by_clustering(
            raw_db, num_clusters=args.num_clusters, sample_ratio=args.sample_ratio,
            n_init=args.n_init, max_iter=args.max_iter, batch_size=args.batch_size,
        )
        idx = hnswlib.Index(space="ip", dim=dim)
        idx.init_index(max_elements=num_elements, ef_construction=args.ef_construction, M=args.m)
        idx.add_items(sorted_db, np.arange(num_elements))
        idx.save_index(reord_path)
        del idx
        gc.collect()

    return dim, num_elements, hdf5_path, base_path, reord_path


def write_worker_script(path, query_repeat, ef_search):
    code = f'''import sys
import h5py
import numpy as np
import hnswlib

index_path, data_path, dim, num_elements = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

with h5py.File(data_path, "r") as f:
    test_q = np.array(f["test"]).astype(np.float32)

idx = hnswlib.Index(space="ip", dim=dim)
idx.load_index(index_path, max_elements=num_elements)
idx.set_ef({ef_search})

for _ in range({query_repeat}):
    idx.knn_query(test_q, k=10, num_threads=1)
'''
    with open(path, "w") as f:
        f.write(code)


def run_perf_stat(worker_script, index_path, data_path, dim, num_elements):
    cmd = ["perf", "stat", "-x", ",", "-e", "LLC-loads,LLC-load-misses,task-clock",
           sys.executable, worker_script, index_path, data_path, str(dim), str(num_elements)]
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)

    loads, misses, cpu_time_ms = 0, 0, 0.0
    for line in result.stderr.split("\n"):
        try:
            if "LLC-loads" in line:
                loads = int(line.split(",")[0])
            elif "LLC-load-misses" in line:
                misses = int(line.split(",")[0])
            elif "task-clock" in line:
                cpu_time_ms = float(line.split(",")[0])
        except ValueError:
            continue

    miss_rate = (misses / loads * 100) if loads > 0 else 0.0
    return loads, misses, miss_rate, cpu_time_ms


def run_dataset(dataset_key, dataset_file, args):
    print(f"\n{'=' * 70}\nDataset: {dataset_key} ({dataset_file})\n{'=' * 70}")
    out_dir = os.path.join(args.output_dir, dataset_key)
    os.makedirs(out_dir, exist_ok=True)

    built = build_indexes(dataset_key, dataset_file, args, out_dir)
    if built is None:
        return None
    dim, num_elements, data_path, base_path, reord_path = built

    worker_script = os.path.join(out_dir, "_perf_worker.py")
    write_worker_script(worker_script, args.query_repeat, args.ef_search)

    b_stats, r_stats = [], []
    for run in range(args.num_runs):
        b_stats.append(run_perf_stat(worker_script, base_path, data_path, dim, num_elements))
        r_stats.append(run_perf_stat(worker_script, reord_path, data_path, dim, num_elements))
        print(f"  run {run + 1}/{args.num_runs} done")

    b_stats = np.array(b_stats)  # columns: loads, misses, miss_rate, cpu_time_ms
    r_stats = np.array(r_stats)
    b_loads, b_misses, b_rate, b_time = b_stats.mean(axis=0)
    r_loads, r_misses, r_rate, r_time = r_stats.mean(axis=0)

    return {
        "Dataset": dataset_key,
        "LLC_Loads_Base": b_loads, "LLC_Loads_Reordered": r_loads,
        "LLC_Misses_Base": b_misses, "LLC_Misses_Reordered": r_misses,
        "LLC_MissRate_Base_pct": b_rate, "LLC_MissRate_Reordered_pct": r_rate,
        "CPU_Time_ms_Base": b_time, "CPU_Time_ms_Reordered": r_time,
        "LLC_Misses_Reduction_pct": (b_misses - r_misses) / b_misses * 100 if b_misses > 0 else 0.0,
        "CPU_Time_Reduction_pct": (b_time - r_time) / b_time * 100 if b_time > 0 else 0.0,
    }


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Reproduce Table 4: L3 cache / CPU time impact of Phase 1 reordering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--datasets-dir", default="./dataset_benchmark")
    p.add_argument("--datasets", nargs="+", default=list(common.PAPER_DATASETS.keys()),
                    choices=list(common.PAPER_DATASETS.keys()))
    p.add_argument("--output-dir", default="./results/table4")

    p.add_argument("--m", type=int, default=32)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--ef-search", type=int, default=200)

    p.add_argument("--num-clusters", type=int, default=512)
    p.add_argument("--sample-ratio", type=float, default=0.05)
    p.add_argument("--n-init", type=int, default=1)
    p.add_argument("--max-iter", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2048)

    p.add_argument("--num-runs", type=int, default=3, help="Repeat measurements and average (paper: 3).")
    p.add_argument("--query-repeat", type=int, default=20,
                    help="Repeat the full test-query batch this many times per perf run.")
    p.add_argument("--skip-perf-check", action="store_true",
                    help="Skip the `perf` availability/permission probe (advanced use only).")
    return p


def main():
    args = build_arg_parser().parse_args()
    if not args.skip_perf_check:
        check_perf_available()
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    for key in args.datasets:
        row = run_dataset(key, common.PAPER_DATASETS[key], args)
        if row:
            rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.output_dir, "table4_hardware_profile.csv")
    df.to_csv(out_csv, index=False)
    print("\n" + "=" * 70)
    print("Table 4: Impact of Cache-Aligned Memory Reordering on L3 Cache and CPU Time")
    print("=" * 70)
    print(df.to_string(index=False))
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
