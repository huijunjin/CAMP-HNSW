#!/usr/bin/env python3
"""Reproduce Figure 6: parameter sensitivity of Phase 1 Cache-Aligned Memory
Reordering to the MiniBatchKMeans sample ratio and cluster count (Sec. 4.6).

For each dataset, this measures the QPS of a raw (unreordered) HNSW index as
a baseline, then sweeps every (K, sample_ratio) combination in
--k-candidates x --sample-ratio-candidates (max_iter, n_init, batch_size held
fixed at their paper defaults) and records the QPS gain over the raw
baseline, and the reordering time. The paper's own parameter search
(Sec. 4.6) was run on the two 1M-scale reference datasets (ImageNet, LAION)
to avoid overfitting the default configuration to the larger sets -- that's
this script's default --datasets too.

Example:
    python scripts/run_figure6_kmeans_tuning.py \\
        --datasets-dir ./dataset_benchmark \\
        --output-dir ./results/figure6
"""
import argparse
import gc
import itertools
import os
import time

import common
import h5py
import hnswlib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


def measure_qps(index, test_q, ef_list, repeats):
    """Median-of-`repeats` QPS at each ef in ef_list, then averaged over
    ef_list -- matches the paper's outlier-robust measurement protocol."""
    qps_per_ef = []
    for ef in ef_list:
        index.set_ef(ef)
        trial_qps = []
        for _ in range(repeats):
            t0 = time.time()
            index.knn_query(test_q, k=10, num_threads=1)
            trial_qps.append(len(test_q) / (time.time() - t0))
        qps_per_ef.append(float(np.median(trial_qps)))
    return float(np.mean(qps_per_ef))


def run_dataset(dataset_key, dataset_file, args):
    print(f"\n{'=' * 70}\nDataset: {dataset_key} ({dataset_file})\n{'=' * 70}")
    hdf5_path = os.path.join(args.datasets_dir, f"{dataset_file}.hdf5")
    if not os.path.exists(hdf5_path):
        print(f"  [skip] dataset file not found: {hdf5_path}")
        return None

    with h5py.File(hdf5_path, "r") as f:
        raw_db = np.array(f["train"]).astype(np.float32)
        test_q = np.array(f["test"]).astype(np.float32)
    num_elements, dim = raw_db.shape

    print("  Measuring raw (unreordered) baseline QPS...")
    index_raw = hnswlib.Index(space="ip", dim=dim)
    index_raw.init_index(max_elements=num_elements, ef_construction=args.ef_construction, M=args.m)
    index_raw.add_items(raw_db, np.arange(num_elements))
    avg_qps_raw = measure_qps(index_raw, test_q, args.ef_list, args.repeats)
    del index_raw
    gc.collect()
    print(f"  Raw baseline avg QPS: {avg_qps_raw:.1f}")

    combinations = list(itertools.product(args.k_candidates, args.sample_ratio_candidates))
    print(f"  Testing {len(combinations)} (K, sample_ratio) combinations "
          f"(fixed: iter={args.max_iter}, n_init={args.n_init}, batch={args.batch_size})")

    rows = []
    for i, (k_val, s_ratio) in enumerate(combinations):
        sample_size = int(num_elements * s_ratio)
        if sample_size < k_val:
            print(f"  [{i + 1:02d}/{len(combinations)}] K={k_val:<4} Sam={s_ratio * 100:>2.0f}% "
                  f"-- skipped (sample size {sample_size} < K)")
            continue

        t0 = time.time()
        sampled_db = raw_db[np.random.RandomState(42).choice(num_elements, sample_size, replace=False)]
        kmeans = MiniBatchKMeans(n_clusters=k_val, batch_size=args.batch_size, random_state=42,
                                  n_init=args.n_init, max_iter=args.max_iter)
        kmeans.fit(sampled_db)
        cluster_labels = kmeans.predict(raw_db)
        sorted_db = raw_db[np.argsort(cluster_labels)]
        reorder_time = time.time() - t0

        index_sorted = hnswlib.Index(space="ip", dim=dim)
        index_sorted.init_index(max_elements=num_elements, ef_construction=args.ef_construction, M=args.m)
        index_sorted.add_items(sorted_db, np.arange(num_elements))
        avg_qps_sorted = measure_qps(index_sorted, test_q, args.ef_list, args.repeats)
        del index_sorted
        gc.collect()

        gain = (avg_qps_sorted - avg_qps_raw) / avg_qps_raw * 100 if avg_qps_raw > 0 else 0.0
        print(f"  [{i + 1:02d}/{len(combinations)}] K={k_val:<4} Sam={s_ratio * 100:>2.0f}% "
              f"-> time {reorder_time:>5.2f}s, QPS {avg_qps_sorted:>7.1f} ({gain:+.1f}%)")

        rows.append({
            "K": k_val, "Sam(%)": int(s_ratio * 100), "Iter": args.max_iter,
            "Init": args.n_init, "Batch": args.batch_size,
            "Time(s)": round(reorder_time, 2), "QPS": round(avg_qps_sorted, 2),
            "Gain(%)": round(gain, 2),
        })

    df = pd.DataFrame(rows).sort_values(by=["Gain(%)", "Time(s)"], ascending=[False, True]).reset_index(drop=True)
    out_path = os.path.join(args.output_dir, f"{dataset_key}_cache_reorder_grid_search_result.csv")
    df.to_csv(out_path, index=False)
    print(f"  [saved] {out_path}")
    return df


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Reproduce Figure 6: Phase 1 reordering parameter sensitivity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--datasets-dir", default="./dataset_benchmark")
    p.add_argument("--datasets", nargs="+", default=["ImageNet", "LAION"],
                    choices=list(common.PAPER_DATASETS.keys()),
                    help="Paper derives the default config on ImageNet and LAION only.")
    p.add_argument("--output-dir", default="./results/figure6")

    p.add_argument("--m", type=int, default=32)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--ef-list", nargs="+", type=int, default=[100, 200, 300, 400],
                    help="ef_search values averaged over for each QPS measurement.")
    p.add_argument("--repeats", type=int, default=3, help="Median-of-N repeats per ef (outlier robustness).")

    p.add_argument("--k-candidates", nargs="+", type=int, default=[32, 64, 128, 256, 512, 1024, 2048])
    p.add_argument("--sample-ratio-candidates", nargs="+", type=float, default=[0.01, 0.05, 0.10, 0.20])
    p.add_argument("--max-iter", type=int, default=50)
    p.add_argument("--n-init", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2048)
    return p


def main():
    args = build_arg_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for key in args.datasets:
        run_dataset(key, common.PAPER_DATASETS[key], args)


if __name__ == "__main__":
    main()
