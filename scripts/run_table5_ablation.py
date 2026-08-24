#!/usr/bin/env python3
"""Reproduce Table 5: the incremental ablation study isolating each
CAMP-HNSW component's contribution (Sec. 4.5).

For each dataset, this builds six index variants that share the same base
construction (M=32, ef_construction=200) but differ in which components are
enabled, using the `USE_CANDIDATE_PRUNING` / `USE_JOINT_PRUNING` environment
switches wired into `injectShortcutsBinary` / `mine_and_filter_cpp`:

  v0  Base HNSW        -- no mining, no injection.
  v1  Naive Injection  -- shortcuts mined and injected with no RNG heuristic
                          at either stage (Sec 4.5: "additive densification").
  v2  Cand Pruning     -- RNG pre-filtering of mined candidates only
                          (Algorithm 2, Stage 1), naive graph-side injection.
  v3  Joint Pruning    -- full Phase 2+3 topological rewiring, unreordered
                          base layout.
  v4  Memory Reorder   -- v3 + Phase 1 cache-aligned reordering (this is the
                          complete CAMP-HNSW configuration).
  v5  Phase 1 Only     -- reordering with no mining/injection at all, to
                          isolate the hardware-only contribution.

QPS is measured 5 times per variant (matching the paper's "5 independent
runs, median reported") and the median, mean, max, min, and std are all
saved, alongside a per-dataset summary interpolated at Recall@95% and
Recall@99%.

Example:
    python scripts/run_table5_ablation.py \\
        --datasets-dir ./dataset_benchmark \\
        --datasets LAION COCO \\
        --output-dir ./results/table5
"""
import argparse
import gc
import os
import time

import common
import hnswlib
import numpy as np
import pandas as pd

from ours_model import Ours_Miner
import ours_utils


VARIANTS = [
    {"id": 0, "name": "Base HNSW",       "use_cand": "1", "use_joint": "1", "reorder": False, "mine": False},
    {"id": 1, "name": "Naive Injection", "use_cand": "0", "use_joint": "0", "reorder": False, "mine": True},
    {"id": 2, "name": "Cand Pruning",    "use_cand": "1", "use_joint": "0", "reorder": False, "mine": True},
    {"id": 3, "name": "Joint Pruning",   "use_cand": "1", "use_joint": "1", "reorder": False, "mine": True},
    {"id": 4, "name": "Memory Reorder",  "use_cand": "1", "use_joint": "1", "reorder": True,  "mine": True},
    {"id": 5, "name": "Phase 1 Only",    "use_cand": "0", "use_joint": "0", "reorder": True,  "mine": False},
]


def interpolate_qps_at_recall(df, target_recall):
    """Extracts the Pareto-optimal (recall, QPS) frontier and linearly
    interpolates the QPS achievable at `target_recall`. Returns 0.0 if the
    target recall is never reached (reported as "Fail" in the table)."""
    if df.empty:
        return np.nan

    df_sorted = df.sort_values(by="qps", ascending=False).reset_index(drop=True)
    pareto = []
    max_recall = -1.0
    for _, row in df_sorted.iterrows():
        if row["recall"] > max_recall:
            pareto.append(row)
            max_recall = row["recall"]
    df_pareto = pd.DataFrame(pareto).sort_values("recall")

    recalls = df_pareto["recall"].values
    qps_vals = df_pareto["qps"].values
    if target_recall > recalls.max():
        return 0.0
    if target_recall in recalls:
        return float(df_pareto[df_pareto["recall"] == target_recall]["qps"].iloc[0])
    return float(np.interp(target_recall, recalls, qps_vals))


def run_dataset(dataset_key, dataset_file, args, final_table_rows):
    print(f"\n{'=' * 70}\nDataset: {dataset_key} ({dataset_file})\n{'=' * 70}")
    hdf5_path = os.path.join(args.datasets_dir, f"{dataset_file}.hdf5")
    if not os.path.exists(hdf5_path):
        print(f"  [skip] dataset file not found: {hdf5_path}")
        return

    out_dir = os.path.join(args.output_dir, dataset_key)
    os.makedirs(out_dir, exist_ok=True)
    ef_list = common.default_ef_search_list()

    raw_db, test_q, train_q, test_gt, train_gt = common.load_dataset(hdf5_path)
    num_nodes = len(raw_db)

    t0 = time.time()
    reorder_db, old_to_new, new_to_old, sorted_labels = ours_utils.reorder_dataset_by_clustering(
        raw_db, num_clusters=args.num_clusters, sample_ratio=args.sample_ratio,
        n_init=args.n_init, max_iter=args.max_iter, batch_size=args.batch_size,
    )
    train_gt_remapped = old_to_new[train_gt]
    reorder_time = time.time() - t0

    base_row_idx = -1

    for var in VARIANTS:
        v_id, v_name = var["id"], var["name"]
        print(f"\n--- Variant {v_id}: {v_name} ---")

        os.environ["USE_CANDIDATE_PRUNING"] = var["use_cand"]
        os.environ["USE_JOINT_PRUNING"] = var["use_joint"]

        current_db = reorder_db if var["reorder"] else raw_db
        current_train_gt = train_gt_remapped if var["reorder"] else train_gt
        mapper = new_to_old if var["reorder"] else None

        index_path = os.path.join(out_dir, f"{dataset_key}_v{v_id}.bin")
        shortcut_path = os.path.join(out_dir, f"{dataset_key}_v{v_id}_shortcuts.bin")

        total_build_time = reorder_time if var["reorder"] else 0.0

        t0 = time.time()
        miner = Ours_Miner(m=args.m, ef_construction=args.ef_construction, dim=current_db.shape[1])
        miner.fit(current_db, save_path=index_path)
        total_build_time += time.time() - t0

        index = hnswlib.Index(space="ip", dim=current_db.shape[1])
        index.load_index(index_path, len(current_db), False)

        avg_degree = 0.0
        if var["mine"]:
            pure_mine_time = miner.train_frequency_shortcuts(
                train_q, current_train_gt, sorted_labels if var["reorder"] else None,
                budget=args.budget, train_ef=args.train_ef, top_k=args.top_k,
                output_path=shortcut_path,
            )
            total_build_time += pure_mine_time

            index.inject_shortcuts_binary(shortcut_path)
            stats = index.get_injection_stats()
            total_build_time += stats.get("pure_inject_time", 0.0)
            avg_degree = stats["final_edges"] / num_nodes

            # Variant 1 (naive injection) always runs before we've recorded
            # v0's true base degree from a mining pass; back-fill it here so
            # the "Base HNSW" row reports the real average out-degree rather
            # than 0.
            if v_id == 1 and base_row_idx != -1:
                final_table_rows[base_row_idx]["Avg Degree"] = round(stats["original_edges"] / num_nodes, 1)
        elif v_id == 5 and base_row_idx != -1:
            # Phase 1 Only never mines, so its degree equals the base graph's.
            avg_degree = final_table_rows[base_row_idx]["Avg Degree"]

        qps_history = []
        for run_idx in range(args.num_runs):
            run_csv = f"{dataset_key}_v{v_id}_run{run_idx + 1}_res.csv"
            res = ours_utils.evaluate_hnsw_variant(
                index, test_q, test_gt, f"v{v_id} run{run_idx + 1}", ef_list, out_dir,
                id_mapper=mapper, csv_name=run_csv, build_time=0.0, target_k=args.target_k,
            )
            qps_history.append(res["qps"])

        qps_array = np.array(qps_history)
        stats_df = pd.DataFrame({
            "Recall": res["recall"],
            "QPS_Median": np.median(qps_array, axis=0),
            "QPS_Mean": np.mean(qps_array, axis=0),
            "QPS_Max": np.max(qps_array, axis=0),
            "QPS_Min": np.min(qps_array, axis=0),
            "QPS_Std": np.std(qps_array, axis=0),
        })
        stats_df.to_csv(os.path.join(out_dir, f"{dataset_key}_v{v_id}_stats_summary.csv"), index=False)

        df_res = pd.DataFrame({"recall": res["recall"], "qps": np.median(qps_array, axis=0).tolist()})
        qps_95 = interpolate_qps_at_recall(df_res, 95.0)
        qps_99 = interpolate_qps_at_recall(df_res, 99.0)

        final_table_rows.append({
            "Dataset": dataset_key,
            "Variant": v_name,
            "Avg Degree": round(avg_degree, 1),
            "Time (s)": round(total_build_time, 1),
            "QPS @95%": round(qps_95) if qps_95 > 0 else "Fail",
            "QPS @99%": round(qps_99) if qps_99 > 0 else "Fail",
        })
        if v_id == 0:
            base_row_idx = len(final_table_rows) - 1

        del index, miner
        gc.collect()


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Reproduce Table 5: incremental ablation study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--datasets-dir", default="./dataset_benchmark")
    p.add_argument("--datasets", nargs="+", default=list(common.PAPER_DATASETS.keys()),
                    choices=list(common.PAPER_DATASETS.keys()))
    p.add_argument("--output-dir", default="./results/table5")
    p.add_argument("--target-k", type=int, default=10, help="Recall@K basis for the table (paper: 10).")
    p.add_argument("--num-runs", type=int, default=5, help="QPS repeats per variant (paper: 5, median reported).")

    p.add_argument("--m", type=int, default=32)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--budget", type=int, default=1000)
    p.add_argument("--train-ef", type=int, default=150)
    p.add_argument("--top-k", type=int, default=70)

    p.add_argument("--num-clusters", type=int, default=512)
    p.add_argument("--sample-ratio", type=float, default=0.05)
    p.add_argument("--n-init", type=int, default=1)
    p.add_argument("--max-iter", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2048)
    return p


def main():
    args = build_arg_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    final_table_rows = []
    for key in args.datasets:
        run_dataset(key, common.PAPER_DATASETS[key], args, final_table_rows)

    df_final = pd.DataFrame(final_table_rows)
    table_path = os.path.join(args.output_dir, "final_ablation_table.csv")
    df_final.to_csv(table_path, index=False)
    print("\n" + "=" * 70)
    print("Table 5: Incremental Ablation Study")
    print("=" * 70)
    print(df_final.to_string(index=False))
    print(f"\nSaved: {table_path}")


if __name__ == "__main__":
    main()
