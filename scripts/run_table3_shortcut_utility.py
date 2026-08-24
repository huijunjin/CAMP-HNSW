#!/usr/bin/env python3
"""Reproduce Table 2 (Graph Edge Composition Before and After Joint
Heuristic Pruning) and Table 3 (In-Depth Shortcut Utility and Trap Escape
Analysis), Sec. 4.3.

Both tables characterize CAMP-HNSW's own topology and search behavior
rather than comparing against baselines, so this script builds a single
CAMP-HNSW index per dataset (Phase 1 reorder -> Phase 2 mine -> Phase 3
inject) and reads off two kinds of built-in instrumentation:

  Table 2 (`get_injection_stats()`): edge counts before/after Phase 3 --
  how many base HNSW edges existed, how many shortcut candidates were
  mined, and how many of each survived the joint competitive pruning.

  Table 3 (`enable_profiling()` / `get_profiling_stats()` + a base-vs-ours
  recall comparison): for the same test queries,
    - Access Freq.  = share of search hops that traversed a surviving
                       shortcut edge rather than a base edge (Sec. 4.3:
                       "comprising 23% to 44% of search hops").
    - Trapped       = queries where the base HNSW graph (no shortcuts)
                       fails to retrieve the full Recall@target_k ground
                       truth -- i.e. genuinely hits a local-minimum trap.
    - Escaped       = of those trapped queries, how many CAMP-HNSW fully
                       recovers (reaches Recall@target_k = 100%) once the
                       surviving shortcuts are in the graph.
    - Escape Rate   = Escaped / Trapped.
  ("Trapped"/"Escaped" are not separately exposed by the C++ engine, so
  this script computes them directly from the base-vs-CAMP-HNSW knn_query
  results against ground truth -- the most direct, literal reading of the
  paper's "trap escape" language.)

Example:
    python scripts/run_table3_shortcut_utility.py \\
        --datasets-dir ./dataset_benchmark \\
        --output-dir ./results/table3
"""
import argparse
import gc
import os

import common
import hnswlib
import numpy as np
import pandas as pd

from ours_model import Ours_Miner
import ours_utils


def run_dataset(dataset_key, dataset_file, args):
    print(f"\n{'=' * 70}\nDataset: {dataset_key} ({dataset_file})\n{'=' * 70}")
    hdf5_path = os.path.join(args.datasets_dir, f"{dataset_file}.hdf5")
    if not os.path.exists(hdf5_path):
        print(f"  [skip] dataset file not found: {hdf5_path}")
        return None, None

    out_dir = os.path.join(args.output_dir, dataset_key)
    os.makedirs(out_dir, exist_ok=True)

    raw_db, test_q, train_q, test_gt, train_gt = common.load_dataset(hdf5_path)
    test_gt_k = test_gt[:, :args.target_k]

    db, old_to_new, new_to_old, sorted_labels = ours_utils.reorder_dataset_by_clustering(
        raw_db, num_clusters=args.num_clusters, sample_ratio=args.sample_ratio,
        n_init=args.n_init, max_iter=args.max_iter, batch_size=args.batch_size,
    )
    train_gt_remapped = old_to_new[train_gt]

    base_index_path = os.path.join(out_dir, f"{dataset_key}_base.bin")
    miner = Ours_Miner(m=args.m, ef_construction=args.ef_construction, dim=db.shape[1])
    if os.path.exists(base_index_path):
        miner.load_index(base_index_path, db)
    else:
        miner.fit(db, save_path=base_index_path)

    # ---- Base HNSW pass: which queries are genuinely trapped? ----
    base_index = hnswlib.Index(space="ip", dim=db.shape[1])
    base_index.load_index(base_index_path, len(db), False)
    base_index.set_ef(args.ef_search)
    base_labels, _ = base_index.knn_query(test_q, k=args.target_k, num_threads=1)
    base_labels_old = new_to_old[base_labels]
    del base_index
    gc.collect()

    # ---- Phase 2 + 3: mine and inject shortcuts (full CAMP-HNSW) ----
    shortcut_path = os.path.join(out_dir, f"{dataset_key}_shortcuts.bin")
    if not os.path.exists(shortcut_path):
        miner.train_frequency_shortcuts(
            train_q, train_gt_remapped, sorted_labels,
            budget=args.budget, train_ef=args.train_ef, top_k=args.top_k,
            output_path=shortcut_path,
        )

    ours_index = hnswlib.Index(space="ip", dim=db.shape[1])
    ours_index.load_index(base_index_path, len(db), False)
    ours_index.inject_shortcuts_binary(shortcut_path)
    injection_stats = ours_index.get_injection_stats()

    # ---- Table 3: profile edge activity + recompute recall with shortcuts ----
    ours_index.set_ef(args.ef_search)
    ours_index.enable_profiling()
    ours_labels, _ = ours_index.knn_query(test_q, k=args.target_k, num_threads=1)
    profiling_stats = ours_index.get_profiling_stats()
    ours_labels_old = new_to_old[ours_labels]
    del ours_index, miner
    gc.collect()

    # ---- Table 2: edge composition ----
    original_edges = injection_stats["original_edges"]
    shortcuts_tried = injection_stats["shortcuts_tried"]
    shortcuts_survived = injection_stats["shortcuts_survived"]
    final_edges = injection_stats["final_edges"]
    refined_hnsw_edges = final_edges - shortcuts_survived

    table2_row = {
        "Dataset": dataset_key,
        "Base_HNSW_Edges": original_edges,
        "Mined_Shortcuts": shortcuts_tried,
        "Refined_HNSW_Edges": refined_hnsw_edges,
        "Refined_HNSW_Reduction_pct": round(
            (original_edges - refined_hnsw_edges) / original_edges * 100, 1) if original_edges else 0.0,
        "OOD_Shortcuts_Survived": shortcuts_survived,
        "OOD_Shortcuts_Reduction_pct": round(
            (shortcuts_tried - shortcuts_survived) / shortcuts_tried * 100, 1) if shortcuts_tried else 0.0,
    }

    # ---- Table 3: access frequency + trap escape ----
    hop_base = profiling_stats["hop_base"]
    hop_shortcut = profiling_stats["hop_shortcut"]
    total_hops = hop_base + hop_shortcut
    access_freq_pct = (hop_shortcut / total_hops * 100) if total_hops > 0 else 0.0

    trapped = 0
    escaped = 0
    for i in range(len(test_q)):
        gt_set = set(test_gt_k[i].tolist())
        base_correct = len(gt_set.intersection(base_labels_old[i].tolist()))
        if base_correct < args.target_k:
            trapped += 1
            ours_correct = len(gt_set.intersection(ours_labels_old[i].tolist()))
            if ours_correct >= args.target_k:
                escaped += 1

    escape_rate_pct = (escaped / trapped * 100) if trapped > 0 else 0.0

    table3_row = {
        "Dataset": dataset_key,
        "Access_Freq_pct": round(access_freq_pct, 1),
        "Trapped": trapped,
        "Escaped": escaped,
        "Escape_Rate_pct": round(escape_rate_pct, 1),
    }

    print(f"  [Table 2] Base HNSW: {original_edges:,} edges | Mined: {shortcuts_tried:,} | "
          f"Refined HNSW: {refined_hnsw_edges:,} ({table2_row['Refined_HNSW_Reduction_pct']}% reduction) | "
          f"OOD Shortcuts survived: {shortcuts_survived:,} ({table2_row['OOD_Shortcuts_Reduction_pct']}% dropped)")
    print(f"  [Table 3] Access Freq: {access_freq_pct:.1f}% | Trapped: {trapped} | "
          f"Escaped: {escaped} | Escape Rate: {escape_rate_pct:.1f}%")

    return table2_row, table3_row


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Reproduce Table 2 (edge composition) and Table 3 (shortcut utility / trap escape).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--datasets-dir", default="./dataset_benchmark")
    p.add_argument("--datasets", nargs="+", default=list(common.PAPER_DATASETS.keys()),
                    choices=list(common.PAPER_DATASETS.keys()))
    p.add_argument("--output-dir", default="./results/table3")
    p.add_argument("--target-k", type=int, default=10, help="Recall@K basis for trap/escape (paper: 10).")
    p.add_argument("--ef-search", type=int, default=200, help="ef used for the profiled/trap-escape search pass.")

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

    table2_rows, table3_rows = [], []
    for key in args.datasets:
        row2, row3 = run_dataset(key, common.PAPER_DATASETS[key], args)
        if row2 is not None:
            table2_rows.append(row2)
            table3_rows.append(row3)

    df2 = pd.DataFrame(table2_rows)
    df3 = pd.DataFrame(table3_rows)
    path2 = os.path.join(args.output_dir, "table2_edge_composition.csv")
    path3 = os.path.join(args.output_dir, "table3_shortcut_utility.csv")
    df2.to_csv(path2, index=False)
    df3.to_csv(path3, index=False)

    print("\n" + "=" * 70)
    print("Table 2: Graph Edge Composition Before and After Joint Heuristic Pruning")
    print("=" * 70)
    print(df2.to_string(index=False))
    print(f"\nSaved: {path2}")

    print("\n" + "=" * 70)
    print("Table 3: In-Depth Shortcut Utility and Trap Escape Analysis")
    print("=" * 70)
    print(df3.to_string(index=False))
    print(f"\nSaved: {path3}")


if __name__ == "__main__":
    main()
