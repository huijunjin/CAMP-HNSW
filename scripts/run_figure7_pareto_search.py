#!/usr/bin/env python3
"""Reproduce Figure 7: the global Pareto frontier of average build time vs.
average recall over the Phase 2 mining parameters ef_mine (--train-ef-list)
and N_mine (--top-k-list) (Sec. 4.6).

For each dataset, builds one base HNSW index, then for every (ef_mine,
N_mine) combination mines shortcuts, injects them into a fresh copy of the
base index, and records the pure build time and the max recall reached
across the ef_search sweep. Results are averaged across --datasets (the
paper uses ImageNet and LAION, the same two 1M-scale reference sets as
Figure 6) to produce the frontier plotted in Figure 7.

The paper's own grid (10 x 10 = 100 combinations per dataset) is
expensive; the default here is a reduced grid centered on the paper's
selected operating point (ef_mine=150, N_mine=70). Pass the full paper grid
explicitly for an exact reproduction, e.g.:
    --train-ef-list 10 30 50 100 150 200 250 300 400 500
    --top-k-list 10 20 30 40 50 60 70 80 90 100

Example:
    python scripts/run_figure7_pareto_search.py \\
        --datasets-dir ./dataset_benchmark \\
        --output-dir ./results/figure7
"""
import argparse
import gc
import os
import time

import common
import hnswlib
import pandas as pd

from ours_model import Ours_Miner
import ours_utils


def run_dataset(dataset_key, dataset_file, args):
    print(f"\n{'=' * 70}\nDataset: {dataset_key} ({dataset_file})\n{'=' * 70}")
    hdf5_path = os.path.join(args.datasets_dir, f"{dataset_file}.hdf5")
    if not os.path.exists(hdf5_path):
        print(f"  [skip] dataset file not found: {hdf5_path}")
        return None

    out_dir = os.path.join(args.output_dir, dataset_key)
    os.makedirs(out_dir, exist_ok=True)
    ef_list = args.eval_ef_list

    raw_db, test_q, train_q, test_gt, train_gt = common.load_dataset(hdf5_path)

    db, old_to_new, new_to_old = ours_utils.reorder_dataset_by_clustering(
        raw_db, num_clusters=args.num_clusters, sample_ratio=args.sample_ratio,
        n_init=args.n_init, max_iter=args.max_iter, batch_size=args.batch_size,
    )
    train_gt_remapped = old_to_new[train_gt]

    base_index_path = os.path.join(
        out_dir, common.base_index_filename(dataset_key, args.m, args.ef_construction))
    miner = Ours_Miner(m=args.m, ef_construction=args.ef_construction, dim=db.shape[1])
    if os.path.exists(base_index_path):
        miner.load_index(base_index_path, db)
    else:
        miner.fit(db, save_path=base_index_path)

    combos = [(ef, k) for ef in args.train_ef_list for k in args.top_k_list]
    print(f"  Testing {len(combos)} (ef_mine, N_mine) combinations...")

    rows = []
    for i, (train_ef, top_k) in enumerate(combos):
        shortcut_path = os.path.join(
            out_dir, common.shortcut_filename(dataset_key, args.budget, train_ef, top_k))

        t0 = time.time()
        if not os.path.exists(shortcut_path):
            miner.mine_shortcuts(
                train_q, train_gt_remapped,
                budget=args.budget, train_ef=train_ef, top_k=top_k,
                output_path=shortcut_path,
            )
        mine_time = time.time() - t0

        eval_index = hnswlib.Index(space="ip", dim=db.shape[1])
        eval_index.load_index(base_index_path, len(db), False)
        eval_index.inject_shortcuts_binary(shortcut_path)
        inject_stats = eval_index.get_injection_stats()
        build_time = mine_time + inject_stats.get("pure_inject_time", 0.0)

        res = ours_utils.evaluate_hnsw_variant(
            eval_index, test_q, test_gt, f"ef{train_ef}_k{top_k}", ef_list, out_dir,
            id_mapper=new_to_old, csv_name=f"{dataset_key}_ef{train_ef}_k{top_k}.csv",
            build_time=build_time, target_k=args.target_k,
        )
        max_recall = max(res["recall"]) if res["recall"] else 0.0

        print(f"  [{i + 1:03d}/{len(combos)}] ef_mine={train_ef:<4} N_mine={top_k:<4} "
              f"-> build {build_time:>6.2f}s, max recall {max_recall:>6.2f}%")

        rows.append({
            "Dataset": dataset_key, "ef_mine": train_ef, "N_mine": top_k,
            "build_time_s": round(build_time, 2), "max_recall_pct": round(max_recall, 2),
        })

        del eval_index
        gc.collect()

    del miner
    gc.collect()

    df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, f"{dataset_key}_pareto_grid.csv")
    df.to_csv(out_path, index=False)
    print(f"  [saved] {out_path}")
    return df


def compute_global_frontier(per_dataset_dfs, output_dir):
    """Averages (build_time, max_recall) for each (ef_mine, N_mine) across
    all datasets that were run, then extracts the Pareto-optimal frontier
    (no other point reaches both a lower time and a higher recall)."""
    if not per_dataset_dfs:
        return

    all_df = pd.concat(per_dataset_dfs, ignore_index=True)
    grouped = all_df.groupby(["ef_mine", "N_mine"]).agg(
        avg_build_time_s=("build_time_s", "mean"),
        avg_max_recall_pct=("max_recall_pct", "mean"),
        dataset_count=("Dataset", "nunique"),
    ).reset_index()

    grouped = grouped.sort_values("avg_build_time_s").reset_index(drop=True)
    frontier_mask = []
    best_recall_so_far = -1.0
    for _, row in grouped.iterrows():
        is_frontier = row["avg_max_recall_pct"] > best_recall_so_far
        frontier_mask.append(is_frontier)
        if is_frontier:
            best_recall_so_far = row["avg_max_recall_pct"]
    grouped["is_pareto_optimal"] = frontier_mask

    out_path = os.path.join(output_dir, "figure7_global_pareto_frontier.csv")
    grouped.sort_values("avg_max_recall_pct", ascending=False).to_csv(out_path, index=False)
    print(f"\n[saved] {out_path}")

    frontier = grouped[grouped["is_pareto_optimal"]].sort_values("avg_build_time_s")
    print("\nPareto frontier (avg build time vs avg max recall):")
    print(frontier.to_string(index=False))


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Reproduce Figure 7: global Pareto frontier over Phase 2 mining parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--datasets-dir", default="./dataset_benchmark")
    p.add_argument("--datasets", nargs="+", default=["ImageNet", "LAION"],
                    choices=list(common.PAPER_DATASETS.keys()),
                    help="Paper derives the default config on ImageNet and LAION only.")
    p.add_argument("--output-dir", default="./results/figure7")
    p.add_argument("--target-k", type=int, default=10)

    p.add_argument("--m", type=int, default=32)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--eval-ef-list", nargs="+", type=int, default=[100, 150, 200, 250, 300],
                    help="ef_search sweep used to find the max recall for each combination.")

    p.add_argument("--budget", type=int, default=1000)
    p.add_argument("--train-ef-list", nargs="+", type=int, default=[60, 100, 150, 200],
                    help="ef_mine candidates (paper's full grid: 10 30 50 100 150 200 250 300 400 500).")
    p.add_argument("--top-k-list", nargs="+", type=int, default=[30, 60, 70, 80, 100],
                    help="N_mine candidates (paper's full grid: 10 20 30 40 50 60 70 80 90 100).")

    p.add_argument("--num-clusters", type=int, default=512)
    p.add_argument("--sample-ratio", type=float, default=0.05)
    p.add_argument("--n-init", type=int, default=1)
    p.add_argument("--max-iter", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2048)
    return p


def main():
    args = build_arg_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    per_dataset_dfs = []
    for key in args.datasets:
        df = run_dataset(key, common.PAPER_DATASETS[key], args)
        if df is not None:
            per_dataset_dfs.append(df)

    compute_global_frontier(per_dataset_dfs, args.output_dir)


if __name__ == "__main__":
    main()
