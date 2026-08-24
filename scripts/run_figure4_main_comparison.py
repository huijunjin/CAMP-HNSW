#!/usr/bin/env python3
"""Reproduce Figure 4: Recall@{1,10,100} vs QPS across ImageNet, LAION,
COCO, and MainSearch (Sec. 4.2 of the paper).

For each dataset, this script:
  1. Applies Phase 1 cache-aligned reordering to the base vectors
     (Algorithm 1).
  2. Builds a base HNSW index (M=32, ef_construction=200) on the reordered
     data -- the "HNSW" curve. Building on the reordered layout isolates
     the topological contribution of Phase 2/3 in the comparison below
     (both curves share the same physical memory layout).
  3. Mines OOD shortcuts from the calibration queries and injects them with
     Joint Heuristic Pruning (Phase 2 + 3, Algorithm 2) -- the "CAMP-HNSW"
     curve.
  4. Optionally runs the RoarGraph baseline if --roar-dir points at a built
     RoarGraph checkout. RoarGraph (and NGFix) are not part of this
     repository's core engine, so this is skipped with a warning if the
     path isn't provided -- the script always reproduces at least the
     central HNSW vs. CAMP-HNSW comparison.
  5. Evaluates every available method at each of --k-values (default
     1, 10, 100) across a dense ef_search sweep, and writes both per-method
     and merged `recall{k}_{dataset}_final_comparison.csv` files.

Example:
    python scripts/run_figure4_main_comparison.py \\
        --datasets-dir ./dataset_benchmark \\
        --datasets LAION COCO \\
        --output-dir ./results/figure4
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


def evaluate_all_k(index, test_q, test_gt, id_mapper, label, csv_prefix, output_dir,
                   ef_list, k_values, build_time):
    """Sweeps ef_search at every requested Recall@K, writing one CSV per K."""
    for k in k_values:
        ours_utils.evaluate_hnsw_variant(
            index, test_q, test_gt, f"{label} (k={k})", ef_list, output_dir,
            id_mapper=id_mapper, csv_name=f"{csv_prefix}_k{k}.csv",
            build_time=build_time, target_k=k,
        )


def merge_final_comparison(out_dir, dataset_key, k, name_map):
    dfs = []
    for method, csv_prefix in name_map.items():
        path = os.path.join(out_dir, f"{csv_prefix}_k{k}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df = df.rename(columns={c: f"{c}_{method}" for c in
                                 ["Recall", "QPS", "Avg_Visited", "Avg_Hops", "Time"]})
        df["EF"] = df["EF"].astype(int)
        dfs.append(df)

    if not dfs:
        return None

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="EF", how="outer")
    merged = merged.sort_values("EF").reset_index(drop=True)

    out_path = os.path.join(out_dir, f"recall{k}_{dataset_key}_final_comparison.csv")
    merged.to_csv(out_path, index=False)
    return out_path


def run_dataset(dataset_key, dataset_file, args):
    print(f"\n{'=' * 70}\nDataset: {dataset_key} ({dataset_file})\n{'=' * 70}")
    hdf5_path = os.path.join(args.datasets_dir, f"{dataset_file}.hdf5")
    if not os.path.exists(hdf5_path):
        print(f"  [skip] dataset file not found: {hdf5_path}")
        return

    out_dir = os.path.join(args.output_dir, dataset_key)
    os.makedirs(out_dir, exist_ok=True)
    ef_list = common.default_ef_search_list()
    name_map = {"HNSW": "hnsw_res"}

    raw_db, test_q, train_q, test_gt, train_gt = common.load_dataset(hdf5_path)

    print("  [Phase 1] Cache-aligned memory reordering...")
    db, old_to_new, new_to_old = ours_utils.reorder_dataset_by_clustering(
        raw_db, num_clusters=args.num_clusters, sample_ratio=args.sample_ratio,
        n_init=args.n_init, max_iter=args.max_iter, batch_size=args.batch_size,
    )
    train_gt_remapped = old_to_new[train_gt]

    # ---- HNSW baseline (reordered layout, no shortcuts) ----
    print("  [HNSW] Building base index...")
    base_index_path = os.path.join(
        out_dir, common.base_index_filename(dataset_key, args.m, args.ef_construction))
    miner = Ours_Miner(m=args.m, ef_construction=args.ef_construction, dim=db.shape[1])
    t0 = time.time()
    if os.path.exists(base_index_path):
        miner.load_index(base_index_path, db)
    else:
        miner.fit(db, save_path=base_index_path)
    hnsw_build_time = time.time() - t0

    hnsw_index = hnswlib.Index(space="ip", dim=db.shape[1])
    hnsw_index.load_index(base_index_path, len(db), False)
    evaluate_all_k(hnsw_index, test_q, test_gt, new_to_old, "HNSW", "hnsw_res", out_dir,
                   ef_list, args.k_values, hnsw_build_time)
    del hnsw_index
    gc.collect()

    # ---- RoarGraph baseline (optional) ----
    if args.roar_dir and os.path.isdir(args.roar_dir):
        name_map["RoarGraph"] = "roar_res"
        roar_params = {"M_sq": args.roar_m_sq, "M_pjbp": args.roar_m_pjbp,
                        "L_pjpq": args.roar_l_pjpq, "T": args.roar_threads}
        for k in args.k_values:
            ours_utils.run_roargraph_logic(
                raw_db, train_q, train_gt, test_q, test_gt,
                args.roar_dir, os.path.join(out_dir, "roar_tmp"), out_dir,
                roar_params, ef_list, target_k=k,
            )
    else:
        print("  [skip] RoarGraph baseline: --roar-dir not given or not found "
              "(RoarGraph is not part of this repository's core engine)")

    # ---- CAMP-HNSW (ours) ----
    print("  [Phase 2] Mining OOD shortcuts...")
    shortcut_path = os.path.join(
        out_dir, common.shortcut_filename(dataset_key, args.budget, args.train_ef, args.top_k))
    ours_additional_time = 0.0
    if not os.path.exists(shortcut_path):
        ours_additional_time += miner.mine_shortcuts(
            train_q, train_gt_remapped,
            budget=args.budget, train_ef=args.train_ef, top_k=args.top_k,
            output_path=shortcut_path,
        )

    print("  [Phase 3] Injecting shortcuts with joint heuristic pruning...")
    ours_index = hnswlib.Index(space="ip", dim=db.shape[1])
    ours_index.load_index(base_index_path, len(db), False)
    ours_index.inject_shortcuts_binary(shortcut_path)
    stats = ours_index.get_injection_stats()
    ours_additional_time += stats.get("pure_inject_time", 0.0)
    print(f"    {stats['shortcuts_survived']:,}/{stats['shortcuts_tried']:,} shortcuts survived "
          f"joint pruning; {stats['hnsw_edges_deleted']:,} base edges evicted.")

    name_map["CAMP-HNSW"] = "ours_res"
    evaluate_all_k(ours_index, test_q, test_gt, new_to_old, "CAMP-HNSW", "ours_res", out_dir,
                   ef_list, args.k_values, ours_additional_time)
    del ours_index, miner
    gc.collect()

    # ---- merge per-method CSVs into recall{k}_{dataset}_final_comparison.csv ----
    for k in args.k_values:
        out_path = merge_final_comparison(out_dir, dataset_key, k, name_map)
        if out_path:
            print(f"  [saved] {out_path}")


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Reproduce Figure 4: Recall vs QPS, CAMP-HNSW vs baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--datasets-dir", default="./dataset_benchmark",
                    help="Directory containing the .hdf5 benchmark files (Table 1).")
    p.add_argument("--datasets", nargs="+", default=list(common.PAPER_DATASETS.keys()),
                    choices=list(common.PAPER_DATASETS.keys()),
                    help="Which paper datasets to run.")
    p.add_argument("--output-dir", default="./results/figure4",
                    help="Where to write per-dataset result CSVs.")
    p.add_argument("--k-values", nargs="+", type=int, default=[1, 10, 100],
                    help="Recall@K targets to evaluate.")

    p.add_argument("--m", type=int, default=32, help="HNSW max out-degree M.")
    p.add_argument("--ef-construction", type=int, default=200)

    p.add_argument("--budget", type=int, default=1000,
                    help="Phase 3 joint-pruning max candidate pool per node.")
    p.add_argument("--train-ef", type=int, default=150,
                    help="Phase 2 mining search depth (ef_mine in the paper).")
    p.add_argument("--top-k", type=int, default=70,
                    help="Phase 2 mining top-k (N_mine in the paper).")

    p.add_argument("--num-clusters", type=int, default=512,
                    help="Phase 1 MiniBatchKMeans cluster count (K).")
    p.add_argument("--sample-ratio", type=float, default=0.05)
    p.add_argument("--n-init", type=int, default=1)
    p.add_argument("--max-iter", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2048)

    p.add_argument("--roar-dir", default=None,
                    help="Path to a built RoarGraph checkout (optional; skipped if omitted).")
    p.add_argument("--roar-m-sq", type=int, default=100)
    p.add_argument("--roar-m-pjbp", type=int, default=32)
    p.add_argument("--roar-l-pjpq", type=int, default=200)
    p.add_argument("--roar-threads", type=int, default=32)
    return p


def main():
    args = build_arg_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for key in args.datasets:
        run_dataset(key, common.PAPER_DATASETS[key], args)


if __name__ == "__main__":
    main()
