#!/usr/bin/env python3
"""Reproduce Figure 9: data efficiency under calibration-query cold-start,
i.e. QPS@99% as the fraction of available calibration queries used for
Phase 2 mining is varied from 10% to 100% (Sec. 4.7).

For each dataset, this builds the base HNSW index once, then for each ratio
in --ratios slices the first `ratio * len(train_q)` calibration queries,
mines shortcuts from just that slice, injects them, and records the
interpolated QPS at Recall@90/95/99%. RoarGraph is included if --roar-dir
is given (optional; RoarGraph is not part of this repository's core
engine).

Example:
    python scripts/run_figure9_query_ratio.py \\
        --datasets-dir ./dataset_benchmark \\
        --datasets LAION \\
        --output-dir ./results/figure9
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


def qps_at_recall(res_dict, target_recall):
    """Recall-matched QPS for one evaluate_hnsw_variant() result dict."""
    if not res_dict:
        return np.nan
    return common.qps_at_recall(res_dict["recall"], res_dict["qps"],
                                target_recall, default=np.nan)


def run_dataset(dataset_key, dataset_file, args):
    print(f"\n{'=' * 70}\nDataset: {dataset_key} ({dataset_file})\n{'=' * 70}")
    hdf5_path = os.path.join(args.datasets_dir, f"{dataset_file}.hdf5")
    if not os.path.exists(hdf5_path):
        print(f"  [skip] dataset file not found: {hdf5_path}")
        return

    out_dir = os.path.join(args.output_dir, dataset_key)
    os.makedirs(out_dir, exist_ok=True)
    ef_list = common.default_ef_search_list()

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

    hnsw_index = hnswlib.Index(space="ip", dim=db.shape[1])
    hnsw_index.load_index(base_index_path, len(db), False)
    res_hnsw = ours_utils.evaluate_hnsw_variant(
        hnsw_index, test_q, test_gt, "HNSW (baseline)", ef_list, out_dir,
        id_mapper=new_to_old, csv_name=f"{dataset_key}_hnsw_res_base.csv",
        build_time=0.0, target_k=args.target_k,
    )
    del hnsw_index

    hnsw_qps = {r: qps_at_recall(res_hnsw, r) for r in (90.0, 95.0, 99.0)}

    summary_rows = []
    total_queries = len(train_q)

    for ratio in args.ratios:
        ratio_pct = int(round(ratio * 100))
        num_q = max(1, int(total_queries * ratio))
        print(f"\n--- Ratio {ratio_pct}% ({num_q}/{total_queries} calibration queries) ---")

        sliced_train_q = train_q[:num_q]
        sliced_train_gt_remapped = train_gt_remapped[:num_q]

        summary_rows.append({"Ratio(%)": ratio_pct, "Method": "HNSW",
                              "QPS@90": hnsw_qps[90.0], "QPS@95": hnsw_qps[95.0], "QPS@99": hnsw_qps[99.0]})

        if args.roar_dir and os.path.isdir(args.roar_dir):
            roar_params = {"M_sq": args.roar_m_sq, "M_pjbp": args.roar_m_pjbp,
                            "L_pjpq": args.roar_l_pjpq, "T": args.roar_threads}
            res_roar = ours_utils.run_roargraph_logic(
                raw_db, sliced_train_q, train_gt[:num_q], test_q, test_gt,
                args.roar_dir, os.path.join(out_dir, f"roar_tmp_r{ratio_pct}"), out_dir,
                roar_params, ef_list, args.target_k,
            )
            if res_roar:
                summary_rows.append({"Ratio(%)": ratio_pct, "Method": "RoarGraph",
                                      "QPS@90": qps_at_recall(res_roar, 90.0),
                                      "QPS@95": qps_at_recall(res_roar, 95.0),
                                      "QPS@99": qps_at_recall(res_roar, 99.0)})

        shortcut_path = os.path.join(out_dir, common.shortcut_filename(
            dataset_key, args.budget, args.train_ef, args.top_k, suffix=f"r{ratio_pct}"))
        miner.mine_shortcuts(
            sliced_train_q, sliced_train_gt_remapped,
            budget=args.budget, train_ef=args.train_ef, top_k=args.top_k,
            output_path=shortcut_path,
        )

        ours_index = hnswlib.Index(space="ip", dim=db.shape[1])
        ours_index.load_index(base_index_path, len(db), False)
        ours_index.inject_shortcuts_binary(shortcut_path)
        res_ours = ours_utils.evaluate_hnsw_variant(
            ours_index, test_q, test_gt, f"CAMP-HNSW (ratio {ratio_pct}%)", ef_list, out_dir,
            id_mapper=new_to_old, csv_name=f"{dataset_key}_ours_res_r{ratio_pct}.csv",
            build_time=0.0, target_k=args.target_k,
        )
        summary_rows.append({"Ratio(%)": ratio_pct, "Method": "CAMP-HNSW",
                              "QPS@90": qps_at_recall(res_ours, 90.0),
                              "QPS@95": qps_at_recall(res_ours, 95.0),
                              "QPS@99": qps_at_recall(res_ours, 99.0)})
        del ours_index
        gc.collect()

    del miner
    gc.collect()

    df = pd.DataFrame(summary_rows)
    out_path = os.path.join(args.output_dir, f"{dataset_key}_query_ratio_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[saved] {out_path}")


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Reproduce Figure 9: QPS@99 vs calibration query ratio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--datasets-dir", default="./dataset_benchmark")
    p.add_argument("--datasets", nargs="+", default=list(common.PAPER_DATASETS.keys()),
                    choices=list(common.PAPER_DATASETS.keys()))
    p.add_argument("--output-dir", default="./results/figure9")
    p.add_argument("--target-k", type=int, default=10)
    p.add_argument("--ratios", nargs="+", type=float, default=[0.10, 0.25, 0.50, 0.75, 1.00],
                    help="Fraction of calibration queries to use for mining, per data point.")

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
