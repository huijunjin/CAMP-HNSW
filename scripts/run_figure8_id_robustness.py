#!/usr/bin/env python3
"""Reproduce Figure 8: CAMP-HNSW's "zero routing penalty" on in-distribution
(ID) datasets -- GloVe and the ID-variant of ImageNet (Sec. 4.7).

This reuses the exact same build/inject/evaluate pipeline as Figure 4
(`run_figure4_main_comparison.run_dataset`), just pointed at the ID dataset
map instead of the OOD paper datasets: Joint Heuristic Pruning is expected
to recognize the topological simplicity of an ID manifold and suppress
redundant shortcut generation, so CAMP-HNSW's QPS should track the base
HNSW curve almost exactly rather than paying an "unnecessary densification"
penalty.

IMPORTANT for interpreting the output: "zero penalty" is a claim about QPS
at *matched recall*, not at matched ef_search. CAMP-HNSW's graph has a
different (slightly higher) average degree than base HNSW, so at the same
raw ef value the two are doing different amounts of work; comparing them
ef-for-ef can show an apparent regression at low ef that has nothing to do
with routing quality and disappears (or reverses) at the practical
recall targets (95-99%) the paper actually reports. This script computes
the recall-matched comparison automatically (`recall_matched_summary.csv`)
so nobody has to redo this by hand.

Example:
    python scripts/run_figure8_id_robustness.py \\
        --datasets-dir ./dataset_benchmark \\
        --output-dir ./results/figure8
"""
import argparse
import os

import common
import pandas as pd
from run_figure4_main_comparison import run_dataset


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Reproduce Figure 8: zero routing penalty on ID datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--datasets-dir", default="./dataset_benchmark")
    p.add_argument("--datasets", nargs="+", default=list(common.ID_DATASETS.keys()),
                    choices=list(common.ID_DATASETS.keys()))
    p.add_argument("--output-dir", default="./results/figure8")
    p.add_argument("--k-values", nargs="+", type=int, default=[10],
                    help="Figure 8 reports Recall@10 vs QPS.")

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


def report_recall_matched_summary(key, args, targets=(90.0, 95.0, 97.0, 99.0)):
    """Reads the recall{k}_{key}_final_comparison.csv that run_dataset just
    wrote and prints/saves the recall-matched QPS comparison -- the number
    that actually supports (or refutes) "zero routing penalty," as opposed
    to a same-ef comparison which compares the two graphs at different
    operating points and can look like a regression that isn't real."""
    rows = []
    for k in args.k_values:
        csv_path = os.path.join(args.output_dir, key, f"recall{k}_{key}_final_comparison.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if "Recall_HNSW" not in df.columns or "Recall_CAMP-HNSW" not in df.columns:
            continue  # HNSW or CAMP-HNSW curve missing (e.g. dataset file wasn't found)

        print(f"\n  [Figure 8] Recall-matched QPS, {key} (k={k}):")
        for target in targets:
            hnsw_qps = common.qps_at_recall(df, "Recall_HNSW", "QPS_HNSW", target)
            ours_qps = common.qps_at_recall(df, "Recall_CAMP-HNSW", "QPS_CAMP-HNSW", target)
            if hnsw_qps is None or ours_qps is None:
                print(f"    @Recall={target}%: not reached by one of the curves")
                continue
            gain_pct = (ours_qps - hnsw_qps) / hnsw_qps * 100
            print(f"    @Recall={target}%: HNSW={hnsw_qps:.0f} QPS, CAMP-HNSW={ours_qps:.0f} QPS "
                  f"({gain_pct:+.1f}%)")
            rows.append({"Dataset": key, "k": k, "Recall_target_pct": target,
                         "QPS_HNSW": round(hnsw_qps, 1), "QPS_CAMP-HNSW": round(ours_qps, 1),
                         "Gain_pct": round(gain_pct, 1)})
    return rows


def main():
    args = build_arg_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    summary_rows = []
    for key in args.datasets:
        run_dataset(key, common.ID_DATASETS[key], args)
        summary_rows.extend(report_recall_matched_summary(key, args))

    if summary_rows:
        out_path = os.path.join(args.output_dir, "recall_matched_summary.csv")
        pd.DataFrame(summary_rows).to_csv(out_path, index=False)
        print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
