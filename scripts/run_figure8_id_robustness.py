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

Example:
    python scripts/run_figure8_id_robustness.py \\
        --datasets-dir ./dataset_benchmark \\
        --output-dir ./results/figure8
"""
import argparse
import os

import common
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


def main():
    args = build_arg_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for key in args.datasets:
        run_dataset(key, common.ID_DATASETS[key], args)


if __name__ == "__main__":
    main()
