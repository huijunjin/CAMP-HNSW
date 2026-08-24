#!/usr/bin/env python3
"""Generates tiny synthetic .hdf5 datasets matching the VIBE benchmark
schema used by every reproduction script in this directory (Table 1):
'train' (base vectors), 'test' (test queries), 'neighbors' (test ground
truth), 'learn' (calibration queries), 'learn_neighbors' (calibration
ground truth).

This exists purely so a reviewer can sanity-check that the full pipeline
(build the C++ engine, then Table 5 / Figure 4 / 6 / 7 / 8 / 9 / Table 4)
runs end to end without errors or a multi-gigabyte download. The random
vectors here have no real OOD structure, so the resulting numbers are
meaningless -- swap in the real datasets from Table 1 for actual results.

Usage:
    python scripts/generate_dummy_data.py
    python scripts/generate_dummy_data.py --output-dir ./dataset_benchmark --n 1000
    bash scripts/run_all.sh   # right after, with zero extra flags
"""
import argparse
import os

import common
import h5py
import numpy as np


def make_dataset(path, n, dim, n_test, n_learn, gt_k, seed):
    rng = np.random.RandomState(seed)

    def unit_vectors(count):
        v = rng.rand(count, dim).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    db = unit_vectors(n)
    test_q = unit_vectors(n_test)
    learn_q = unit_vectors(n_learn)

    def brute_force_gt(queries, k):
        scores = queries @ db.T
        return np.argsort(-scores, axis=1)[:, :k].astype(np.int64)

    test_gt = brute_force_gt(test_q, gt_k)
    learn_gt = brute_force_gt(learn_q, gt_k)

    with h5py.File(path, "w") as f:
        f.create_dataset("train", data=db)
        f.create_dataset("test", data=test_q)
        f.create_dataset("neighbors", data=test_gt)
        f.create_dataset("learn", data=learn_q)
        f.create_dataset("learn_neighbors", data=learn_gt.astype(np.uint32))

    print(f"  wrote {path}  (train={db.shape}, test={test_q.shape}, learn={learn_q.shape})")


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Generate tiny synthetic benchmark datasets for pipeline smoke-testing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", default="./dataset_benchmark",
                    help="Where to write the .hdf5 files (this is every reproduction "
                         "script's default --datasets-dir).")
    p.add_argument("--n", type=int, default=1000, help="Number of base vectors per dataset.")
    p.add_argument("--dim", type=int, default=32, help="Vector dimensionality.")
    p.add_argument("--n-test", type=int, default=200, help="Number of test queries.")
    p.add_argument("--n-learn", type=int, default=300, help="Number of calibration queries.")
    p.add_argument("--gt-k", type=int, default=100, help="Ground-truth neighbors precomputed per query.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-id-datasets", dest="include_id_datasets", action="store_false", default=True,
                    help="Skip generating the GloVe / ID-ImageNet stand-ins used by Figure 8.")
    p.add_argument("--force", action="store_true", help="Overwrite existing .hdf5 files.")
    return p


def main():
    args = build_arg_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset_map = dict(common.PAPER_DATASETS)
    if args.include_id_datasets:
        dataset_map.update(common.ID_DATASETS)

    print(f"Generating {len(dataset_map)} synthetic datasets under {args.output_dir} "
          f"(N={args.n}, dim={args.dim})...")
    for key, filename in dataset_map.items():
        path = os.path.join(args.output_dir, f"{filename}.hdf5")
        if os.path.exists(path) and not args.force:
            print(f"  [skip] {path} already exists (use --force to overwrite)")
            continue
        make_dataset(path, args.n, args.dim, args.n_test, args.n_learn, args.gt_k, args.seed)

    print("\nDone. These are random vectors with no real OOD structure -- only useful for "
          "verifying the pipeline runs end to end, not for evaluating CAMP-HNSW itself.")
    print(f"Next: bash scripts/run_all.sh   "
          f"(or DATASETS_DIR={args.output_dir} bash scripts/run_all.sh if you used a custom --output-dir)")


if __name__ == "__main__":
    main()
