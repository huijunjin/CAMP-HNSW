#!/usr/bin/env python3
"""Prepares the two in-distribution (ID) datasets used by Figure 8 / Sec. 4.7
(GloVe, ImageNet-ID) from their raw base/test `.hdf5` files.

Unlike the OOD datasets, GloVe and ImageNet-ID have no separately-collected
calibration ("learn") query stream -- Sec. 4.7 of the paper instead reserves
part of the base set itself as calibration queries:

  1. Load the raw file's `train` (base) and `test` (query) vectors, L2-normalize.
  2. Randomly hold out `--num-learn` (default 200,000) vectors from `train`
     to serve as calibration queries -- removed from the base set, not just
     copied, so the base a reviewer searches against never contains its own
     calibration queries.
  3. The remaining vectors become the new, smaller base set.
  4. Ground truth for both the calibration queries and the test queries is
     recomputed from scratch by exact brute-force search against this new,
     reduced base (their neighbors changed once the calibration queries were
     removed from it) -- faiss if available, otherwise a pure-numpy
     brute-force fallback (`common.compute_exact_gt`).
  5. Saved in this repository's schema (`train`/`test`/`neighbors`/`learn`/
     `learn_neighbors`) under the clean filenames `common.ID_DATASETS`
     expects (`glove.hdf5`, `imagenet_id.hdf5`).

The raw source files (`--glove-raw-file`, `--imagenet-id-raw-file`) are NOT
included in this repository -- obtain them from the VIBE benchmark toolkit
(see README.md §4.2) and place them under `--datasets-dir` before running
this script.

Example:
    python scripts/prepare_id_datasets.py --datasets-dir ./dataset_benchmark
"""
import argparse
import os

import common
import h5py
import numpy as np
from sklearn.preprocessing import normalize

# Friendly ID_DATASETS key -> raw source file basename (as shipped by VIBE).
RAW_SOURCE_FILES = {
    "GloVe": "glove-200-cosine",
    "ImageNet-ID": "imagenet-clip-512-normalized",
}


def prepare_one(key, raw_basename, args):
    raw_path = os.path.join(args.datasets_dir, f"{raw_basename}.hdf5")
    out_path = os.path.join(args.datasets_dir, f"{common.ID_DATASETS[key]}.hdf5")

    print(f"\n{'=' * 70}\n{key}: {raw_basename}.hdf5 -> {common.ID_DATASETS[key]}.hdf5\n{'=' * 70}")

    if not os.path.exists(raw_path):
        print(f"  [skip] raw file not found: {raw_path}")
        print(f"         Obtain it via the VIBE benchmark toolkit (README.md Sec. 4.2) "
              f"and place it at that path first.")
        return
    if os.path.exists(out_path) and not args.force:
        print(f"  [skip] {out_path} already exists (use --force to overwrite)")
        return

    print(f"  Loading raw base/test vectors from {raw_path}...")
    with h5py.File(raw_path, "r") as f:
        train_data = normalize(np.array(f["train"]).astype(np.float32))
        test_data = normalize(np.array(f["test"]).astype(np.float32))

    num_learn = min(args.num_learn, len(train_data) - 1)
    if num_learn != args.num_learn:
        print(f"  Warning: --num-learn ({args.num_learn}) exceeds the base set size; "
              f"using {num_learn} instead.")

    print(f"  Holding out {num_learn:,} of {len(train_data):,} base vectors as calibration queries...")
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(train_data))
    learn_idx, base_idx = indices[:num_learn], indices[num_learn:]
    learn_data = train_data[learn_idx]
    base_data = train_data[base_idx]

    print(f"  Computing exact top-{args.gt_k} ground truth against the reduced base "
          f"({len(base_data):,} vectors)...")
    test_gt = common.compute_exact_gt(test_data, base_data, k=args.gt_k)
    learn_gt = common.compute_exact_gt(learn_data, base_data, k=args.gt_k)

    print(f"  Saving to {out_path}...")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("train", data=base_data.astype(np.float32))
        f.create_dataset("test", data=test_data.astype(np.float32))
        f.create_dataset("neighbors", data=test_gt.astype(np.int64))
        f.create_dataset("learn", data=learn_data.astype(np.float32))
        f.create_dataset("learn_neighbors", data=learn_gt.astype(np.uint32))

    print(f"  Done: train={base_data.shape}, test={test_data.shape}, learn={learn_data.shape}")


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Prepare the GloVe / ImageNet-ID in-distribution datasets for Figure 8.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--datasets-dir", default="./dataset_benchmark",
                    help="Directory containing the raw source file(s) and where the "
                         "prepared .hdf5 files are written.")
    p.add_argument("--datasets", nargs="+", default=list(RAW_SOURCE_FILES.keys()),
                    choices=list(RAW_SOURCE_FILES.keys()))
    p.add_argument("--glove-raw-file", default=RAW_SOURCE_FILES["GloVe"],
                    help="Basename (no .hdf5) of the raw GloVe base/test file.")
    p.add_argument("--imagenet-id-raw-file", default=RAW_SOURCE_FILES["ImageNet-ID"],
                    help="Basename (no .hdf5) of the raw ImageNet-ID base/test file.")
    p.add_argument("--num-learn", type=int, default=200_000,
                    help="Number of base vectors to hold out as calibration queries (Sec. 4.1).")
    p.add_argument("--gt-k", type=int, default=100, help="Ground-truth neighbors to precompute per query.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true", help="Overwrite existing prepared files.")
    return p


def main():
    args = build_arg_parser().parse_args()
    raw_files = {"GloVe": args.glove_raw_file, "ImageNet-ID": args.imagenet_id_raw_file}
    for key in args.datasets:
        prepare_one(key, raw_files[key], args)


if __name__ == "__main__":
    main()
