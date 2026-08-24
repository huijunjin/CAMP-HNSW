# CAMP-HNSW

**Cache-Aligned Memory reordering and Pruning for Hierarchical Navigable Small World graphs**

Official artifact for:

> Huijun Jin and Sanghyun Park. **CAMP-HNSW: Hardware-Topological Co-Design for OOD-Robust ANNS via Cache Alignment and Targeted Rewiring.** PVLDB, 19(1), 2026.

[![VLDB 2027 Artifact](https://img.shields.io/badge/VLDB-Artifact%20Evaluation-blue)]()
[![License](https://img.shields.io/badge/license-CC%20BY--NC--ND%204.0-lightgrey)]()

---

## 1. Overview

Graph-based ANNS indices such as HNSW break down under **Out-of-Distribution (OOD) queries** — the routine case in cross-modal retrieval (e.g. text-to-image search), where the query distribution deviates from the indexed base data. We show this failure is not primarily about *distance*: the true nearest neighbor is often physically close to where greedy search gets stuck, but the graph has no edge to reach it. We call this **topological blindness**. Prior fixes (RoarGraph, NGFix) patch it by *additively* injecting extra routing edges on top of an untouched base graph, which piles up redundant, hardware-unaware connectivity.

**CAMP-HNSW** is a Pareto-superior framework that closes this gap through **hardware-topological co-design**, built as a three-phase pipeline on top of the standard HNSW greedy router:

| Phase | Name | What it does |
|---|---|---|
| **1** | **Cache-Aligned Memory Reordering** | Clusters the dataset with MiniBatchKMeans and physically repacks vectors so topologically adjacent nodes share L3 cache lines — a pure hardware optimization; the graph topology and distance-computation count are unchanged. |
| **2** | **Targeted Shortcut Mining** | Runs calibration queries against the base graph, diffs the retrieved neighborhood against ground truth to find exactly where routing fails, and mines candidate bridging edges to the missing targets. |
| **3** | **Joint Heuristic Pruning** | Pools each mined shortcut together with the target node's *existing* base edges and forces them to compete under the same RNG heuristic HNSW already uses, bounded by a fixed max out-degree — evicting redundant base edges instead of just piling shortcuts on top of them. |

The result: CAMP-HNSW achieves SOTA query throughput at practical, high-recall regimes (95–99%) across four diverse datasets, using **one fixed default configuration** — no per-dataset hyperparameter tuning — while keeping memory and build cost tightly bounded.

This repository contains **only CAMP-HNSW's own engine**: the modified HNSW fork (Phases 1 and 3), the C++ mining engine (Phase 2), and the Python orchestration and evaluation pipeline. Baseline methods (RoarGraph, NGFix) are intentionally **not** re-implemented here — see [§6](#6-baselines-ngfix--roargraph) for why, and where to get the official code.

---

## 2. Environment Setup

The engine requires a C++17 compiler with OpenMP, and Python 3.9+.

```bash
# 1. Create and activate a dedicated conda environment
conda create -n camp-hnsw python=3.9 -y
conda activate camp-hnsw

# 2. Install Python dependencies
pip install -r requirements.txt
```

`requirements.txt` only lists what this repository's own code actually imports (numpy, pandas, scikit-learn, h5py, pybind11) — it deliberately does **not** include `hnswlib`. See [§3](#3-build-instructions) for why.

---

## 3. Build Instructions

CAMP-HNSW's engine is two separate native Python extensions, both built from source under `src/`:

```bash
# 1. The modified hnswlib fork (Phase 1 + Phase 3, plus the search engine itself)
pip install --no-build-isolation -e src/hnswlib

# 2. The Phase 2 mining engine
pip install --no-build-isolation -e src
```

> **Why not `pip install hnswlib`?** This repository ships a *modified* fork of hnswlib (`src/hnswlib`) with CAMP-HNSW's core algorithm built directly into `hnswalg.h` — `injectShortcutsBinary()` (Joint Heuristic Pruning), the fixed-level-0 base graph, real-centroid entry point, and edge-activity profiling hooks. Installing the vanilla `hnswlib` package from PyPI would silently shadow this fork and every result would be wrong. `requirements.txt` deliberately excludes it, and `scripts/run_all.sh` (§5) builds both extensions automatically if they aren't importable yet.

**Portability note (AVX-512):** the build's compiler flags use `-march=native` only — CAMP-HNSW does **not** hardcode `-mavx512f`/`-mavx512cd`/etc. `-march=native` already asks the compiler to auto-detect and use the best instruction set the *build machine* actually supports, AVX-512 included on hardware like the paper's Xeon w5-2465X. If a reviewer's machine lacks AVX-512, the build still succeeds and simply falls back to whatever the CPU does support (e.g. AVX2) — no illegal-instruction crashes, no manual flag-editing required.

Verify the build:

```bash
python -c "import hnswlib, ours_backend; print('CAMP-HNSW engine OK')"
```

---

## 4. Datasets

Every script in `scripts/` reads a benchmark dataset from a single `.hdf5` file, following the schema below (this is the format produced by the VIBE benchmark tooling the paper's OOD datasets are drawn from):

| HDF5 key | dtype | shape | Meaning |
|---|---|---|---|
| `train` | float32 | `(N, D)` | Base vectors indexed by the graph |
| `test` | float32 | `(N_test, D)` | Test queries (OOD; used for the reported Recall/QPS) |
| `neighbors` | int | `(N_test, ≥100)` | Exact ground-truth neighbors for `test` |
| `learn` | float32 | `(N_calib, D)` | Calibration queries used for Phase 2 shortcut mining |
| `learn_neighbors` | int | `(N_calib, ≥100)` *(optional)* | Ground truth for `learn`; if absent, it's computed automatically (`scripts/common.py` uses faiss if installed, otherwise a pure-numpy brute-force fallback — no hard faiss dependency) |

All vectors are expected L2-normalized (inner product then doubles as cosine similarity — `scripts/common.py` normalizes on load regardless).

### 4.1 Quick Functionality Test (recommended first step)

Before touching real data, generate a tiny synthetic dataset and confirm the whole pipeline runs end to end in well under a minute:

```bash
python scripts/generate_dummy_data.py     # writes 6 small synthetic .hdf5 files to ./dataset_benchmark/
bash scripts/run_all.sh                   # should complete with no errors
```

This produces random vectors with no real OOD structure — the *numbers* are meaningless, but a clean run confirms the build, every script's argument wiring, and the file I/O all work correctly on your machine before committing to hours of real-data experiments.

### 4.2 Real Datasets for Full Reproducibility

| Paper name | `.hdf5` filename (goes in `dataset_benchmark/`) | Source |
|---|---|---|
| ImageNet | `imagenet-align-640-normalized.hdf5` | Text-to-image embeddings + query split from the **VIBE benchmark** (Jääsaari et al., *VIBE: Vector Index Benchmark for Embeddings*, [arXiv:2505.17810](https://arxiv.org/abs/2505.17810)) |
| LAION | `LAION-512.hdf5` | VIBE benchmark, as above |
| COCO | `coco-nomic-768-normalized.hdf5` | VIBE benchmark, as above |
| MainSearch | `MainSearch-11M.hdf5` | Large-scale industrial dataset introduced in NGFix (Hua et al., *Dynamically Fix Hardness for Efficient Approximate Nearest Neighbor Search*, PACMMOD 3(6), 2025), **publicly hosted on Zenodo: [zenodo.org/records/17257137](https://zenodo.org/records/17257137)**. See the setup instructions below. |
| GloVe (ID) | `glove-200-cosine-maphnsw.hdf5` | Standard word embeddings (Pennington et al., *GloVe: Global Vectors for Word Representation*, EMNLP 2014) — commonly distributed via [nlp.stanford.edu/projects/glove](https://nlp.stanford.edu/projects/glove/) or ANN-Benchmarks-style mirrors. |
| ImageNet-ID | `imagenet-clip-512-normalized-maphnsw.hdf5` | Same source as ImageNet above, used as an in-distribution (query = base manifold) control instead of the OOD split. |

**Setting up MainSearch:** the dataset is distributed on Zenodo as a set of split RAR archives (`mainsearch.part01.rar`, `mainsearch.part02.rar`, ... — a single archive too large for one part). To set it up:

1. Download **every** `mainsearch.partNN.rar` file from **[zenodo.org/records/17257137](https://zenodo.org/records/17257137)** into one local directory.
2. Extract with any multi-part-aware unrar tool, pointed at `mainsearch.part01.rar` (it will pull in the remaining parts automatically):
   ```bash
   unrar x mainsearch.part01.rar
   ```
3. Convert the extracted base/query/ground-truth arrays into this repo's VIBE HDF5 schema (§4, table above) using the conversion snippet below, and save the result as `dataset_benchmark/MainSearch-11M.hdf5`.

Once you have the raw base/query/ground-truth arrays for a dataset, convert them into this repo's schema (§4, table above) with a short script, e.g.:

```python
import h5py, numpy as np

with h5py.File("my_dataset.hdf5", "w") as f:
    f.create_dataset("train", data=base_vectors.astype(np.float32))
    f.create_dataset("test", data=test_queries.astype(np.float32))
    f.create_dataset("neighbors", data=test_ground_truth.astype(np.int64))
    f.create_dataset("learn", data=calibration_queries.astype(np.float32))
    f.create_dataset("learn_neighbors", data=calibration_ground_truth.astype(np.uint32))  # optional
```

For the two **in-distribution (ID)** datasets (GloVe, ImageNet-ID), `learn`/`learn_neighbors` are simply a held-out partition of `train` itself (Sec. 4.1 of the paper: 200,000 vectors reserved as calibration queries, with GT recomputed by exact brute-force search over the remaining base set) rather than a separate OOD query stream.

Place every `.hdf5` file under `./dataset_benchmark/` (or point `--datasets-dir` / `DATASETS_DIR` at wherever you keep them — see §5).

---

## 5. Reproducing the Main Results

### The one-liner

```bash
bash scripts/run_all.sh
```

This single command builds the C++ engine if it isn't built yet, then runs **every** reproduction script below in sequence — Table 5, Figure 4, Figure 6, Figure 7, Figure 8, Figure 9, then Table 4 — against whatever datasets it finds under `./dataset_benchmark/`. Missing individual dataset files are skipped per-dataset with a warning rather than aborting the run, and Table 4 (which needs the Linux `perf` tool) degrades gracefully with a clear message if `perf` is unavailable or unauthorized — every other result still completes.

```bash
# Point at a real dataset directory and a custom output location
DATASETS_DIR=/path/to/datasets OUTPUT_DIR=/path/to/results bash scripts/run_all.sh

# Restrict to specific datasets, or override any hyperparameter — forwarded
# to every OOD-dataset script automatically
bash scripts/run_all.sh --datasets LAION COCO --budget 500
```

Each script is also fully usable standalone with `--help` for the complete list of overridable parameters (paths, M, ef_construction, Phase 1/2/3 hyperparameters, etc.).

### Script → paper result mapping

| Script | Paper result | Output |
|---|---|---|
| `scripts/run_table5_ablation.py` | **Table 5** — incremental ablation (Base HNSW → Naive Injection → Candidate Pruning → Joint Pruning → + Memory Reorder → Phase 1 Only), Sec. 4.5 | `results/table5/final_ablation_table.csv` + per-variant/per-run CSVs |
| `scripts/run_figure4_main_comparison.py` | **Figure 4** — Recall@{1,10,100} vs QPS across ImageNet/LAION/COCO/MainSearch, Sec. 4.2 | `results/figure4/<dataset>/recall{1,10,100}_<dataset>_final_comparison.csv` |
| `scripts/run_figure6_kmeans_tuning.py` | **Figure 6** — Phase 1 sensitivity to MiniBatchKMeans sample ratio and cluster count K, Sec. 4.6 | `results/figure6/<dataset>_cache_reorder_grid_search_result.csv` |
| `scripts/run_figure7_pareto_search.py` | **Figure 7** — global Pareto frontier of build time vs. recall over Phase 2's `ef_mine`/`N_mine`, Sec. 4.6 | `results/figure7/figure7_global_pareto_frontier.csv` + per-dataset grids |
| `scripts/run_figure8_id_robustness.py` | **Figure 8** — "zero routing penalty" on in-distribution datasets (GloVe, ImageNet-ID), Sec. 4.7 | `results/figure8/<dataset>/recall10_<dataset>_final_comparison.csv` |
| `scripts/run_figure9_query_ratio.py` | **Figure 9** — data efficiency / cold-start: QPS@99% vs. calibration query ratio (10–100%), Sec. 4.7 | `results/figure9/<dataset>_query_ratio_summary.csv` |
| `scripts/run_table4_hardware_profile.py` | **Table 4** — L3 cache / CPU-time impact of Phase 1 reordering (`perf stat`), Sec. 4.4 | `results/table4/table4_hardware_profile.csv` |
| `scripts/generate_dummy_data.py` | *(not a paper result)* | Synthetic datasets for the quick functionality test in §4.1 |
| `scripts/common.py` | *(shared library)* | Dataset loading, ground-truth computation, and the default `ef_search` sweep used by every script above |

---

## 6. Baselines (NGFix, RoarGraph)

This repository focuses on validating **CAMP-HNSW's own core logic** — the cache-aligned reordering, shortcut mining, and joint heuristic pruning engine described above. For a fair, unmodified comparison, the baseline methods used in Figures 4/8/9 and Table 4 are **not re-implemented here**; please refer to their official repositories:

- **RoarGraph** — [github.com/matchyc/RoarGraph](https://github.com/matchyc/RoarGraph) (Chen et al., VLDB 2024)
- **NGFix** — [github.com/yuhuifishash/NGFix](https://github.com/yuhuifishash/NGFix) (Hua et al., PACMMOD 2025)

`scripts/run_figure4_main_comparison.py` and `scripts/run_figure9_query_ratio.py` accept an optional `--roar-dir` pointing at a built RoarGraph checkout and will include it in the comparison automatically; if omitted, those scripts still run end-to-end and simply report the HNSW vs. CAMP-HNSW comparison. `src/hnswlib/hnswlib/hnswalg.h` additionally includes an `importEdgesFromNGFix()` helper for translating an NGFix-produced graph into this repository's cache-reordered ID space, for anyone wiring NGFix into the same evaluation harness.

Our modified HNSW fork itself is based on the upstream [nmslib/hnswlib](https://github.com/nmslib/hnswlib).

---

## Repository Layout

```
CAMP-HNSW-Release/
├── src/
│   ├── hnswlib/          # Modified HNSW fork: Phase 1 (real-centroid EP) + Phase 3
│   │                      # (Joint Heuristic Pruning, in hnswlib/hnswalg.h)
│   ├── ours_backend.cpp   # Phase 2: C++ shortcut-mining engine (pybind11 extension)
│   ├── setup.py           # Build config for ours_backend
│   └── python/
│       ├── ours_model.py  # Pipeline orchestration (Ours_Miner)
│       └── ours_utils.py  # Phase 1 reordering + evaluation utilities
├── scripts/               # Paper reproduction scripts (see §5) + run_all.sh
├── requirements.txt
└── README.md
```
