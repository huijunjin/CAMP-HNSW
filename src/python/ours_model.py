import os
import time
import multiprocessing

import numpy as np

import hnswlib
import ours_backend


class Ours_Miner:
    """Orchestrates the CAMP-HNSW pipeline: base index construction with a
    real-centroid entry point, Phase 2 shortcut mining, and Phase 3 injection
    (delegated to the C++ engines in `ours_backend` / the modified hnswlib)."""

    def __init__(self, metric='ip', m=16, ef_construction=200, dim=None):
        self.space = 'ip' if metric in ['ip', 'dot', 'inner_product', 'cosine'] else 'l2'
        self.m = m
        self.ef_construction = ef_construction
        self.index = None
        self.dim = dim
        self.data = None

    def fit(self, data, save_path=None):
        """Build the base HNSW index with the entry point fixed to the real
        dataset node closest to the geometric centroid (Sec. 3.2), rather than
        an arbitrary first-inserted node. This removes the physical-memory
        entry-point bias that would otherwise skew Phase 1 reordering."""
        print(f"   [Miner] Building base index (M={self.m}, real-centroid EP)...")
        self.dim = data.shape[1]
        self.data = data

        centroid = np.mean(data, axis=0, keepdims=True)
        centroid = centroid / np.linalg.norm(centroid, axis=1, keepdims=True)

        sims = np.dot(data, centroid.T).flatten()
        closest_idx = int(np.argmax(sims))
        print(f"   -> Selected node {closest_idx} (closest to centroid) as entry point.")

        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.init_index(max_elements=len(data), ef_construction=self.ef_construction, M=self.m)

        # hnswlib designates the first inserted point as the entry point, so
        # the centroid-closest node must be added before the rest.
        self.index.add_items(data[closest_idx:closest_idx + 1], np.array([closest_idx]))

        mask = np.ones(len(data), dtype=bool)
        mask[closest_idx] = False
        self.index.add_items(data[mask], np.arange(len(data))[mask])

        if save_path:
            self.index.save_index(save_path)
        return self

    def load_index(self, path, data):
        self.dim = data.shape[1]
        self.data = data
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.load_index(path, len(data), False)

    def inject_shortcuts(self, shortcut_path):
        if self.index is None:
            raise RuntimeError("Index is not loaded.")
        if not os.path.exists(shortcut_path):
            raise FileNotFoundError(f"Shortcut file not found: {shortcut_path}")

        print(f"[Injector] Injecting shortcuts from {shortcut_path}...")
        self.index.inject_shortcuts_binary(shortcut_path)
        print("[Injector] Injection complete.")

    def mine_shortcuts(self, train_q, train_gt, budget, train_ef, top_k, output_path):
        """Phase 2 (Algorithm 2): search the calibration queries against the
        base index, then hand the retrieved neighborhoods + ground truth to
        the fused C++ engine for mining, deduplication, and RNG-heuristic
        candidate pre-filtering. Returns the pure algorithmic time (build
        time reported in the paper excludes file I/O)."""
        if self.index is None or self.data is None:
            raise RuntimeError("Index/data not ready.")

        n_cores = multiprocessing.cpu_count()
        print(f"\n[Miner] Mining shortcuts (budget={budget}, cores={n_cores}, k={top_k})...")

        # Phase 2a: base HNSW search (stays on the Python/hnswlib side).
        self.index.set_ef(train_ef)
        t0 = time.time()
        labels, _ = self.index.knn_query(train_q, k=top_k, num_threads=n_cores)
        p1_time = time.time() - t0
        print(f"   [Phase 2a] Base HNSW search: {p1_time:.2f}s")

        # Phase 2b-d: mining + deduplication + filtering, fused in C++.
        # The C++ side expects one rectangular int32 array, with -1 padding
        # for queries that have fewer ground-truth neighbors than the widest
        # row. Benchmark ground truth is already rectangular, so that case
        # is just a dtype cast.
        gt_padded = self._as_padded_gt(train_gt)

        print("   [Phase 2b-d] Running unified C++ engine (mine -> dedup -> filter)...")
        p2_pure_time = ours_backend.mine_and_filter(
            labels.astype(np.int32),
            gt_padded,
            self.data,
            top_k,
            budget,
            n_cores,
            output_path,
        )
        print(f"   [Phase 2b-d] C++ engine pure time: {p2_pure_time:.2f}s (file I/O excluded)")

        pure_algo_time = p1_time + p2_pure_time
        print(f"[Miner] Total pure mining time: {pure_algo_time:.2f}s")

        return pure_algo_time

    @staticmethod
    def _as_padded_gt(train_gt):
        """Ground truth as a rectangular int32 array, -1 padded (see caller)."""
        gt = np.asarray(train_gt)
        if gt.ndim == 2:
            return gt.astype(np.int32, copy=False)

        max_len = max(len(row) for row in train_gt)
        padded = np.full((len(train_gt), max_len), -1, dtype=np.int32)
        for i, row in enumerate(train_gt):
            padded[i, :len(row)] = row
        return padded
