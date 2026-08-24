"""Shared utilities for the CAMP-HNSW paper-reproduction scripts in this
directory: dataset name maps, benchmark loading, cache-file naming, and the
recall-matched QPS interpolation the result tables are read off.

Importing this module adds `src/python/` to `sys.path`, so `ours_model`
and `ours_utils` are importable without a separate `pip install`. The
compiled `hnswlib` and `ours_backend` extensions, however, must already be
built and installed (see `src/hnswlib/` and `src/setup.py`, or just run
`run_all.sh`, which builds them automatically).
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PYTHON_DIR = os.path.join(REPO_ROOT, "src", "python")
if SRC_PYTHON_DIR not in sys.path:
    sys.path.insert(0, SRC_PYTHON_DIR)

import h5py
import numpy as np

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


# Friendly dataset name -> .hdf5 basename (Table 1 in the paper).
PAPER_DATASETS = {
    "ImageNet": "imagenet",
    "LAION": "laion",
    "COCO": "coco",
    "MainSearch": "mainsearch",
}

# In-distribution datasets used for the zero-routing-penalty check (Sec. 4.7).
# Prepared from their raw base/test files by scripts/prepare_id_datasets.py.
ID_DATASETS = {
    "GloVe": "glove",
    "ImageNet-ID": "imagenet_id",
}


def compute_exact_gt(queries, db, k=100):
    """Brute-force exact top-k neighbors under inner product. Only used when
    a dataset's .hdf5 file has no precomputed ground truth for its
    calibration queries. Uses faiss when available (much faster on large
    datasets); falls back to a batched pure-numpy implementation otherwise,
    so this repository has no hard dependency on faiss being installed.
    """
    if _HAS_FAISS:
        index = faiss.IndexFlatIP(db.shape[1])
        index.add(db)
        _, gt = index.search(queries, k)
        return gt.astype(np.int32)

    gt = np.empty((len(queries), k), dtype=np.int32)
    batch = 256
    for start in range(0, len(queries), batch):
        end = min(start + batch, len(queries))
        scores = queries[start:end] @ db.T
        top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        # argpartition doesn't sort within the top-k; sort for determinism.
        row_scores = np.take_along_axis(scores, top, axis=1)
        order = np.argsort(-row_scores, axis=1)
        gt[start:end] = np.take_along_axis(top, order, axis=1)
    return gt


def load_dataset(hdf5_path, gt_k=100):
    """Loads a VIBE-format benchmark .hdf5 file (Table 1): 'train' (base
    vectors), 'test' (test queries), 'neighbors' (test ground truth),
    'learn' (calibration queries), and optionally 'learn_neighbors'
    (calibration ground truth; computed on the fly if absent). All vectors
    are L2-normalized so inner product doubles as cosine similarity.

    Returns: (raw_db, test_q, train_q, test_gt, train_gt)
    """
    from sklearn.preprocessing import normalize

    with h5py.File(hdf5_path, "r") as f:
        raw_db = normalize(np.array(f["train"]).astype(np.float32))
        test_q = normalize(np.array(f["test"]).astype(np.float32))
        train_q = normalize(np.array(f["learn"]).astype(np.float32))
        test_gt = np.array(f["neighbors"]).astype(np.int32)
        if "learn_neighbors" in f:
            train_gt = np.array(f["learn_neighbors"]).astype(np.int32)
        else:
            train_gt = compute_exact_gt(train_q, raw_db, k=gt_k)

    return raw_db, test_q, train_q, test_gt, train_gt


def default_ef_search_list():
    """Dense ef_search sweep used throughout the paper's evaluation: finer
    granularity near the low-ef 'knee' of the recall/QPS curve, coarser in
    the high-ef tail where returns diminish."""
    return (
        list(range(50, 100, 5))
        + list(range(100, 300, 10))
        + list(range(300, 1000, 50))
        + list(range(1000, 1501, 100))
    )


def base_index_filename(dataset_key, m, ef_construction):
    """Cache filename for a built base HNSW index.

    The construction parameters are part of the name on purpose: the scripts
    reuse an existing index file instead of rebuilding, so a name that
    ignored M / ef_construction would silently serve a stale index after the
    reviewer changes those flags.
    """
    return f"{dataset_key}_base_M{m}_efc{ef_construction}.bin"


def shortcut_filename(dataset_key, budget, train_ef, top_k, suffix=""):
    """Cache filename for a mined shortcut set, keyed by the Phase 2/3
    parameters that determine its contents (see base_index_filename for why).
    `suffix` distinguishes otherwise-identical runs, e.g. Figure 9's
    calibration-ratio sweep."""
    name = f"{dataset_key}_shortcuts_b{budget}_ef{train_ef}_k{top_k}"
    if suffix:
        name += f"_{suffix}"
    return name + ".bin"


def _finite_curve(recalls, qps_values):
    """(recall, qps) pairs as float arrays with non-finite entries dropped.
    Merged comparison tables can carry NaN rows when two methods were
    evaluated at different ef values."""
    r = np.asarray(recalls, dtype=float)
    q = np.asarray(qps_values, dtype=float)
    keep = np.isfinite(r) & np.isfinite(q)
    return r[keep], q[keep]


def qps_at_recall(recalls, qps_values, target_recall, default=None):
    """Interpolates the QPS a (recall, qps) curve achieves at `target_recall`.

    This is the methodologically correct way to compare two methods'
    throughput: at matched *recall*, not matched *ef_search*. Two graphs
    with different edge sets reach a given recall at different ef values,
    so comparing their QPS at the same raw ef number silently compares them
    at two different operating points and can produce a misleading gap in
    either direction. Returns `default` if the curve never reaches
    target_recall.
    """
    r, q = _finite_curve(recalls, qps_values)
    if r.size == 0 or target_recall > r.max():
        return default

    order = np.argsort(r)
    return float(np.interp(target_recall, r[order], q[order]))


def qps_at_recall_pareto(recalls, qps_values, target_recall, default=None):
    """Like `qps_at_recall`, but first reduces the curve to its Pareto-optimal
    frontier (the points no other point beats on both recall and QPS).

    Raw ef sweeps are measured under timing noise, so an unlucky sample can
    leave a point that is worse on both axes than another; interpolating
    through those dips understates the achievable throughput. Table 5 reports
    its QPS@95 / QPS@99 operating points off this frontier.
    """
    r, q = _finite_curve(recalls, qps_values)
    if r.size == 0 or target_recall > r.max():
        return default

    # Walk from the fastest point down; keep a point only when it improves on
    # the best recall seen so far.
    frontier_r, frontier_q = [], []
    best_recall = -np.inf
    for i in np.argsort(-q):
        if r[i] > best_recall:
            best_recall = r[i]
            frontier_r.append(r[i])
            frontier_q.append(q[i])

    order = np.argsort(frontier_r)
    return float(np.interp(target_recall,
                           np.asarray(frontier_r)[order],
                           np.asarray(frontier_q)[order]))
