import os
import re
import struct
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


def write_fbin(filename, data):
    with open(filename, "wb") as f:
        n, d = data.shape
        f.write(struct.pack("I", n))
        f.write(struct.pack("I", d))
        f.write(data.astype(np.float32).tobytes())


def write_ibin_with_dists(filename, ids):
    n, d = ids.shape
    with open(filename, "wb") as f:
        f.write(struct.pack("I", n))
        f.write(struct.pack("I", d))
        f.write(ids.astype(np.int32).tobytes())
        f.write(np.zeros((n, d), dtype=np.float32).tobytes())


def reorder_dataset_by_clustering(db, num_clusters=256, sample_ratio=0.05, n_init=1, max_iter=30, batch_size=2048):
    """Phase 1 (Algorithm 1): Cache-Aligned Memory Reordering.

    Clusters a subsample of the dataset with MiniBatchKMeans, assigns every
    vector to its nearest centroid, and physically sorts the array by cluster
    label so that topologically adjacent nodes end up in adjacent memory
    (and therefore adjacent cache lines).
    """
    num_elements = len(db)
    if num_clusters is None:
        num_clusters = int(np.sqrt(num_elements))

    # Production-scale datasets always have sample_size >> num_clusters (the
    # default K=512 assumes millions of rows, see Sec. 3.3/4.6); on a much
    # smaller dataset (e.g. a smoke-test run) MiniBatchKMeans would otherwise
    # raise since it requires n_samples >= n_clusters. This clamp never
    # triggers at the scales the paper evaluates.
    sample_size = int(num_elements * sample_ratio)
    if num_clusters > sample_size:
        clamped = max(1, sample_size)
        print(f"[Reordering] Warning: num_clusters ({num_clusters}) exceeds the training "
              f"sample size ({sample_size}); clamping to {clamped} for this run.")
        num_clusters = clamped

    print(f"[Reordering] Clustering dataset (K={num_clusters}, sample={int(sample_ratio*100)}%, "
          f"max_iter={max_iter}, n_init={n_init}, batch={batch_size})...")
    t0 = time.time()

    # Train centroids on a small subsample only (Sec. 3.3: this is a
    # hardware-alignment heuristic, not a clustering-quality objective).
    np.random.seed(42)
    sampled_indices = np.random.choice(num_elements, sample_size, replace=False)
    sampled_db = db[sampled_indices]

    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        batch_size=batch_size,
        n_init=n_init,
        max_iter=max_iter,
        random_state=42,
    )
    kmeans.fit(sampled_db)
    cluster_labels = kmeans.predict(db)

    # Physically pack the dataset by cluster label (Algorithm 1, lines 6-12).
    sorted_indices = np.argsort(cluster_labels)
    sorted_db = db[sorted_indices]
    sorted_labels = cluster_labels[sorted_indices]

    # Old<->new ID mapping tables, needed to translate results back to the
    # original dataset IDs after search (Sec. 3.3).
    old_to_new = np.zeros(num_elements, dtype=np.int32)
    old_to_new[sorted_indices] = np.arange(num_elements)
    new_to_old = sorted_indices.astype(np.uint32)

    print(f"[Reordering] Done in {time.time() - t0:.2f}s.")
    return sorted_db, old_to_new, new_to_old, sorted_labels


# =========================================================
# Baseline wrappers and evaluation helpers
# =========================================================

def run_roargraph_logic(db, train_q, train_gt, test_q, test_gt,
                        roar_dir, temp_dir, temp_data_dir, params, ef_list, target_k=10):
    """Builds and searches a RoarGraph index by shelling out to the official
    RoarGraph binaries (test_build_roargraph / test_search_roargraph) under
    `roar_dir`. Requires the RoarGraph repository to be built separately;
    not part of the CAMP-HNSW core engine."""
    print(f"\n{'='*60}")
    print(f"Running baseline: RoarGraph (official binary) - k={target_k}")
    print(f"{'='*60}")

    bin_build = os.path.join(roar_dir, "build/tests/test_build_roargraph")
    bin_search = os.path.join(roar_dir, "build/tests/test_search_roargraph")

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(temp_data_dir, exist_ok=True)

    p = {
        "base":  os.path.join(temp_dir, "roar_base.bin"),
        "query": os.path.join(temp_dir, "roar_query_train.bin"),
        "gt":    os.path.join(temp_dir, "roar_gt_train.bin"),
        "q_test": os.path.join(temp_dir, "roar_q_test.bin"),
        "gt_test": os.path.join(temp_dir, "roar_gt_test.bin"),
        "index": os.path.join(temp_dir, "roar.index"),
        "res":   os.path.join(temp_dir, "roar_res.txt"),
        "csv":   os.path.join(temp_data_dir, "roar_res.csv"),
    }

    if not os.path.exists(p["base"]):
        print("   [RoarGraph] Writing binaries...")
        write_fbin(p["base"], db)
        write_fbin(p["query"], train_q)
        write_ibin_with_dists(p["gt"], train_gt)
        write_fbin(p["q_test"], test_q)
        write_ibin_with_dists(p["gt_test"], test_gt)

    build_time = 0.0
    if not os.path.exists(p["index"]):
        print("   [RoarGraph] Building index...")
        process = subprocess.Popen([bin_build, "--data_type", "float", "--dist", "ip",
                        "--base_data_path", p["base"], "--sampled_query_data_path", p["query"],
                        "--learn_base_nn_path", p["gt"], "--projection_index_save_path", p["index"],
                        "--M_sq", str(params['M_sq']), "--M_pjbp", str(params['M_pjbp']),
                        "--L_pjpq", str(params['L_pjpq']), "-T", str(params['T'])],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        full_log = []
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                if "% of projection" not in line and "% of" not in line:
                    print(line.strip())
                full_log.append(line)

        full_log_str = "".join(full_log)
        match = re.search(r"Build projection graph time:\s*([\d\.]+)", full_log_str)
        if match:
            build_time = float(match.group(1))
            print(f"   [RoarGraph] Parsed build time: {build_time:.2f}s")
    else:
        print("   [RoarGraph] Index found, using cached copy.")

    if os.path.exists(p["res"]) and os.path.getsize(p["res"]) > 0:
        print("   [RoarGraph] Found cached results, skipping search.")
    else:
        print(f"   [RoarGraph] Searching (target k={target_k})...")

        # RoarGraph errors out if L_pq < k, so drop ef values below target_k.
        valid_ef_list = [v for v in ef_list if v >= target_k]
        if not valid_ef_list:
            print("   [RoarGraph] Warning: no valid ef values >= k, skipping search.")
            return None

        l_pq_str = " ".join(str(v) for v in valid_ef_list)
        subprocess.run([bin_search, "--data_type", "float", "--dist", "ip",
                        "--base_data_path", p["base"], "--projection_index_save_path", p["index"],
                        "--gt_path", p["gt_test"], "--query_path", p["q_test"],
                        "--L_pq", *l_pq_str.split(), "--k", str(target_k),
                        "-T", "1", "--evaluation_save_path", p["res"]],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    parsed_data = []
    if os.path.exists(p["res"]):
        with open(p["res"], "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',') if ',' in line else line.split()
                if len(parts) >= 6:
                    try:
                        parsed_data.append({
                            "EF": int(parts[0]),
                            "Recall": float(parts[4]) * 100,
                            "QPS": float(parts[1]),
                            "Avg_Visited": float(parts[2]),
                            "Avg_Hops": float(parts[5]),
                            "Time": build_time,
                        })
                    except ValueError:
                        continue

    if parsed_data:
        df_roar = pd.DataFrame(parsed_data)
        df_roar.to_csv(p["csv"], index=False)
        return {"recall": df_roar["Recall"].tolist(), "qps": df_roar["QPS"].tolist()}
    return None


def evaluate_hnsw_variant(index, test_q, test_gt, label, ef_list, temp_data_dir,
                          id_mapper=None, csv_name="result.csv", build_time=0.0, target_k=10):
    print(f"\nEvaluating {label} (k={target_k})...")
    if build_time > 0:
        print(f"Recorded build time: {build_time:.2f}s")

    print(f"{'EF':<5} | {'Recall':<8} | {'QPS':<8} | {'Avg Visited':<12} | {'Avg Hops':<8}")
    print("-" * 65)

    results_list = []
    for ef in ef_list:
        if ef < target_k:
            continue

        index.set_ef(ef)
        index.reset_performance_metrics()
        start = time.time()
        labels, _ = index.knn_query(test_q, k=target_k, num_threads=1)
        end = time.time()

        stats = index.get_performance_metrics()
        avg_visited = stats['visited'] / len(test_q)
        avg_hops = stats['hops'] / len(test_q)

        if id_mapper is not None:
            labels = id_mapper[labels]
        correct = sum(
            len(set(labels[i]).intersection(test_gt[i][:target_k]))
            for i in range(len(test_q))
        )
        recall = (correct / (len(test_q) * target_k)) * 100.0
        qps = len(test_q) / (end - start) if (end - start) > 0 else 0

        results_list.append({
            "EF": ef, "Recall": recall, "QPS": qps,
            "Avg_Visited": avg_visited, "Avg_Hops": avg_hops,
            "Time": build_time,
        })

        if ef % 20 == 0 or ef == 10:
            print(f"{ef:<5} | {recall:.2f}%   | {qps:<8.0f} | {avg_visited:<12.1f} | {avg_hops:<8.1f}")

    os.makedirs(temp_data_dir, exist_ok=True)
    csv_path = os.path.join(temp_data_dir, csv_name)
    df = pd.DataFrame(results_list)
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    return {"recall": df["Recall"].tolist(), "qps": df["QPS"].tolist()}


def compare_results(temp_data_dir):
    files = {
        "HNSW": os.path.join(temp_data_dir, "hnsw_res.csv"),
        "Ours": os.path.join(temp_data_dir, "ours_res.csv"),
        "Roar": os.path.join(temp_data_dir, "roar_res.csv"),
    }
    dfs = []
    for name, path in files.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = df.rename(columns={
                "Recall": f"Recall_{name}", "QPS": f"QPS_{name}",
                "Avg_Visited": f"Vis_{name}", "Avg_Hops": f"Hop_{name}",
                "Time": f"Time_{name}",
            })
            df["EF"] = df["EF"].astype(int)
            dfs.append(df)

    if not dfs:
        return

    merged_df = dfs[0]
    for i in range(1, len(dfs)):
        merged_df = pd.merge(merged_df, dfs[i], on="EF", how="outer")
    merged_df = merged_df.sort_values("EF").reset_index(drop=True)

    if "Time_Ours" in merged_df.columns and "Time_Roar" in merged_df.columns:
        merged_df["Gap_Time"] = merged_df["Time_Ours"] - merged_df["Time_Roar"]
    if "Recall_Ours" in merged_df.columns and "Recall_Roar" in merged_df.columns:
        merged_df["Gap_Recall"] = merged_df["Recall_Ours"] - merged_df["Recall_Roar"]
    if "QPS_Ours" in merged_df.columns and "QPS_Roar" in merged_df.columns:
        merged_df["Gap_QPS%"] = (merged_df["QPS_Ours"] - merged_df["QPS_Roar"]) / merged_df["QPS_Roar"] * 100
    if "Vis_Ours" in merged_df.columns and "Vis_Roar" in merged_df.columns:
        merged_df["Gap_Vis%"] = (merged_df["Vis_Ours"] - merged_df["Vis_Roar"]) / merged_df["Vis_Roar"] * 100
    if "Hop_Ours" in merged_df.columns and "Hop_Roar" in merged_df.columns:
        merged_df["Gap_Hop"] = merged_df["Hop_Ours"] - merged_df["Hop_Roar"]

    final_cols = ["EF"]
    for col in ["Time_HNSW", "Time_Roar", "Time_Ours", "Gap_Time",
                "Recall_HNSW", "Recall_Roar", "Recall_Ours", "Gap_Recall",
                "QPS_HNSW", "QPS_Roar", "QPS_Ours", "Gap_QPS%",
                "Vis_HNSW", "Vis_Roar", "Vis_Ours", "Gap_Vis%",
                "Hop_HNSW", "Hop_Roar", "Hop_Ours", "Gap_Hop"]:
        if col in merged_df.columns:
            final_cols.append(col)

    final_df = merged_df[final_cols].round(2)
    merged_path = os.path.join(temp_data_dir, "final_comparison.csv")
    final_df.to_csv(merged_path, index=False)
    print(f"Combined table saved to: {merged_path}")
