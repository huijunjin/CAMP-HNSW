#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <mutex>
#include <chrono>
#include <iostream>

namespace py = pybind11;

inline float compute_ip(const float* a, const float* b, int dim) {
    float res = 0;
    for (int i = 0; i < dim; i++) res += a[i] * b[i];
    return res;
}

// Phase 2 (Targeted Shortcut Mining): for every calibration query, diff the
// retrieved neighborhood against the ground truth to find the missing targets,
// then emit bidirectional candidate edges between them (Algorithm 2, lines 6-11).
std::pair<std::vector<int>, std::vector<int>> mine_shortcuts_cpp(
    py::array_t<int> labels,
    py::array_t<int> gt,
    int top_k,
    int n_cores
) {
    auto r_labels = labels.unchecked<2>(); // [num_queries, top_k]
    auto r_gt = gt.unchecked<2>();         // [num_queries, gt_len]

    int num_queries = r_labels.shape(0);
    int gt_len = r_gt.shape(1);
    int k = r_labels.shape(1);

    // Numpy/OpenMP-based libraries loaded earlier in the process (e.g. faiss)
    // can pin the OpenMP thread count to 1; restore it explicitly here.
    omp_set_num_threads(n_cores);

    std::vector<std::vector<int>> thread_rows(n_cores);
    std::vector<std::vector<int>> thread_cols(n_cores);

    py::gil_scoped_release release;

    #pragma omp parallel for
    for (int i = 0; i < num_queries; i++) {
        int thread_id = omp_get_thread_num();

        std::vector<int> found;
        found.reserve(k);
        for (int j = 0; j < k; j++) found.push_back(r_labels(i, j));
        std::sort(found.begin(), found.end());

        std::vector<int> targets;
        targets.reserve(top_k);
        for (int j = 0; j < std::min(gt_len, top_k); j++) {
            int val = r_gt(i, j);
            if (val != -1) targets.push_back(val); // -1 is padding
        }
        std::sort(targets.begin(), targets.end());

        std::vector<int> missing;
        std::set_difference(targets.begin(), targets.end(),
                            found.begin(), found.end(),
                            std::back_inserter(missing));

        if (missing.empty()) continue;

        for (int h_node : found) {
            for (int m_node : missing) {
                if (h_node != m_node) {
                    thread_rows[thread_id].push_back(h_node);
                    thread_cols[thread_id].push_back(m_node);
                    thread_rows[thread_id].push_back(m_node);
                    thread_cols[thread_id].push_back(h_node);
                }
            }
        }
    }

    std::vector<int> final_rows, final_cols;
    for (int i = 0; i < n_cores; i++) {
        final_rows.insert(final_rows.end(), thread_rows[i].begin(), thread_rows[i].end());
        final_cols.insert(final_cols.end(), thread_cols[i].begin(), thread_cols[i].end());
    }

    return {final_rows, final_cols};
}

// Standalone candidate-filtering step (RNG heuristic pre-filtering, Algorithm 2
// line 13). Superseded by the unified mine_and_filter_cpp below, kept for
// backward compatibility with callers that mine and filter separately.
void filter_and_save_cpp(
    py::array_t<int> indptr,
    py::array_t<int> indices,
    py::array_t<float> data_vectors,
    int budget,
    std::string output_path
) {
    auto r_indptr = indptr.unchecked<1>();
    auto r_indices = indices.unchecked<1>();
    auto r_data = data_vectors.unchecked<2>();

    int num_nodes = r_indptr.shape(0) - 1;
    int dim = r_data.shape(1);

    std::vector<std::pair<int, int>> final_edges;
    std::mutex res_mutex;

    py::gil_scoped_release release;

    #pragma omp parallel for
    for (int i = 0; i < num_nodes; i++) {
        int start = r_indptr(i);
        int end = r_indptr(i + 1);
        if (start == end) continue;

        std::vector<int> candidates;
        for (int j = start; j < end; j++) candidates.push_back(r_indices(j));

        if (candidates.size() > budget * 2) {
             std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
                 float dist_a = compute_ip(&r_data(i, 0), &r_data(a, 0), dim);
                 float dist_b = compute_ip(&r_data(i, 0), &r_data(b, 0), dim);
                 return dist_a > dist_b; // larger inner product = closer
             });
             candidates.resize(budget * 2);
        }

        std::vector<std::pair<float, int>> cand_dists;
        for (int c : candidates) {
            cand_dists.push_back({compute_ip(&r_data(i, 0), &r_data(c, 0), dim), c});
        }
        std::sort(cand_dists.rbegin(), cand_dists.rend());

        std::vector<int> selected;
        for (auto& p : cand_dists) {
            if (selected.size() >= budget) break;
            int cand_idx = p.second;
            float dist_to_base = p.first;

            bool is_good = true;
            for (int exist_idx : selected) {
                float dist_between = compute_ip(&r_data(cand_idx, 0), &r_data(exist_idx, 0), dim);
                if (dist_between > dist_to_base) {
                    is_good = false;
                    break;
                }
            }
            if (is_good) selected.push_back(cand_idx);
        }

        std::lock_guard<std::mutex> lock(res_mutex);
        for (int tgt : selected) {
            final_edges.push_back({i, tgt});
        }
    }

    FILE* f = fopen(output_path.c_str(), "wb");
    for (auto& edge : final_edges) {
        unsigned int src = edge.first;
        unsigned int dst = edge.second;
        fwrite(&src, 4, 1, f);
        fwrite(&dst, 4, 1, f);
    }
    fclose(f);
}

// Unified Phase 2 engine: mining (edge diff against GT) + deduplication +
// RNG-heuristic candidate pre-filtering (Algorithm 2, lines 2-13), fused into
// a single pass to avoid materializing the full candidate adjacency in Python.
//
// USE_CANDIDATE_PRUNING=0 disables the RNG diversity filter and instead keeps
// the `budget` closest candidates verbatim; this reproduces the "Naive
// Injection" ablation variant in Table 5.
double mine_and_filter_cpp(
    py::array_t<int> labels,
    py::array_t<int> gt,
    py::array_t<float> data_vectors,
    int top_k,
    int budget,
    int n_cores,
    std::string output_path
) {
    auto r_labels = labels.unchecked<2>();
    auto r_gt = gt.unchecked<2>();
    auto r_data = data_vectors.unchecked<2>();

    int num_queries = r_labels.shape(0);
    int gt_len = r_gt.shape(1);
    int num_nodes = r_data.shape(0);
    int k = r_labels.shape(1);
    int dim = r_data.shape(1);

    omp_set_num_threads(n_cores);
    py::gil_scoped_release release;

    auto t_start = std::chrono::high_resolution_clock::now();
    std::cout << "\n      [C++ Engine] Started..." << std::endl;

    std::vector<std::vector<std::vector<int>>> thread_adj(n_cores, std::vector<std::vector<int>>(num_nodes));

    // ------------------------------------------------------------------------
    // Edge mining: diff each query's retrieved neighborhood against its GT.
    // ------------------------------------------------------------------------
    std::cout << "      [C++] -> Mining missing links (" << n_cores << " cores)..." << std::flush;

    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < num_queries; i++) {
        int tid = omp_get_thread_num();

        std::vector<int> found;
        found.reserve(k);
        for (int j = 0; j < k; j++) found.push_back(r_labels(i, j));
        std::sort(found.begin(), found.end());

        std::vector<int> targets;
        targets.reserve(top_k);
        for (int j = 0; j < std::min(gt_len, top_k); j++) {
            int val = r_gt(i, j);
            if (val != -1) targets.push_back(val);
        }
        std::sort(targets.begin(), targets.end());

        std::vector<int> missing;
        std::set_difference(targets.begin(), targets.end(),
                            found.begin(), found.end(),
                            std::back_inserter(missing));

        if (missing.empty()) continue;

        for (int h_node : found) {
            for (int m_node : missing) {
                if (h_node != m_node) {
                    thread_adj[tid][h_node].push_back(m_node);
                    thread_adj[tid][m_node].push_back(h_node);
                }
            }
        }
    }
    auto t_p2 = std::chrono::high_resolution_clock::now();
    std::cout << " done in " << std::chrono::duration<double>(t_p2 - t_start).count() << "s" << std::endl;

    // ------------------------------------------------------------------------
    // Deduplication + RNG-heuristic candidate pre-filtering, per node.
    // ------------------------------------------------------------------------
    std::cout << "      [C++] -> Deduplication & filtering (" << n_cores << " cores)..." << std::flush;

    bool use_cand_prune = true;
    if (const char* env_p = std::getenv("USE_CANDIDATE_PRUNING")) {
        if (std::string(env_p) == "0") use_cand_prune = false;
    }

    // Thread-local output buffers avoid a global lock on the shared edge list.
    std::vector<std::vector<std::pair<int, int>>> thread_local_edges(n_cores);
    for (int t = 0; t < n_cores; t++) {
        thread_local_edges[t].reserve((num_nodes / n_cores) * budget);
    }

    #pragma omp parallel
    {
        int t_id = omp_get_thread_num();

        #pragma omp for schedule(dynamic, 32)
        for (int i = 0; i < num_nodes; i++) {
            std::vector<int> candidates;
            for (int t = 0; t < n_cores; t++) {
                candidates.insert(candidates.end(), thread_adj[t][i].begin(), thread_adj[t][i].end());
            }

            if (candidates.empty()) continue;

            std::sort(candidates.begin(), candidates.end());
            candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

            // Distances are computed once per candidate and reused for both
            // the truncation step and the heuristic filter below (O(N*D)).
            std::vector<std::pair<float, int>> cand_dists;
            cand_dists.reserve(candidates.size());
            for (int c : candidates) {
                cand_dists.push_back({compute_ip(&r_data(i, 0), &r_data(c, 0), dim), c});
            }
            std::sort(cand_dists.begin(), cand_dists.end(), [](const auto& a, const auto& b) {
                return a.first > b.first; // descending inner product
            });

            if (cand_dists.size() > budget * 2) {
                cand_dists.resize(budget * 2);
            }

            std::vector<int> selected;
            selected.reserve(budget);

            if (use_cand_prune) {
                // RNG heuristic: keep a candidate only if no previously
                // accepted neighbor lies strictly closer to it than to i.
                for (auto& p : cand_dists) {
                    if (selected.size() >= budget) break;
                    int cand_idx = p.second;
                    float dist_to_base = p.first;

                    bool is_good = true;
                    for (int exist_idx : selected) {
                        float dist_between = compute_ip(&r_data(cand_idx, 0), &r_data(exist_idx, 0), dim);
                        if (dist_between > dist_to_base) {
                            is_good = false;
                            break;
                        }
                    }
                    if (is_good) selected.push_back(cand_idx);
                }
            } else {
                // Ablation: ignore RNG diversity, keep the closest `budget`.
                for (auto& p : cand_dists) {
                    if (selected.size() >= budget) break;
                    selected.push_back(p.second);
                }
            }

            for (int tgt : selected) {
                thread_local_edges[t_id].push_back({i, tgt});
            }
        }
    }

    std::vector<std::pair<int, int>> final_edges;
    size_t total_edges = 0;
    for (int t = 0; t < n_cores; t++) total_edges += thread_local_edges[t].size();
    final_edges.reserve(total_edges);
    for (int t = 0; t < n_cores; t++) {
        final_edges.insert(final_edges.end(), thread_local_edges[t].begin(), thread_local_edges[t].end());
    }

    auto t_p34 = std::chrono::high_resolution_clock::now();
    double pure_time = std::chrono::duration<double>(t_p34 - t_start).count();
    std::cout << " done in " << std::chrono::duration<double>(t_p34 - t_start).count() << "s" << std::endl;

    // ------------------------------------------------------------------------
    // File I/O (excluded from the reported pure algorithm time).
    // ------------------------------------------------------------------------
    std::cout << "      [C++] -> Saving candidate edges to disk..." << std::flush;
    FILE* f = fopen(output_path.c_str(), "wb");
    if (f) {
        for (auto& edge : final_edges) {
            unsigned int src = edge.first;
            unsigned int dst = edge.second;
            fwrite(&src, 4, 1, f);
            fwrite(&dst, 4, 1, f);
        }
        fclose(f);
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << " done in " << std::chrono::duration<double>(t_end - t_p34).count() << "s" << std::endl;

    return pure_time;
}

PYBIND11_MODULE(ours_backend, m) {
    m.def("mine_shortcuts", &mine_shortcuts_cpp, "Mine OOD shortcut candidates in C++",
          py::arg("labels"), py::arg("gt"), py::arg("top_k"), py::arg("n_cores"));
    m.def("filter_and_save", &filter_and_save_cpp, "Filter candidate edges (RNG heuristic) and save to disk");
    m.def("mine_and_filter", &mine_and_filter_cpp, "Unified mining + filtering engine (Phase 2)");
}
