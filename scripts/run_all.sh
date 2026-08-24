#!/usr/bin/env bash
# Reproduces the paper's main results end to end: builds the C++ engine if
# needed, then runs the Table 5 ablation study, the Figure 4 main
# comparison, and the Table 4 hardware profile, in that order.
#
# Usage:
#   ./scripts/run_all.sh
#   DATASETS_DIR=/path/to/datasets OUTPUT_DIR=/path/to/results ./scripts/run_all.sh
#   ./scripts/run_all.sh --datasets LAION COCO   # forwarded to every sub-script
#
# Datasets not present under DATASETS_DIR are skipped per-dataset with a
# warning rather than aborting the whole run. Table 4 requires the Linux
# `perf` tool; if it's unavailable or unauthorized (common in containers),
# that step is skipped and the other results still complete.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASETS_DIR="${DATASETS_DIR:-./dataset_benchmark}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXTRA_ARGS=("$@")

echo "=== CAMP-HNSW: reproduce paper results ==="
echo "Repo root:    $REPO_ROOT"
echo "Datasets dir: $DATASETS_DIR"
echo "Output dir:   $OUTPUT_DIR"
echo

# ---- Step 0: build the C++ extensions if they aren't importable yet ----
if ! "$PYTHON_BIN" -c "import hnswlib" 2>/dev/null; then
    echo "[build] hnswlib extension not found, building..."
    pip install --no-build-isolation -e "$REPO_ROOT/src/hnswlib"
fi
if ! "$PYTHON_BIN" -c "import ours_backend" 2>/dev/null; then
    echo "[build] ours_backend extension not found, building..."
    pip install --no-build-isolation -e "$REPO_ROOT/src"
fi

if [ ! -d "$DATASETS_DIR" ]; then
    echo
    echo "[error] Datasets directory not found: $DATASETS_DIR"
    echo "        Download the benchmark .hdf5 files (Table 1) first, then either"
    echo "        place them under ./dataset_benchmark or run:"
    echo "          DATASETS_DIR=/path/to/datasets $0"
    exit 1
fi

run_step () {
    local name="$1"; shift
    echo
    echo "############################################################"
    echo "# $name"
    echo "############################################################"
    "$PYTHON_BIN" "$@" --datasets-dir "$DATASETS_DIR" "${EXTRA_ARGS[@]}"
}

run_step "Table 5: Incremental Ablation Study" \
    "$REPO_ROOT/scripts/run_table5_ablation.py" --output-dir "$OUTPUT_DIR/table5"

run_step "Figure 4: Main Recall vs QPS Comparison" \
    "$REPO_ROOT/scripts/run_figure4_main_comparison.py" --output-dir "$OUTPUT_DIR/figure4"

echo
echo "############################################################"
echo "# Table 4: Hardware Profile (L3 cache / CPU time via perf)"
echo "############################################################"
if command -v perf >/dev/null 2>&1; then
    if ! "$PYTHON_BIN" "$REPO_ROOT/scripts/run_table4_hardware_profile.py" \
            --datasets-dir "$DATASETS_DIR" --output-dir "$OUTPUT_DIR/table4" "${EXTRA_ARGS[@]}"; then
        echo "[warn] Table 4 profiling failed (often a perf permissions issue in" \
             "containers/CI) -- skipping, other results are unaffected."
    fi
else
    echo "[skip] 'perf' not found on PATH -- skipping Table 4 hardware profiling."
fi

echo
echo "=== All done. Results are under: $OUTPUT_DIR ==="
