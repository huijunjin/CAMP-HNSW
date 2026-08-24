#!/usr/bin/env bash
# Reproduces the paper's results end to end: builds the C++ engine if
# needed, then runs every reproduction script in scripts/ in order --
# Table 5, Figure 4, Table 2/3, Figure 6, Figure 7, Figure 8, Figure 9,
# then Table 4.
#
# Usage:
#   ./scripts/run_all.sh
#   DATASETS_DIR=/path/to/datasets OUTPUT_DIR=/path/to/results ./scripts/run_all.sh
#   ./scripts/run_all.sh --datasets LAION COCO
#
# Any extra arguments (e.g. `--datasets LAION COCO`, `--budget 500`) are
# forwarded to every OOD-dataset script (Table 5, Figure 4/6/7/9, Table 4),
# since they all share the same --datasets-dir / --datasets vocabulary
# (Table 1's ImageNet/LAION/COCO/MainSearch). Figure 8 runs over the
# separate in-distribution dataset pair (GloVe / ID-ImageNet) and is only
# given --datasets-dir, since its --datasets choices differ.
#
# On a default (no --datasets) invocation, a preflight check requires all 6
# expected dataset files (see README.md Sec. 4.2) to be present and exits 1
# with a hint if any are missing, since a missing/misnamed file is otherwise
# a silent per-dataset skip rather than a crash. Passing --datasets narrows
# this check to a warning, since you're explicitly asking for a subset.
# Table 4 requires the Linux `perf` tool; if it's unavailable or unauthorized
# (common in containers), that step is skipped and the other results still
# complete.
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
    echo "        place them under ./dataset_benchmark, or generate a small synthetic"
    echo "        set to sanity-check the pipeline:"
    echo "          python scripts/generate_dummy_data.py"
    echo "        or point at a real directory:"
    echo "          DATASETS_DIR=/path/to/datasets $0"
    exit 1
fi

# ---- Preflight: check that the 6 expected dataset files are present ----
# A missing/misnamed file is NOT a crash in the individual scripts -- they
# just print "[skip] dataset file not found" and move on, so a whole run can
# silently finish with exit 0 and zero real results if a name doesn't match.
# This check surfaces that up front instead of hours into a run.
EXPECTED_DATASET_FILES=(imagenet laion coco mainsearch glove imagenet_id)

HAS_DATASETS_OVERRIDE=false
for arg in "${EXTRA_ARGS[@]}"; do
    if [ "$arg" = "--datasets" ]; then
        HAS_DATASETS_OVERRIDE=true
        break
    fi
done

echo "Dataset file check (under $DATASETS_DIR):"
MISSING_DATASET_FILES=()
for name in "${EXPECTED_DATASET_FILES[@]}"; do
    if [ -f "$DATASETS_DIR/$name.hdf5" ]; then
        echo "  [found]   $name.hdf5"
    else
        echo "  [missing] $name.hdf5"
        MISSING_DATASET_FILES+=("$name")
    fi
done
echo

if [ ${#MISSING_DATASET_FILES[@]} -gt 0 ]; then
    if [ "$HAS_DATASETS_OVERRIDE" = true ]; then
        echo "[warn] Some expected files are missing (listed above), but continuing since"
        echo "       --datasets was passed explicitly -- affected scripts will just skip"
        echo "       the datasets they can't find."
        echo
    else
        echo "[error] Missing ${#MISSING_DATASET_FILES[@]}/6 expected dataset file(s): ${MISSING_DATASET_FILES[*]}"
        echo "        Fix options:"
        echo "          - Place the missing file(s) under $DATASETS_DIR (README.md Sec. 4.2)."
        echo "            glove.hdf5 / imagenet_id.hdf5 need their raw files processed first:"
        echo "              python scripts/prepare_id_datasets.py --datasets-dir $DATASETS_DIR"
        echo "          - Or run only the datasets you have, e.g.:"
        echo "              $0 --datasets LAION COCO"
        echo "          - Or sanity-check the pipeline with synthetic data first:"
        echo "              python scripts/generate_dummy_data.py && $0"
        exit 1
    fi
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

run_step "Table 2/3: Edge Composition and Shortcut Utility Analysis" \
    "$REPO_ROOT/scripts/run_table3_shortcut_utility.py" --output-dir "$OUTPUT_DIR/table3"

run_step "Figure 6: Cache Reordering Parameter Sensitivity" \
    "$REPO_ROOT/scripts/run_figure6_kmeans_tuning.py" --output-dir "$OUTPUT_DIR/figure6"

run_step "Figure 7: Shortcut Mining Pareto Frontier" \
    "$REPO_ROOT/scripts/run_figure7_pareto_search.py" --output-dir "$OUTPUT_DIR/figure7"

echo
echo "############################################################"
echo "# Figure 8: Zero Routing Penalty on ID Datasets"
echo "############################################################"
"$PYTHON_BIN" "$REPO_ROOT/scripts/run_figure8_id_robustness.py" \
    --datasets-dir "$DATASETS_DIR" --output-dir "$OUTPUT_DIR/figure8"

run_step "Figure 9: Data Efficiency vs Calibration Query Ratio" \
    "$REPO_ROOT/scripts/run_figure9_query_ratio.py" --output-dir "$OUTPUT_DIR/figure9"

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
