#!/usr/bin/env bash
# Find nearest train windows for weak normal-holdout windows in v5 diagnostics.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_HOME="${QUANTYCAT_DATA_HOME:-$HOME/quantycat-data}"
REPO="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
ANALYSIS_DIR="$REPO/models/openpi/eval/analysis/dacha"

DATASET_ROOT="${DATASET_ROOT:-$DATA_HOME/datasets/dacha/dacha_v5_openpi_v21}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_HOME/eval_output/openpi/dacha/pi05_wire_v5/nearest_train_windows}"
TRAIN_EPISODES="${TRAIN_EPISODES:-0-59,70-87,91-97}"
WINDOW_SIZE="${WINDOW_SIZE:-50}"
STRIDE="${STRIDE:-5}"
TOP_K="${TOP_K:-10}"
SLOPE_THRESHOLD="${SLOPE_THRESHOLD:-0.6}"
RANK_BY="${RANK_BY:-full}"
PYTHON_BIN="${PYTHON_BIN:-$REPO/vendor/openpi/.venv/bin/python}"

run_summary() {
    local summary="$1"
    "$PYTHON_BIN" "$ANALYSIS_DIR/diagnose_nearest_train_windows.py" \
        --dataset-root "$DATASET_ROOT" \
        --summary "$summary" \
        --output-dir "$OUTPUT_DIR" \
        --train-episodes "$TRAIN_EPISODES" \
        --window-size "$WINDOW_SIZE" \
        --stride "$STRIDE" \
        --top-k "$TOP_K" \
        --rank-by "$RANK_BY" \
        --slope-threshold "$SLOPE_THRESHOLD"
}

run_summary "$DATA_HOME/eval_output/openpi/dacha/pi05_wire_v5/dacha_wire_v5_h50_early_weighted_9999_even_window_normal_holdout/window_strategy_summary.json"
run_summary "$DATA_HOME/eval_output/openpi/dacha/pi05_wire_v5/dacha_wire_v5_h50_early_weighted_9999_random_window_normal_holdout/window_strategy_summary.json"
