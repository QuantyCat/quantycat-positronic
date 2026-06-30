#!/usr/bin/env bash
# Run fixed-window train-vs-normal-holdout diagnostics for v5 orange-wire checkpoints.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
OPENPI_REPO="${OPENPI_REPO:-$REPO/vendor/openpi}"
DATA_HOME="${QUANTYCAT_DATA_HOME:-$HOME/quantycat-data}"
PATCH_SRC="$REPO/models/openpi/vendor_patches/src"
ANALYSIS_DIR="$REPO/models/openpi/eval/analysis/dacha"

CONFIG_NAME="${CONFIG_NAME:-pi05_dacha_wire_v5_h50_early_weighted}"
EXP_NAME="${EXP_NAME:-06282026_pi05_dacha_wire_v5_h50_early_weighted}"
STEP="${STEP:-9999}"
DATASET_ROOT="${DATASET_ROOT:-$DATA_HOME/datasets/dacha/dacha_v5_openpi_v21}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_HOME/eval_output/openpi/dacha/pi05_wire_v5}"
CHECKPOINT="${CHECKPOINT:-$DATA_HOME/checkpoints/openpi/dacha/$CONFIG_NAME/$EXP_NAME/$STEP}"
STRATEGY="${STRATEGY:-even}"
WINDOW_SIZE="${WINDOW_SIZE:-50}"
ACTION_HORIZON="${ACTION_HORIZON:-50}"
SAMPLE_STEPS="${SAMPLE_STEPS:-10}"
MAX_WINDOWS="${MAX_WINDOWS:-10}"
WINDOWS_PER_EPISODE="${WINDOWS_PER_EPISODE:-1}"
SEED="${SEED:-17}"
TRAIN_EPISODES="${TRAIN_EPISODES:-0-9,10-19,20-29,30-39,40-59,70-87,91-97}"
NORMAL_HOLDOUT_EPISODES="${NORMAL_HOLDOUT_EPISODES:-60-69}"
LABEL_PREFIX="${LABEL_PREFIX:-dacha_wire_v5_h50_early_weighted_${STEP}_${STRATEGY}_window}"
LOG_DIR="${LOG_DIR:-$DATA_HOME/logs/openpi/dacha}"

if command -v uv >/dev/null 2>&1; then
    UV_CMD=(uv)
elif python3 -m uv --version >/dev/null 2>&1; then
    UV_CMD=(python3 -m uv)
else
    echo "ERROR: uv is not installed or not runnable."
    exit 1
fi

run_split() {
    local split="$1"
    local episodes="$2"
    local label="${LABEL_PREFIX}_${split}"
    local log="$LOG_DIR/${label}.log"

    echo "Running $STRATEGY-window diagnostic: $split"
    echo "  checkpoint: $CHECKPOINT"
    echo "  dataset:    $DATASET_ROOT"
    echo "  episodes:   $episodes"
    echo "  output:     $OUTPUT_ROOT/$label"

    cd "$OPENPI_REPO"
    mkdir -p "$LOG_DIR"
    export QUANTYCAT_DATA_HOME="$DATA_HOME"
    export QUANTYCAT_POSITRONIC_REPO="${QUANTYCAT_POSITRONIC_REPO:-$REPO}"
    export HF_LEROBOT_HOME="$DATA_HOME/datasets"
    export DACHA_EVAL_PROMPT="${DACHA_EVAL_PROMPT:-pick up the orange wire and put it in the cup}"
    export PYTHONPATH="$PATCH_SRC:$OPENPI_REPO/src:${PYTHONPATH:-}"

    "${UV_CMD[@]}" run python "$ANALYSIS_DIR/eval_dacha_window_strategy.py" \
        --checkpoint "$CHECKPOINT" \
        --config-name "$CONFIG_NAME" \
        --dataset-root "$DATASET_ROOT" \
        --label "$label" \
        --episodes "$episodes" \
        --output-root "$OUTPUT_ROOT" \
        --strategy "$STRATEGY" \
        --window-size "$WINDOW_SIZE" \
        --action-horizon "$ACTION_HORIZON" \
        --sample-steps "$SAMPLE_STEPS" \
        --windows-per-episode "$WINDOWS_PER_EPISODE" \
        --max-windows "$MAX_WINDOWS" \
        --seed "$SEED" \
        --save-traces \
        2>&1 | tee "$log"
}

run_split train "$TRAIN_EPISODES"
run_split normal_holdout "$NORMAL_HOLDOUT_EPISODES"
