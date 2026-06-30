#!/usr/bin/env bash
# Run OpenPI high-motion evals for the Dacha pi0.5 LoRA checkpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
OPENPI_REPO="${OPENPI_REPO:-$REPO/vendor/openpi}"
POSITRONIC_REPO="${QUANTYCAT_POSITRONIC_REPO:-$REPO}"
PATCH_SRC="$REPO/models/openpi/vendor_patches/src"
ANALYSIS_DIR="$REPO/models/openpi/eval/analysis/dacha"
CONFIG_NAME="${CONFIG_NAME:-pi05_dacha_lora}"
EXP_NAME="${EXP_NAME:-06202026_pi05_dacha_lora}"
DATA_HOME="${QUANTYCAT_DATA_HOME:-$HOME/quantycat-data}"
DATASET_ROOT="${DATASET_ROOT:-$DATA_HOME/datasets/dacha/dacha_v3_openpi_v21}"
CHECKPOINT_ROOT="$DATA_HOME/checkpoints/openpi/dacha/$CONFIG_NAME/$EXP_NAME"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_HOME/eval_output/openpi/dacha/pi05}"
SPLIT="${SPLIT:-holdout}"
TOP_K="${TOP_K:-5}"
JOINTS="${JOINTS:-0,1,2,3,4}"
ACTION_HORIZON="${ACTION_HORIZON:-50}"
WINDOW_SIZE="${WINDOW_SIZE:-50}"
SAMPLE_STEPS="${SAMPLE_STEPS:-10}"
SAVE_TRACES="${SAVE_TRACES:-1}"
RUN_COUPLING_DIAGNOSTIC="${RUN_COUPLING_DIAGNOSTIC:-1}"
RUN_LAG_SWEEP_DIAGNOSTIC="${RUN_LAG_SWEEP_DIAGNOSTIC:-1}"
RUN_EXECUTION_INDEX_SWEEP_DIAGNOSTIC="${RUN_EXECUTION_INDEX_SWEEP_DIAGNOSTIC:-1}"
PROXIMAL_JOINTS="${PROXIMAL_JOINTS:-0,1}"
DISTAL_JOINTS="${DISTAL_JOINTS:-2,3,4}"
LAG_SWEEP_MIN="${LAG_SWEEP_MIN:--25}"
LAG_SWEEP_MAX="${LAG_SWEEP_MAX:-25}"
EXECUTION_SWEEP_GT_INDEX="${EXECUTION_SWEEP_GT_INDEX:-0}"
EXECUTION_SWEEP_MAX_INDEX="${EXECUTION_SWEEP_MAX_INDEX:-}"

case "$SPLIT" in
    holdout)
        EPISODES="${EPISODES:-40,41,42,43,44,45,46,47,48,49}"
        ;;
    train)
        EPISODES="${EPISODES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39}"
        ;;
    *)
        EPISODES="${EPISODES:-$SPLIT}"
        SPLIT="custom"
        ;;
esac

if [ ! -d "$OPENPI_REPO" ]; then
    echo "ERROR: openpi repo not found at: $OPENPI_REPO"
    exit 1
fi

if [ ! -d "$DATASET_ROOT" ]; then
    echo "ERROR: Dacha eval dataset not found at: $DATASET_ROOT"
    exit 1
fi

if command -v uv >/dev/null 2>&1; then
    UV_CMD=(uv)
elif python3 -m uv --version >/dev/null 2>&1; then
    UV_CMD=(python3 -m uv)
else
    echo "ERROR: uv is not installed or not runnable."
    exit 1
fi

CHECKPOINT="${1:-${CHECKPOINT:-}}"
if [ -z "$CHECKPOINT" ]; then
    if [ ! -d "$CHECKPOINT_ROOT" ]; then
        echo "ERROR: checkpoint root not found: $CHECKPOINT_ROOT"
        exit 1
    fi
    CHECKPOINT="$(find "$CHECKPOINT_ROOT" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9]+' -printf '%f\n' \
        | sort -n \
        | tail -n 1)"
    if [ -z "$CHECKPOINT" ]; then
        echo "ERROR: no numeric checkpoint directories found under: $CHECKPOINT_ROOT"
        exit 1
    fi
    CHECKPOINT="$CHECKPOINT_ROOT/$CHECKPOINT"
fi

CHECKPOINT="$(realpath -m "$CHECKPOINT")"
STEP="$(basename "$CHECKPOINT")"
LABEL="${LABEL:-dacha_${EXP_NAME}_${STEP}_${SPLIT}}"

if [ ! -d "$CHECKPOINT/params" ]; then
    echo "ERROR: no params/ directory at checkpoint: $CHECKPOINT"
    exit 1
fi

ARGS=(
    --checkpoint "$CHECKPOINT"
    --config-name "$CONFIG_NAME"
    --dataset-root "$DATASET_ROOT"
    --label "$LABEL"
    --episodes "$EPISODES"
    --joints "$JOINTS"
    --top-k "$TOP_K"
    --action-horizon "$ACTION_HORIZON"
    --window-size "$WINDOW_SIZE"
    --sample-steps "$SAMPLE_STEPS"
    --output-root "$OUTPUT_ROOT"
    --force
)

if [ "$SAVE_TRACES" != "0" ]; then
    ARGS+=(--save-traces)
fi

cd "$OPENPI_REPO"

export QUANTYCAT_DATA_HOME="$DATA_HOME"
export QUANTYCAT_POSITRONIC_REPO="$POSITRONIC_REPO"
export HF_LEROBOT_HOME="$DATA_HOME/datasets"
export DACHA_EVAL_PROMPT="${DACHA_EVAL_PROMPT:-pick up the white square and put it in the cup}"
export PYTHONPATH="$PATCH_SRC:$OPENPI_REPO/src:${PYTHONPATH:-}"

echo "Running Dacha OpenPI high-motion eval"
echo "  checkpoint:  $CHECKPOINT"
echo "  config:      $CONFIG_NAME"
echo "  dataset:     $DATASET_ROOT"
echo "  split:       $SPLIT"
echo "  episodes:    $EPISODES"
echo "  horizon:     $ACTION_HORIZON"
echo "  window:      $WINDOW_SIZE"
echo "  output:      $OUTPUT_ROOT/$LABEL"
echo ""

"${UV_CMD[@]}" run python "$ANALYSIS_DIR/_eval_dacha_launcher.py" "${ARGS[@]}"

if [ "$SAVE_TRACES" != "0" ] && [ "$RUN_COUPLING_DIAGNOSTIC" != "0" ]; then
    "${UV_CMD[@]}" run python "$ANALYSIS_DIR/diagnose_proximal_coupling.py" \
        "$OUTPUT_ROOT/$LABEL" \
        --proximal-joints "$PROXIMAL_JOINTS" \
        --distal-joints "$DISTAL_JOINTS"
fi

if [ "$SAVE_TRACES" != "0" ] && [ "$RUN_LAG_SWEEP_DIAGNOSTIC" != "0" ]; then
    "${UV_CMD[@]}" run python "$ANALYSIS_DIR/diagnose_lag_sweep.py" \
        "$OUTPUT_ROOT/$LABEL" \
        --joints "$JOINTS" \
        --lag-min "$LAG_SWEEP_MIN" \
        --lag-max "$LAG_SWEEP_MAX"
fi

if [ "$SAVE_TRACES" != "0" ] && [ "$RUN_EXECUTION_INDEX_SWEEP_DIAGNOSTIC" != "0" ]; then
    EXECUTION_SWEEP_ARGS=(
        "$OUTPUT_ROOT/$LABEL"
        --joints "$JOINTS"
        --gt-index "$EXECUTION_SWEEP_GT_INDEX"
    )
    if [ -n "$EXECUTION_SWEEP_MAX_INDEX" ]; then
        EXECUTION_SWEEP_ARGS+=(--max-index "$EXECUTION_SWEEP_MAX_INDEX")
    fi
    "${UV_CMD[@]}" run python "$ANALYSIS_DIR/diagnose_execution_index_sweep.py" "${EXECUTION_SWEEP_ARGS[@]}"
fi
