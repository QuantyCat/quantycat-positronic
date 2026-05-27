#!/usr/bin/env bash
# Fine-tune pi05 on the Quantycat SO-101 screwdriver dataset.
#
# Usage from the repo root:
#   bash models/openpi/run_scripts/training.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OPENPI_REPO="${OPENPI_REPO:-$REPO/vendor/openpi}"
CONFIG_NAME="${CONFIG_NAME:-pi05_quantycat_lora}"
EXP_NAME="${EXP_NAME:-05232026_pi05_lora}"
DATA_HOME="${QUANTYCAT_DATA_HOME:-$HOME/quantycat-data}"
NORM_STATS_DIR="$DATA_HOME/norm_stats/openpi/${CONFIG_NAME}"
CHECKPOINT_DIR="$DATA_HOME/checkpoints/openpi/${CONFIG_NAME}/${EXP_NAME}"

if [ ! -d "$OPENPI_REPO" ]; then
    echo "ERROR: openpi repo not found at: $OPENPI_REPO"
    exit 1
fi

if command -v uv >/dev/null 2>&1; then
    UV_CMD=(uv)
elif python3 -m uv --version >/dev/null 2>&1; then
    UV_CMD=(python3 -m uv)
else
    echo "ERROR: uv is not installed or not runnable. Run setup first:"
    echo "  bash models/openpi/run_scripts/setup.sh"
    exit 1
fi

if [ -z "$(find "$NORM_STATS_DIR" -name "norm_stats.json" 2>/dev/null)" ]; then
    echo "ERROR: norm stats are missing under:"
    echo "  $NORM_STATS_DIR"
    echo "Run:"
    echo "  bash models/openpi/run_scripts/preprocess.sh"
    exit 1
fi

cd "$OPENPI_REPO"

export PYTHONPATH="$REPO/models/openpi/training_config${PYTHONPATH:+:$PYTHONPATH}"
export HF_LEROBOT_HOME="$DATA_HOME/datasets"

echo "Starting openpi training"
echo "  config:      $CONFIG_NAME"
echo "  exp name:    $EXP_NAME"
echo "  checkpoints: $CHECKPOINT_DIR"
echo ""

XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
    "${UV_CMD[@]}" run scripts/train.py "$CONFIG_NAME" --exp-name="$EXP_NAME" --overwrite

echo ""
echo "Training command finished."
echo "Checkpoints:"
echo "  $CHECKPOINT_DIR"
