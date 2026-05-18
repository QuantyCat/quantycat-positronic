#!/usr/bin/env bash
# Fine-tune pi05 on the Quantycat SO-101 screwdriver dataset.
#
# Usage from /home/caroline/Desktop/quantycat-positronic:
#   bash models/openpi/run_scripts/training.sh

set -euo pipefail

OPENPI_REPO="${OPENPI_REPO:-/home/caroline/Desktop/OpenPi}"
CONFIG_NAME="${CONFIG_NAMcaE:-pi05_quantycat_lora}"
EXP_NAME="${EXP_NAME:-screwdriver_so101_pi05_h20_lora_v1}"
NORM_STATS_PATH="/home/caroline/Desktop/quantycat-positronic/my_data/training_pipeline/openpi/norm_stats.json"
CHECKPOINT_DIR="/home/caroline/Desktop/quantycat-positronic/my_data/training_pipeline/openpi/checkpoints/${CONFIG_NAME}/${EXP_NAME}"

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

if [ ! -f "$NORM_STATS_PATH" ]; then
    echo "ERROR: norm stats are missing:"
    echo "  $NORM_STATS_PATH"
    echo "Run:"
    echo "  bash models/openpi/run_scripts/preprocess.sh"
    exit 1
fi

cd "$OPENPI_REPO"

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
