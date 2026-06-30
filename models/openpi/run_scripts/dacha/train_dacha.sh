#!/usr/bin/env bash
# Fine-tune pi0.5 LoRA on the Dacha SO-101 dataset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
OPENPI_REPO="${OPENPI_REPO:-$REPO/vendor/openpi}"
PATCH_SRC="$REPO/models/openpi/vendor_patches/src"
CONFIG_NAME="${CONFIG_NAME:-pi05_dacha_lora}"
EXP_NAME="${EXP_NAME:-$(TZ=America/Los_Angeles date +%m%d%Y)_pi05_dacha_lora}"
DATA_HOME="${QUANTYCAT_DATA_HOME:-$HOME/quantycat-data}"
DATASET_ROOT="$DATA_HOME/datasets/dacha/dacha_v3_openpi_v21"
CHECKPOINT_DIR="$DATA_HOME/checkpoints/openpi/dacha/$CONFIG_NAME/$EXP_NAME"
LOG_DIR="${LOG_DIR:-$DATA_HOME/logs/openpi/dacha}"
LOG_PATH="$LOG_DIR/${EXP_NAME}.log"
SPLIT_PATH="$LOG_DIR/${EXP_NAME}_split.json"

if [ ! -d "$OPENPI_REPO" ]; then
    echo "ERROR: openpi repo not found at: $OPENPI_REPO"
    exit 1
fi

if [ ! -d "$DATASET_ROOT" ]; then
    echo "ERROR: dacha dataset not found at: $DATASET_ROOT"
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

mkdir -p "$LOG_DIR"

cat >"$SPLIT_PATH" <<EOF
{
  "dataset": "$DATASET_ROOT",
  "config": "$CONFIG_NAME",
  "exp_name": "$EXP_NAME",
  "checkpoint_dir": "$CHECKPOINT_DIR",
  "train_episodes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
  "holdout_episodes": [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
}
EOF

cd "$OPENPI_REPO"

export QUANTYCAT_DATA_HOME="$DATA_HOME"
export QUANTYCAT_POSITRONIC_REPO="$REPO"
export HF_LEROBOT_HOME="$DATA_HOME/datasets"
export PYTHONPATH="$PATCH_SRC:$OPENPI_REPO/src:${PYTHONPATH:-}"

echo "Starting OpenPI Dacha training"
echo "  config:       $CONFIG_NAME"
echo "  exp name:     $EXP_NAME"
echo "  dataset:      $DATASET_ROOT"
echo "  train:        episodes 0-39"
echo "  holdout:      episodes 40-49"
echo "  checkpoints:  $CHECKPOINT_DIR"
echo "  split record: $SPLIT_PATH"
echo "  log:          $LOG_PATH"
echo ""

XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
    "${UV_CMD[@]}" run scripts/train.py "$CONFIG_NAME" --exp-name="$EXP_NAME" --overwrite 2>&1 | tee "$LOG_PATH"

echo ""
echo "Training command finished."
echo "Checkpoints:"
echo "  $CHECKPOINT_DIR"
