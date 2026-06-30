#!/usr/bin/env bash
# Fine-tune pi05 on a Quantycat SO-101 dataset.
#
# Usage from the repo root:
#   bash models/openpi/run_scripts/train.sh
#
# Extra arguments are forwarded to scripts/train.py, e.g. to point the
# dataset-less pi05_quantycat_template config at a real dataset:
#   bash models/openpi/run_scripts/train.sh --data.repo-id=<dataset>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OPENPI_REPO="${OPENPI_REPO:-$REPO/vendor/openpi}"
OPENPI_VENV="${OPENPI_VENV:-$REPO/.venvs/openpi}"
CONFIG_NAME="${CONFIG_NAME:-pi05_quantycat_template}"

# Derive a default EXP_NAME from --data.repo-id (if passed) so checkpoint/log
# paths identify the dataset, not just a generic date stamp.
DATASET_NAME=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    arg="${args[$i]}"
    if [[ "$arg" == --data.repo-id=* ]]; then
        DATASET_NAME="${arg#--data.repo-id=}"
    elif [[ "$arg" == "--data.repo-id" ]]; then
        DATASET_NAME="${args[$((i + 1))]:-}"
    fi
done
DATASET_NAME="${DATASET_NAME//\//_}"

if [ -n "$DATASET_NAME" ]; then
    DEFAULT_EXP_NAME="$(TZ=America/Los_Angeles date +%m%d%Y)_${DATASET_NAME}"
else
    DEFAULT_EXP_NAME="$(TZ=America/Los_Angeles date +%m%d%Y)_pi05_openpi"
fi
EXP_NAME="${EXP_NAME:-$DEFAULT_EXP_NAME}"
DATA_HOME="${QUANTYCAT_DATA_HOME:-$HOME/quantycat-data}"
CHECKPOINT_DIR="$DATA_HOME/checkpoints/openpi/${CONFIG_NAME}/${EXP_NAME}"
LOG_DIR="${LOG_DIR:-$DATA_HOME/logs/openpi}"
LOG_PATH="$LOG_DIR/${EXP_NAME}.log"

if [ ! -d "$OPENPI_REPO" ]; then
    echo "ERROR: openpi repo not found at: $OPENPI_REPO"
    exit 1
fi

if [ ! -d "$DATA_HOME/datasets" ]; then
    echo "ERROR: no datasets found at: $DATA_HOME/datasets"
    echo "Run preprocessing first:"
    echo "  bash models/openpi/run_scripts/preprocess.sh"
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

mkdir -p "$LOG_DIR"

cd "$OPENPI_REPO"

export UV_PROJECT_ENVIRONMENT="$OPENPI_VENV"
export HF_LEROBOT_HOME="$DATA_HOME/datasets"

echo "Starting openpi training"
echo "  config:      $CONFIG_NAME"
echo "  exp name:    $EXP_NAME"
echo "  checkpoints: $CHECKPOINT_DIR"
echo "  log:         $LOG_PATH"
echo ""

# The full, unfiltered output always goes to LOG_PATH. The live screen only
# shows progress/checkpoint/error lines — openpi/JAX/orbax startup logging is
# extremely verbose (parameter shape dumps, checkpoint handler internals,
# backend probing) and drowns out anything useful.
SCREEN_FILTER='Progress on:|Step [0-9]+:|Restoring checkpoint|Finished restoring|Found [0-9]+ checkpoint|[Ss]aving checkpoint|[Ss]aved checkpoint|\[(W|E|C)\]|Traceback|Error:|error:'

XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
    PYTHONUNBUFFERED=1 \
    "${UV_CMD[@]}" run scripts/train.py "$CONFIG_NAME" --exp-name="$EXP_NAME" --overwrite "$@" 2>&1 \
    | tee "$LOG_PATH" | grep --line-buffered -E "$SCREEN_FILTER"

echo ""
echo "Training command finished."
echo "Checkpoints:"
echo "  $CHECKPOINT_DIR"
echo ""
echo "Retained checkpoints:"
find "$CHECKPOINT_DIR" -mindepth 1 -maxdepth 1 -type d -printf '  %f\n' | sort -n
