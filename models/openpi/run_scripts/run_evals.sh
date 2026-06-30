#!/usr/bin/env bash
# Run the OpenPI pi05 high-motion eval for a LoRA checkpoint.
#
# Usage from the repo root:
#   bash models/openpi/run_scripts/run_evals.sh
#   bash models/openpi/run_scripts/run_evals.sh /path/to/checkpoint/<step>
#   CHECKPOINT=/path/to/checkpoint/<step> bash models/openpi/run_scripts/run_evals.sh
#
# Environment variables:
#   CHECKPOINT     Path to a checkpoint step directory (contains params/)
#   DATASET_ROOT   Dataset root  (default: $DATA_HOME/datasets/screwdriver_so101_clean_v2)
#   DATA_HOME      Quantycat data root  (default: $HOME/quantycat-data)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OPENPI_REPO="${OPENPI_REPO:-$REPO/vendor/openpi}"
OPENPI_VENV="${OPENPI_VENV:-$REPO/.venvs/openpi}"
DATA_HOME="${QUANTYCAT_DATA_HOME:-$HOME/quantycat-data}"
EVAL_SCRIPT="$REPO/models/openpi/eval/core_evals/relative_eval.py"
DATASET_ROOT="${DATASET_ROOT:-$DATA_HOME/datasets/screwdriver_so101_clean_v2}"

CHECKPOINT="${1:-${CHECKPOINT:-}}"

if [ ! -d "$OPENPI_REPO" ]; then
    echo "ERROR: openpi repo not found at: $OPENPI_REPO"
    exit 1
fi

if command -v uv >/dev/null 2>&1; then
    UV_CMD=(uv)
elif python3 -m uv --version >/dev/null 2>&1; then
    UV_CMD=(python3 -m uv)
else
    echo "ERROR: uv is not installed."
    exit 1
fi

if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT_ROOT="$DATA_HOME/checkpoints/openpi/pi05_quantycat_lora"
    echo "Available OpenPI checkpoints:"
    find "$CHECKPOINT_ROOT" -name "params" -type d 2>/dev/null | sed 's#/params$##' | sort | sed 's/^/  /'
    echo ""
    read -r -p "Checkpoint path: " CHECKPOINT
fi

CHECKPOINT="$(realpath -m "$CHECKPOINT")"

if [ ! -d "$CHECKPOINT/params" ]; then
    echo "ERROR: No params/ directory at: $CHECKPOINT"
    exit 1
fi

echo "Running OpenPI pi05 eval"
echo "  checkpoint:  $CHECKPOINT"
echo "  dataset:     $DATASET_ROOT"
echo ""

cd "$OPENPI_REPO"

export UV_PROJECT_ENVIRONMENT="$OPENPI_VENV"

HF_LEROBOT_HOME="$DATA_HOME/datasets" \
    "${UV_CMD[@]}" run python "$EVAL_SCRIPT" \
    --checkpoint "$CHECKPOINT" \
    --dataset-root "$DATASET_ROOT"
