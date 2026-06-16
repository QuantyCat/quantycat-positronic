#!/usr/bin/env bash
# Run the LeRobot pi05 high-motion eval for a LoRA checkpoint.
#
# Usage from the repo root:
#   bash models/lerobot/run_scripts/run_evals.sh
#   bash models/lerobot/run_scripts/run_evals.sh /path/to/pretrained_model
#   CHECKPOINT=/path/to/pretrained_model bash models/lerobot/run_scripts/run_evals.sh
#
# Environment variables:
#   CHECKPOINT     Path to a run dir, checkpoint dir, or pretrained_model dir
#   DATASET_ROOT   LeRobot dataset root  (default: $DATA_HOME/datasets/screwdriver_so101_clean_v3)
#   DATA_HOME      Quantycat data root   (default: $HOME/quantycat-data)
#   LEROBOT_VENV   LeRobot venv path     (default: .venvs/lerobot)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_HOME="${QUANTYCAT_DATA_HOME:-$HOME/quantycat-data}"
LEROBOT_VENV="${LEROBOT_VENV:-$REPO/.venvs/lerobot}"
PYTHON="$LEROBOT_VENV/bin/python"
EVAL_SCRIPT="$REPO/models/lerobot/eval/relative_eval.py"
DATASET_ROOT="${DATASET_ROOT:-$DATA_HOME/datasets/screwdriver_so101_clean_v3}"

CHECKPOINT="${1:-${CHECKPOINT:-}}"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: LeRobot python not found: $PYTHON"
    echo "Run setup first:"
    echo "  bash models/lerobot/run_scripts/setup.sh"
    exit 1
fi

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "ERROR: eval script not found: $EVAL_SCRIPT"
    exit 1
fi

latest_pretrained_model() {
    local root="$1"
    find "$root" -path "*/checkpoints/*/pretrained_model/adapter_config.json" -type f 2>/dev/null \
        | sort \
        | tail -n 1 \
        | sed 's#/adapter_config.json$##'
}

resolve_checkpoint() {
    local value="$1"
    local path
    path="$(realpath -m "$value")"

    if [ -f "$path/adapter_config.json" ]; then
        echo "$path"
        return 0
    fi

    if [ -f "$path/pretrained_model/adapter_config.json" ]; then
        echo "$path/pretrained_model"
        return 0
    fi

    if [ -d "$path/checkpoints" ]; then
        latest_pretrained_model "$path"
        return 0
    fi

    echo "$path"
}

if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT_ROOT="$DATA_HOME/checkpoints/lerobot"
    echo "Available LeRobot pi05 checkpoints:"
    latest_pretrained_model "$CHECKPOINT_ROOT" | sed 's/^/  /'
    echo ""
    read -r -p "Checkpoint path: " CHECKPOINT
fi

CHECKPOINT="$(resolve_checkpoint "$CHECKPOINT")"

if [ ! -f "$CHECKPOINT/adapter_config.json" ]; then
    echo "ERROR: No adapter_config.json at: $CHECKPOINT"
    echo "Pass a pretrained_model directory, a checkpoint step directory, or a run directory."
    exit 1
fi

echo "Running LeRobot pi05 eval"
echo "  checkpoint:  $CHECKPOINT"
echo "  dataset:     $DATASET_ROOT"
echo ""


"$PYTHON" "$EVAL_SCRIPT" \
    --checkpoint "$CHECKPOINT" \
    --dataset-root "$DATASET_ROOT"
