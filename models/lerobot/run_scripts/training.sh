#!/usr/bin/env bash
# Fine-tune pi05 via LeRobot on the Quantycat SO-101 screwdriver dataset.
#
# Usage from the repo root:
#   bash models/lerobot/run_scripts/training.sh
#
# Environment variables:
#   DATASET_REPO_ID   LeRobot dataset name under HF_LEROBOT_HOME  (default: screwdriver_so101)
#   EXP_NAME          W&B / output run name  (default: datestamp_pi05_lerobot)
#   LEROBOT_VENV      Path to the lerobot venv  (default: vendor/lerobot/.venv)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LEROBOT_VENV="${LEROBOT_VENV:-$REPO/vendor/lerobot/.venv}"
DATA_HOME="${QUANTYCAT_DATA_HOME:-$HOME/quantycat-data}"
DATASET_REPO_ID="${DATASET_REPO_ID:-screwdriver_so101}"
EXP_NAME="${EXP_NAME:-$(TZ=America/Los_Angeles date +%m%d%Y_%H%M)_pi05_lerobot}"
CHECKPOINT_DIR="$DATA_HOME/checkpoints/lerobot/pi05/$EXP_NAME"

if [ ! -d "$LEROBOT_VENV" ]; then
    echo "ERROR: lerobot venv not found at: $LEROBOT_VENV"
    echo "Run setup first:"
    echo "  bash models/lerobot/run_scripts/setup.sh"
    exit 1
fi

LEROBOT_BIN="$LEROBOT_VENV/bin/lerobot-train"
if [ ! -f "$LEROBOT_BIN" ]; then
    echo "ERROR: lerobot-train not found in venv: $LEROBOT_BIN"
    echo "Run setup first:"
    echo "  bash models/lerobot/run_scripts/setup.sh"
    exit 1
fi

echo "Starting lerobot pi05 training"
echo "  dataset:     $DATASET_REPO_ID"
echo "  exp name:    $EXP_NAME"
echo "  checkpoints: $CHECKPOINT_DIR"
echo ""

HF_LEROBOT_HOME="$DATA_HOME/datasets" \
    "$LEROBOT_BIN" \
    --dataset.repo_id="$DATASET_REPO_ID" \
    --job_name="$EXP_NAME" \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}' \
    --policy.dtype=bfloat16 \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --peft.method_type=LORA \
    --steps=3000 \
    --batch_size=2 \
    --policy.device=cuda \
    --wandb.enable=true \
    --policy.push_to_hub=false \
    --output_dir="$CHECKPOINT_DIR"

echo ""
echo "Training finished."
echo "Checkpoints: $CHECKPOINT_DIR"
