#!/usr/bin/env bash
# Fine-tune pi0.5 LoRA on Dacha with motion-aware sampling and joint-weighted loss.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
OPENPI_REPO="${OPENPI_REPO:-$REPO/vendor/openpi}"
PATCH_SRC="$REPO/models/openpi/vendor_patches/src"
TRAINING_DIR="$REPO/models/openpi/training"
CONFIG_NAME="${CONFIG_NAME:-pi05_dacha_lora_motion_weighted}"
EXP_NAME="${EXP_NAME:-$(TZ=America/Los_Angeles date +%m%d%Y)_pi05_dacha_lora_motion_weighted}"
DATA_HOME="${QUANTYCAT_DATA_HOME:-$HOME/quantycat-data}"
DATASET_ROOT="${DATASET_ROOT:-$DATA_HOME/datasets/dacha/dacha_v3_openpi_v21}"
CHECKPOINT_DIR="$DATA_HOME/checkpoints/openpi/dacha/$CONFIG_NAME/$EXP_NAME"
LOG_DIR="${LOG_DIR:-$DATA_HOME/logs/openpi/dacha}"
LOG_PATH="$LOG_DIR/${EXP_NAME}.log"
SPLIT_PATH="$LOG_DIR/${EXP_NAME}_split.json"
TRAIN_EPISODES_JSON="${TRAIN_EPISODES_JSON:-[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]}"
HOLDOUT_EPISODES_JSON="${HOLDOUT_EPISODES_JSON:-[40, 41, 42, 43, 44, 45, 46, 47, 48, 49]}"
TRAIN_LABEL="${TRAIN_LABEL:-episodes 0-39}"
HOLDOUT_LABEL="${HOLDOUT_LABEL:-episodes 40-49}"

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
  "train_episodes": $TRAIN_EPISODES_JSON,
  "holdout_episodes": $HOLDOUT_EPISODES_JSON,
  "motion_sampling": {
    "enabled": "${DACHA_WEIGHTED_SAMPLING:-1}",
    "j3_delta_thresholds": "${DACHA_WEIGHTED_J3_DELTA_THRESHOLDS:-2,4,6}",
    "j4_delta_thresholds": "${DACHA_WEIGHTED_J4_DELTA_THRESHOLDS:-1,2,3}",
    "j3_velocity_thresholds": "${DACHA_WEIGHTED_J3_VEL_THRESHOLDS:-2,4}",
    "j4_velocity_thresholds": "${DACHA_WEIGHTED_J4_VEL_THRESHOLDS:-1,2}",
    "max_extra_repeats": "${DACHA_WEIGHTED_MAX_EXTRA_REPEATS:-6}"
  },
  "joint_loss_weights": "${DACHA_JOINT_LOSS_WEIGHTS:-1,1,1,2,3,1}",
  "early_horizon_loss_weights": {
    "end": "${DACHA_EARLY_HORIZON_END:--1}",
    "horizon_weight": "${DACHA_EARLY_HORIZON_WEIGHT:-1}",
    "joints": "${DACHA_EARLY_JOINTS:-3,4}",
    "joint_multipliers": "${DACHA_EARLY_JOINT_MULTIPLIERS:-1,1}"
  }
}
EOF

cd "$OPENPI_REPO"

export OPENPI_REPO
export QUANTYCAT_POSITRONIC_REPO="$REPO"
export QUANTYCAT_DATA_HOME="$DATA_HOME"
export HF_LEROBOT_HOME="$DATA_HOME/datasets"
export DACHA_WEIGHTED_DATASET_ROOT="$DATASET_ROOT"
export PYTHONPATH="$PATCH_SRC:$OPENPI_REPO/src:${PYTHONPATH:-}"

echo "Starting weighted OpenPI Dacha training"
echo "  config:       $CONFIG_NAME"
echo "  exp name:     $EXP_NAME"
echo "  dataset:      $DATASET_ROOT"
echo "  train:        $TRAIN_LABEL"
echo "  holdout:      $HOLDOUT_LABEL"
echo "  checkpoints:  $CHECKPOINT_DIR"
echo "  split record: $SPLIT_PATH"
echo "  log:          $LOG_PATH"
echo "  joint weights:${DACHA_JOINT_LOSS_WEIGHTS:-1,1,1,2,3,1}"
echo "  early loss:   h0-${DACHA_EARLY_HORIZON_END:--1} x${DACHA_EARLY_HORIZON_WEIGHT:-1}; joints ${DACHA_EARLY_JOINTS:-3,4} x${DACHA_EARLY_JOINT_MULTIPLIERS:-1,1}"
echo ""

XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
    "${UV_CMD[@]}" run python "$TRAINING_DIR/dacha_weighted_launcher.py" \
        "$CONFIG_NAME" --exp-name="$EXP_NAME" --overwrite 2>&1 | tee "$LOG_PATH"

echo ""
echo "Weighted training command finished."
echo "Checkpoints:"
echo "  $CHECKPOINT_DIR"
