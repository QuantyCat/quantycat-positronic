#!/usr/bin/env bash
# Fine-tune pi0.5 LoRA on Dacha v5 orange-wire data with 50-step horizon,
# mild motion-aware sampling, and early-horizon wrist/elbow loss emphasis.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONFIG_NAME="${CONFIG_NAME:-pi05_dacha_wire_v5_h50_early_weighted}"
export EXP_NAME="${EXP_NAME:-$(TZ=America/Los_Angeles date +%m%d%Y)_pi05_dacha_wire_v5_h50_early_weighted}"
export DATASET_ROOT="${DATASET_ROOT:-${QUANTYCAT_DATA_HOME:-$HOME/quantycat-data}/datasets/dacha/dacha_v5_train_openpi_v21}"

export TRAIN_EPISODES_JSON="${TRAIN_EPISODES_JSON:-[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84]}"
export HOLDOUT_EPISODES_JSON="${HOLDOUT_EPISODES_JSON:-[60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 88, 89, 90, 98, 99]}"
export TRAIN_LABEL="${TRAIN_LABEL:-renumbered train-only view, 85 episodes; source excludes 60-69, 88-90, 98-99}"
export HOLDOUT_LABEL="${HOLDOUT_LABEL:-source episodes 60-69, 88-90, 98-99}"

# Keep sampling mild; v5 intentionally contains more wrist-relevant demos.
export DACHA_WEIGHTED_MAX_EXTRA_REPEATS="${DACHA_WEIGHTED_MAX_EXTRA_REPEATS:-3}"
export DACHA_WEIGHTED_J3_DELTA_THRESHOLDS="${DACHA_WEIGHTED_J3_DELTA_THRESHOLDS:-3,6}"
export DACHA_WEIGHTED_J4_DELTA_THRESHOLDS="${DACHA_WEIGHTED_J4_DELTA_THRESHOLDS:-2,4}"
export DACHA_WEIGHTED_J3_VEL_THRESHOLDS="${DACHA_WEIGHTED_J3_VEL_THRESHOLDS:-3}"
export DACHA_WEIGHTED_J4_VEL_THRESHOLDS="${DACHA_WEIGHTED_J4_VEL_THRESHOLDS:-2}"

export DACHA_JOINT_LOSS_WEIGHTS="${DACHA_JOINT_LOSS_WEIGHTS:-1,1,1,1,1,1}"
export DACHA_EARLY_HORIZON_END="${DACHA_EARLY_HORIZON_END:-10}"
export DACHA_EARLY_HORIZON_WEIGHT="${DACHA_EARLY_HORIZON_WEIGHT:-2}"
export DACHA_EARLY_JOINTS="${DACHA_EARLY_JOINTS:-3,4}"
export DACHA_EARLY_JOINT_MULTIPLIERS="${DACHA_EARLY_JOINT_MULTIPLIERS:-2,3}"

"$SCRIPT_DIR/train_dacha_weighted.sh"

CHECKPOINT_DIR="${QUANTYCAT_DATA_HOME:-$HOME/quantycat-data}/checkpoints/openpi/dacha/$CONFIG_NAME/$EXP_NAME"
for step in 5000 7500; do
    if [ -d "$CHECKPOINT_DIR/$step" ]; then
        rm -rf "$CHECKPOINT_DIR/$step"
    fi
done

echo ""
echo "Retained checkpoints:"
find "$CHECKPOINT_DIR" -mindepth 1 -maxdepth 1 -type d -printf '  %f\n' | sort -n
