#!/usr/bin/env bash
# Fine-tune pi0.5 LoRA on Dacha for 2500 steps with 50-step action horizon,
# mild motion-aware sampling, and early-horizon loss emphasis.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONFIG_NAME="${CONFIG_NAME:-pi05_dacha_lora_h50_early_weighted_2500}"
export EXP_NAME="${EXP_NAME:-$(TZ=America/Los_Angeles date +%m%d%Y)_pi05_dacha_lora_h50_early_weighted_2500}"

# Keep dataset sampling comparable to the h50 mild-sampling baseline.
export DACHA_WEIGHTED_MAX_EXTRA_REPEATS="${DACHA_WEIGHTED_MAX_EXTRA_REPEATS:-3}"
export DACHA_WEIGHTED_J3_DELTA_THRESHOLDS="${DACHA_WEIGHTED_J3_DELTA_THRESHOLDS:-3,6}"
export DACHA_WEIGHTED_J4_DELTA_THRESHOLDS="${DACHA_WEIGHTED_J4_DELTA_THRESHOLDS:-2,4}"
export DACHA_WEIGHTED_J3_VEL_THRESHOLDS="${DACHA_WEIGHTED_J3_VEL_THRESHOLDS:-3}"
export DACHA_WEIGHTED_J4_VEL_THRESHOLDS="${DACHA_WEIGHTED_J4_VEL_THRESHOLDS:-2}"

# Base joint weights stay flat; joint-specific emphasis applies only to early
# horizon positions so we do not globally overfit j3/j4 amplitude.
export DACHA_JOINT_LOSS_WEIGHTS="${DACHA_JOINT_LOSS_WEIGHTS:-1,1,1,1,1,1}"
export DACHA_EARLY_HORIZON_END="${DACHA_EARLY_HORIZON_END:-10}"
export DACHA_EARLY_HORIZON_WEIGHT="${DACHA_EARLY_HORIZON_WEIGHT:-2}"
export DACHA_EARLY_JOINTS="${DACHA_EARLY_JOINTS:-3,4}"
export DACHA_EARLY_JOINT_MULTIPLIERS="${DACHA_EARLY_JOINT_MULTIPLIERS:-2,3}"

"$SCRIPT_DIR/train_dacha_weighted.sh"

CHECKPOINT_DIR="${QUANTYCAT_DATA_HOME:-$HOME/quantycat-data}/checkpoints/openpi/dacha/$CONFIG_NAME/$EXP_NAME"

echo ""
echo "Retained checkpoints:"
find "$CHECKPOINT_DIR" -mindepth 1 -maxdepth 1 -type d -printf '  %f\n' | sort -n
