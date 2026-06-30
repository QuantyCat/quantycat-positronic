#!/usr/bin/env bash
# Fine-tune pi0.5 LoRA on Dacha with 50-step action horizon, all-ones loss,
# and mild motion-aware sampling.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONFIG_NAME="${CONFIG_NAME:-pi05_dacha_lora_h50_motion_sampled}"
export EXP_NAME="${EXP_NAME:-$(TZ=America/Los_Angeles date +%m%d%Y)_pi05_dacha_lora_h50_motion_sampled}"

# Keep this comparable to train_dacha_h20_motion_sampled.sh except for horizon.
export DACHA_JOINT_LOSS_WEIGHTS="${DACHA_JOINT_LOSS_WEIGHTS:-1,1,1,1,1,1}"
export DACHA_WEIGHTED_MAX_EXTRA_REPEATS="${DACHA_WEIGHTED_MAX_EXTRA_REPEATS:-3}"
export DACHA_WEIGHTED_J3_DELTA_THRESHOLDS="${DACHA_WEIGHTED_J3_DELTA_THRESHOLDS:-3,6}"
export DACHA_WEIGHTED_J4_DELTA_THRESHOLDS="${DACHA_WEIGHTED_J4_DELTA_THRESHOLDS:-2,4}"
export DACHA_WEIGHTED_J3_VEL_THRESHOLDS="${DACHA_WEIGHTED_J3_VEL_THRESHOLDS:-3}"
export DACHA_WEIGHTED_J4_VEL_THRESHOLDS="${DACHA_WEIGHTED_J4_VEL_THRESHOLDS:-2}"

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
