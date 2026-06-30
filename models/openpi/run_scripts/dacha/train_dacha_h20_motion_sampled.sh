#!/usr/bin/env bash
# Fine-tune pi0.5 LoRA on Dacha with 20-step action horizon, all-ones loss,
# and mild motion-aware sampling.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONFIG_NAME="${CONFIG_NAME:-pi05_dacha_lora_h20_motion_sampled}"
export EXP_NAME="${EXP_NAME:-$(TZ=America/Los_Angeles date +%m%d%Y)_pi05_dacha_lora_h20_motion_sampled}"

# Keep the loss geometry neutral; this run isolates shorter action horizon plus
# modest resampling of sharper wrist/forearm moments.
export DACHA_JOINT_LOSS_WEIGHTS="${DACHA_JOINT_LOSS_WEIGHTS:-1,1,1,1,1,1}"
export DACHA_WEIGHTED_MAX_EXTRA_REPEATS="${DACHA_WEIGHTED_MAX_EXTRA_REPEATS:-3}"
export DACHA_WEIGHTED_J3_DELTA_THRESHOLDS="${DACHA_WEIGHTED_J3_DELTA_THRESHOLDS:-3,6}"
export DACHA_WEIGHTED_J4_DELTA_THRESHOLDS="${DACHA_WEIGHTED_J4_DELTA_THRESHOLDS:-2,4}"
export DACHA_WEIGHTED_J3_VEL_THRESHOLDS="${DACHA_WEIGHTED_J3_VEL_THRESHOLDS:-3}"
export DACHA_WEIGHTED_J4_VEL_THRESHOLDS="${DACHA_WEIGHTED_J4_VEL_THRESHOLDS:-2}"

exec "$SCRIPT_DIR/train_dacha_weighted.sh"
