#!/usr/bin/env bash
# Evaluate the v5 run trained on source episodes 16-99 and held out on 1-15.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONFIG_NAME="${CONFIG_NAME:-pi05_dacha_wire_v5_h50_early_weighted_ep16_99}"
export EXP_NAME="${EXP_NAME:-$(TZ=America/Los_Angeles date +%m%d%Y)_pi05_dacha_wire_v5_h50_early_weighted_ep16_99}"
export DATASET_ROOT="${DATASET_ROOT:-${QUANTYCAT_DATA_HOME:-$HOME/quantycat-data}/datasets/dacha/dacha_v5_openpi_v21}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${QUANTYCAT_DATA_HOME:-$HOME/quantycat-data}/eval_output/openpi/dacha/pi05_wire_v5_ep16_99}"
export LABEL_PREFIX="${LABEL_PREFIX:-dacha_wire_v5_h50_early_weighted_ep16_99}"
export LOG_PREFIX="${LOG_PREFIX:-$EXP_NAME}"

export HOLDOUT_EPISODES="${HOLDOUT_EPISODES:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export TRAIN_EPISODES="${TRAIN_EPISODES:-16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99}"
export EVAL_STEPS="${EVAL_STEPS:-2500,9999}"

"$SCRIPT_DIR/eval_dacha_wire_v5_checkpoints.sh"
