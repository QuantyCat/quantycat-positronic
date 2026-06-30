#!/usr/bin/env bash
# Run v5 orange-wire evals for retained checkpoints against holdout and train splits.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_HOME="${QUANTYCAT_DATA_HOME:-$HOME/quantycat-data}"

export CONFIG_NAME="${CONFIG_NAME:-pi05_dacha_wire_v5_h50_early_weighted}"
export EXP_NAME="${EXP_NAME:-06282026_pi05_dacha_wire_v5_h50_early_weighted}"
export DATASET_ROOT="${DATASET_ROOT:-$DATA_HOME/datasets/dacha/dacha_v5_openpi_v21}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_HOME/eval_output/openpi/dacha/pi05_wire_v5}"
export DACHA_EVAL_PROMPT="${DACHA_EVAL_PROMPT:-pick up the orange wire and put it in the cup}"
export ACTION_HORIZON="${ACTION_HORIZON:-50}"
export WINDOW_SIZE="${WINDOW_SIZE:-50}"
export SAMPLE_STEPS="${SAMPLE_STEPS:-10}"
export TOP_K="${TOP_K:-5}"
export JOINTS="${JOINTS:-0,1,2,3,4}"
export SAVE_TRACES="${SAVE_TRACES:-1}"
export RUN_COUPLING_DIAGNOSTIC="${RUN_COUPLING_DIAGNOSTIC:-1}"
export RUN_LAG_SWEEP_DIAGNOSTIC="${RUN_LAG_SWEEP_DIAGNOSTIC:-1}"
export RUN_EXECUTION_INDEX_SWEEP_DIAGNOSTIC="${RUN_EXECUTION_INDEX_SWEEP_DIAGNOSTIC:-1}"
export LABEL_PREFIX="${LABEL_PREFIX:-dacha_wire_v5_h50_early_weighted}"
export LOG_PREFIX="${LOG_PREFIX:-06282026_pi05_dacha_wire_v5_h50_early_weighted}"

LOG_DIR="${LOG_DIR:-$DATA_HOME/logs/openpi/dacha}"
mkdir -p "$LOG_DIR"

CHECKPOINT_ROOT="$DATA_HOME/checkpoints/openpi/dacha/$CONFIG_NAME/$EXP_NAME"
HOLDOUT_EPISODES="${HOLDOUT_EPISODES:-60,61,62,63,64,65,66,67,68,69,88,89,90,98,99}"
TRAIN_EPISODES="${TRAIN_EPISODES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,91,92,93,94,95,96,97}"
EVAL_STEPS="${EVAL_STEPS:-2500,9999}"

run_eval() {
    local step="$1"
    local split="$2"
    local episodes="$3"
    local label="${LABEL_PREFIX}_${step}_${split}"
    local log="$LOG_DIR/${LOG_PREFIX}_${step}_eval_${split}.log"

    export SPLIT="$split"
    export EPISODES="$episodes"
    export LABEL="$label"

    "$SCRIPT_DIR/eval_dacha_high_motion.sh" "$CHECKPOINT_ROOT/$step" 2>&1 | tee "$log"
}

IFS=',' read -r -a steps <<<"$EVAL_STEPS"
for step in "${steps[@]}"; do
    run_eval "$step" holdout "$HOLDOUT_EPISODES"
    run_eval "$step" train "$TRAIN_EPISODES"
done
