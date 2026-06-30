#!/usr/bin/env bash
# Run v5 orange-wire holdout subgroup evals for selected 9999 checkpoints.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_HOME="${QUANTYCAT_DATA_HOME:-$HOME/quantycat-data}"

export CONFIG_NAME="${CONFIG_NAME:-pi05_dacha_wire_v5_h50_early_weighted}"
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

LOG_DIR="${LOG_DIR:-$DATA_HOME/logs/openpi/dacha}"
mkdir -p "$LOG_DIR"

CHECKPOINT_BASE="$DATA_HOME/checkpoints/openpi/dacha/$CONFIG_NAME"

run_eval() {
    local exp_name="$1"
    local label_prefix="$2"
    local subgroup="$3"
    local episodes="$4"
    local checkpoint="$CHECKPOINT_BASE/$exp_name/9999"
    local label="${label_prefix}_9999_holdout_${subgroup}"
    local log="$LOG_DIR/${label}.log"

    export EXP_NAME="$exp_name"
    export SPLIT="$subgroup"
    export EPISODES="$episodes"
    export LABEL="$label"

    "$SCRIPT_DIR/eval_dacha_high_motion.sh" "$checkpoint" 2>&1 | tee "$log"
}

run_checkpoint() {
    local exp_name="$1"
    local label_prefix="$2"

    run_eval "$exp_name" "$label_prefix" normal "60,61,62,63,64,65,66,67,68,69"
    run_eval "$exp_name" "$label_prefix" distal "88,89,90"
    run_eval "$exp_name" "$label_prefix" oddball "98,99"
}

run_checkpoint "06282026_pi05_dacha_wire_v5_h50_early_weighted" "dacha_wire_v5_h50_early_weighted"
run_checkpoint "06282026_pi05_dacha_wire_v5_h50_coverage_weighted" "dacha_wire_v5_h50_coverage_weighted"
