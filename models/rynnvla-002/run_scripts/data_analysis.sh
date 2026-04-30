#!/usr/bin/env bash
# Run all data analysis scripts against the current training data.
# CPU-only — no GPU or RynnVLA repo required.
#
# Answers:
#   - Are action normalization stats correct?         (action_stats_audit)
#   - How much motion is in each episode?             (action_episode_motion_report)
#   - Which episode windows are best for model eval?  (find_high_motion_windows)
#
# Usage:
#   source models/rynnvla-002/run_scripts/setup.sh
#   ./models/rynnvla-002/run_scripts/data_analysis.sh [--checkpoint <path>] [--top-k N]
#
# Outputs go to eval_output/screwdriver_so101/data_analysis/

set -e

if [ -z "$MODEL_ROOT" ] || [ -z "$PYTHON" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    MODEL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    if [ -z "$PYTHON" ]; then
        PYTHON="$(command -v python3)"
    fi
fi

if [ -z "$MODEL_ROOT" ] || [ -z "$PYTHON" ]; then
    echo "ERROR: source $MODEL_ROOT/run_scripts/setup.sh first, or ensure python3 is on PATH"
    exit 1
fi

TASK_DIR="my_data/training_pipeline/training_data/Put_the_screwdriver_into_the_cup"
CHECKPOINT=""
TOP_K=5

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --top-k)      TOP_K="$2";      shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

CKPT_ARG=""
[ -n "$CHECKPOINT" ] && CKPT_ARG="--checkpoint $CHECKPOINT"

REPO_ROOT="$(cd "$MODEL_ROOT/../.." && pwd)"
cd "$REPO_ROOT"

echo "--- Step 1: Action stats audit ---"
$PYTHON "$MODEL_ROOT/eval/data_analysis/action_stats_audit.py"

echo ""
echo "--- Step 2: Episode motion report ---"
$PYTHON "$MODEL_ROOT/eval/data_analysis/action_episode_motion_report.py"

echo ""
echo "--- Step 3: Low-motion distribution plot ---"
$PYTHON "$MODEL_ROOT/eval/data_analysis/plot_low_motion_distribution.py"

echo ""
echo "--- Step 4: High-motion eval windows ---"
$PYTHON "$MODEL_ROOT/eval/data_analysis/find_high_motion_windows.py" \
    --task-dir "$TASK_DIR" \
    --top-k "$TOP_K" \
    $CKPT_ARG

echo ""
echo "Data analysis complete."
echo "Motion report: eval_output/screwdriver_so101/data_analysis/action_motion_report/"
