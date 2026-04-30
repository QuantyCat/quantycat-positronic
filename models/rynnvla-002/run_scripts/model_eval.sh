#!/usr/bin/env bash
# Evaluate the model on one or more training episodes and save results.
#
# Usage:
#   source models/rynnvla-002/run_scripts/setup.sh
#   ./models/rynnvla-002/run_scripts/model_eval.sh \
#       --episode my_data/training_pipeline/training_data/.../episode_000025 \
#       --episode my_data/training_pipeline/training_data/.../episode_000000 \
#       [--checkpoint /path/to/checkpoint] \
#       [--start-step 100] [--max-steps 50]
#
# Checkpoint falls back to config.yaml if not passed.
# Set RYNNVLA_REPO env var or pass --rynnvla-repo if the solver is not on PYTHONPATH.
# Outputs (JSON + plots) go to training_output/screwdriver_so101/model_eval_reports/

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

EPISODES=()
CHECKPOINT=""
START_STEP=0
MAX_STEPS=50
RYNNVLA_REPO="${RYNNVLA_REPO:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --episode)      EPISODES+=("$2");    shift 2 ;;
        --checkpoint)   CHECKPOINT="$2";     shift 2 ;;
        --start-step)   START_STEP="$2";     shift 2 ;;
        --max-steps)    MAX_STEPS="$2";      shift 2 ;;
        --rynnvla-repo) RYNNVLA_REPO="$2";   shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ ${#EPISODES[@]} -eq 0 ]; then
    echo "Usage: $0 --episode <path> [--episode <path> ...] [--checkpoint <path>] [--start-step N] [--max-steps N]"
    exit 1
fi

CKPT_ARG=""
[ -n "$CHECKPOINT" ] && CKPT_ARG="--checkpoint $CHECKPOINT"

RYNNVLA_ARG=""
[ -n "$RYNNVLA_REPO" ] && RYNNVLA_ARG="--rynnvla-repo $RYNNVLA_REPO"

REPO_ROOT="$(cd "$MODEL_ROOT/../.." && pwd)"
cd "$REPO_ROOT"

for EPISODE in "${EPISODES[@]}"; do
    echo ""
    echo "--- Evaluating: $(basename "$EPISODE") ---"
    $PYTHON "$MODEL_ROOT/eval/model_eval/episode_batch_eval.py" \
        --episode "$EPISODE" \
        --positronic-config "$MODEL_ROOT/config.yaml" \
        --start-step "$START_STEP" \
        --max-steps "$MAX_STEPS" \
        --save-json \
        --save-plots \
        $CKPT_ARG \
        $RYNNVLA_ARG
done

echo ""
echo "Model eval complete."
echo "Reports saved to eval_output/screwdriver_so101/model_eval/<checkpoint>/<episode>/"
