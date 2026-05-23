#!/usr/bin/env bash
# Run the full preprocessing pipeline in one command:
#   1. trim_dataset.py   — remove countdown hold + drop bad episodes
#   2. remove_pauses.py  — remove intra-episode stationary pauses
#   3. smooth_actions.py — smooth action jitter with Gaussian filter
#
# Intermediate datasets are written to a temp directory and cleaned up.
# Only the final cleaned dataset is kept at --dst.
#
# Usage (from repo root):
#   bash models/preprocessing/pipeline.sh \
#       --src my_data/input_data \
#       --dst my_data/clean_data \
#       --trim-frames 165 \
#       --remove-episodes "45 12" \
#       --sigma 1.5
#
# All arguments except --src and --dst are optional.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-conda run -n rynnvla002 python3}"

# Defaults
SRC=""
DST=""
TRIM_FRAMES=0
REMOVE_EPISODES=""
SPEED_THRESHOLD=0.01
MIN_PAUSE_FRAMES=15
SIGMA=1.5
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --src)             SRC="$2";              shift 2 ;;
        --dst)             DST="$2";              shift 2 ;;
        --trim-frames)     TRIM_FRAMES="$2";      shift 2 ;;
        --remove-episodes) REMOVE_EPISODES="$2";  shift 2 ;;
        --speed-threshold) SPEED_THRESHOLD="$2";  shift 2 ;;
        --min-pause-frames) MIN_PAUSE_FRAMES="$2"; shift 2 ;;
        --sigma)           SIGMA="$2";            shift 2 ;;
        --dry-run)         DRY_RUN=true;          shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$SRC" || -z "$DST" ]]; then
    echo "Usage: bash pipeline.sh --src <input> --dst <output> [options]"
    echo ""
    echo "Options:"
    echo "  --trim-frames N          Frames to cut from start of every episode (default: 0)"
    echo "  --remove-episodes '1 2'  Space-separated episode indices to drop (default: none)"
    echo "  --speed-threshold F      Pause detection speed threshold (default: 0.01 rad/frame)"
    echo "  --min-pause-frames N     Minimum pause length to remove (default: 15 frames)"
    echo "  --sigma F                Gaussian smoothing sigma in frames (default: 1.5)"
    echo "  --dry-run                Preview without writing"
    exit 1
fi

TMPDIR="${DST}__tmp"
STEP1="${TMPDIR}/step1_trimmed"
STEP2="${TMPDIR}/step2_nopause"

cleanup() {
    if [[ -d "$TMPDIR" ]]; then
        echo "Cleaning up temp directory..."
        rm -rf "$TMPDIR"
    fi
}
trap cleanup EXIT

echo "========================================="
echo "Preprocessing pipeline"
echo "  src:              $SRC"
echo "  dst:              $DST"
echo "  trim-frames:      $TRIM_FRAMES"
echo "  remove-episodes:  ${REMOVE_EPISODES:-none}"
echo "  speed-threshold:  $SPEED_THRESHOLD rad/frame"
echo "  min-pause-frames: $MIN_PAUSE_FRAMES"
echo "  sigma:            $SIGMA"
echo "  dry-run:          $DRY_RUN"
echo "========================================="
echo ""

if [[ "$DRY_RUN" == true ]]; then
    echo "--- Step 1: trim_dataset (dry run) ---"
    TRIM_ARGS=(--src "$SRC" --dst "$STEP1" --trim-frames "$TRIM_FRAMES" --dry-run)
    [[ -n "$REMOVE_EPISODES" ]] && TRIM_ARGS+=(--remove-episodes $REMOVE_EPISODES)
    $PYTHON "$SCRIPT_DIR/trim_dataset.py" "${TRIM_ARGS[@]}"
    echo ""
    echo "--- Step 2: remove_pauses (dry run) ---"
    $PYTHON "$SCRIPT_DIR/remove_pauses.py" \
        --src "$SRC" --dst "$STEP2" \
        --speed-threshold "$SPEED_THRESHOLD" \
        --min-pause-frames "$MIN_PAUSE_FRAMES" \
        --dry-run
    echo ""
    echo "--- Step 3: smooth_actions (dry run) ---"
    $PYTHON "$SCRIPT_DIR/smooth_actions.py" \
        --src "$SRC" --dst "$DST" \
        --sigma "$SIGMA" \
        --dry-run
    exit 0
fi

# Step 1: trim
echo "--- Step 1/3: trim_dataset ---"
TRIM_ARGS=(--src "$SRC" --dst "$STEP1" --trim-frames "$TRIM_FRAMES")
[[ -n "$REMOVE_EPISODES" ]] && TRIM_ARGS+=(--remove-episodes $REMOVE_EPISODES)
$PYTHON "$SCRIPT_DIR/trim_dataset.py" "${TRIM_ARGS[@]}"
echo ""

# Step 2: remove pauses
echo "--- Step 2/3: remove_pauses ---"
$PYTHON "$SCRIPT_DIR/remove_pauses.py" \
    --src "$STEP1" --dst "$STEP2" \
    --speed-threshold "$SPEED_THRESHOLD" \
    --min-pause-frames "$MIN_PAUSE_FRAMES"
echo ""

# Step 3: smooth actions
echo "--- Step 3/3: smooth_actions ---"
$PYTHON "$SCRIPT_DIR/smooth_actions.py" \
    --src "$STEP2" --dst "$DST" \
    --sigma "$SIGMA"
echo ""

echo "========================================="
echo "Done → $DST"
echo "========================================="
