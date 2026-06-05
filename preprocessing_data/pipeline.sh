#!/usr/bin/env bash
# Run the full preprocessing pipeline in one command:
#   1. trim_dataset.py   — remove countdown hold + drop bad episodes
#   2. remove_pauses.py  — remove intra-episode stationary pauses
#   3. smooth_actions.py — smooth action jitter with Gaussian filter
#   4. (optional) convert to lerobot v3.0 format via lerobot's official converter
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
#   # Output as lerobot v3.0 (calls lerobot's official converter):
#   bash models/preprocessing/pipeline.sh \
#       --src my_data/input_data \
#       --dst my_data/clean_data \
#       --format v3
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
FORMAT="v2"
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
        --format)          FORMAT="$2";           shift 2 ;;
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
echo "  format:           $FORMAT"
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

if [[ "$FORMAT" == "v3" ]]; then
    LEROBOT_CONVERT="$(dirname "$($PYTHON -c 'import lerobot; print(lerobot.__file__)')")/scripts/convert_dataset_v21_to_v30.py"
    DST_NAME="$(basename "$DST")"
    echo "--- Step 4/4: convert to lerobot v3.0 ---"
    $PYTHON "$LEROBOT_CONVERT" \
        --repo-id="$DST_NAME" \
        --root="$DST" \
        --push-to-hub=false
    echo ""

    # The official converter omits camera keys from stats.json (pixels aren't in
    # parquet), but lerobot's factory.py requires them to exist before overwriting
    # with ImageNet stats. Add placeholder entries for all video-dtype features.
    $PYTHON - <<'PYEOF'
import json, sys
from pathlib import Path

dst = Path("$DST")
info = json.loads((dst / "meta/info.json").read_text())
stats_path = dst / "meta/stats.json"
stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

placeholder = {"min": [0.0], "max": [1.0], "mean": [0.0], "std": [1.0], "count": [0]}
added = []
for key, feat in info.get("features", {}).items():
    if feat.get("dtype") == "video" and key not in stats:
        stats[key] = placeholder
        added.append(key)

if added:
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Added camera placeholders to stats.json: {added}")
PYEOF
    echo ""
fi

echo "========================================="
echo "Done → $DST"
echo "========================================="
