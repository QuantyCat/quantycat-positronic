#!/usr/bin/env bash
# Compute quantile norm stats for the Quantycat SO-101 screwdriver dataset.
#
# Runs augment_dataset_quantile_stats on screwdriver_so101_clean_v3 and writes
# q01/q99 into its meta/stats.json. Both the LeRobot and OpenPI training
# pipelines read from that file, so this is the only preprocessing step needed.
#
# Usage from the repo root:
#   bash models/lerobot/run_scripts/preprocess.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [ -f "$REPO/.env" ]; then
    set -a && source "$REPO/.env" && set +a
fi
LEROBOT_VENV="${LEROBOT_VENV:-$REPO/.venvs/lerobot}"
DATA_HOME="${QUANTYCAT_DATA_HOME:-$HOME/quantycat-data}"
DATASET_NAME="${DATASET_NAME:-screwdriver_so101_clean_v3}"
HF_REPO_ID="${HF_REPO_ID:-cshourab/screwdriver_so101_clean_v3}"
DATASET_PATH="$DATA_HOME/datasets/$DATASET_NAME"

if [ ! -d "$LEROBOT_VENV" ]; then
    echo "ERROR: lerobot venv not found at: $LEROBOT_VENV"
    echo "Run setup first:"
    echo "  bash models/lerobot/run_scripts/setup.sh"
    exit 1
fi

if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: dataset not found at: $DATASET_PATH"
    exit 1
fi

PYTHON="$LEROBOT_VENV/bin/python"
AUGMENT_SCRIPT="$("$PYTHON" -c 'import lerobot; import pathlib; print(pathlib.Path(lerobot.__file__).parent / "scripts/augment_dataset_quantile_stats.py")')"

echo "Computing quantile norm stats"
echo "  dataset: $HF_REPO_ID"
echo "  path:    $DATASET_PATH"
echo ""

HF_TOKEN="$HF_TOKEN_WRITE" \
    "$PYTHON" "$AUGMENT_SCRIPT" \
    --repo-id="$HF_REPO_ID" \
    --root="$DATASET_PATH"

echo ""
echo "Done. Stats written to:"
echo "  $DATASET_PATH/meta/stats.json"
