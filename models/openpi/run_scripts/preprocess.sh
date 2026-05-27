#!/usr/bin/env bash
# Compute openpi normalization statistics for the Quantycat SO-101 dataset.
#
# Usage from the repo root:
#   bash models/openpi/run_scripts/preprocess.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OPENPI_REPO="${OPENPI_REPO:-$REPO/vendor/openpi}"
DATA_HOME="${QUANTYCAT_DATA_HOME:-$HOME/quantycat-data}"
CONFIG_NAME="${CONFIG_NAME:-pi05_quantycat_lora}"
NORM_STATS_DIR="$DATA_HOME/norm_stats/openpi/${CONFIG_NAME}"

if [ ! -d "$OPENPI_REPO" ]; then
    echo "ERROR: openpi repo not found at: $OPENPI_REPO"
    exit 1
fi

if command -v uv >/dev/null 2>&1; then
    UV_CMD=(uv)
elif python3 -m uv --version >/dev/null 2>&1; then
    UV_CMD=(python3 -m uv)
else
    echo "ERROR: uv is not installed or not runnable. Run setup first:"
    echo "  bash models/openpi/run_scripts/setup.sh"
    exit 1
fi

cd "$OPENPI_REPO"

export PYTHONPATH="$REPO/models/openpi/training_config${PYTHONPATH:+:$PYTHONPATH}"
export HF_LEROBOT_HOME="$DATA_HOME/datasets"

echo "Computing norm stats for $CONFIG_NAME"
"${UV_CMD[@]}" run scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"

if [ -z "$(find "$NORM_STATS_DIR" -name "norm_stats.json" 2>/dev/null)" ]; then
    echo "ERROR: expected norm stats not found under:"
    echo "  $NORM_STATS_DIR"
    exit 1
fi

echo ""
echo "Norm stats ready:"
echo "  $NORM_STATS_DIR"
