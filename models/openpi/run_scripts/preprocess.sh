#!/usr/bin/env bash
# Compute openpi normalization statistics for the Quantycat SO-101 dataset.
#
# Usage from /home/caroline/quantycat-positronic:
#   bash models/openpi/run_scripts/preprocess.sh

set -euo pipefail

OPENPI_REPO="${OPENPI_REPO:-/home/caroline/openpi}"
NORM_STATS_PATH="/home/caroline/quantycat-positronic/my_data/training_pipeline/openpi/norm_stats.json"

if [ ! -d "$OPENPI_REPO" ]; then
    echo "ERROR: openpi repo not found at: $OPENPI_REPO"
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is not installed or not on PATH. Run setup first:"
    echo "  bash models/openpi/run_scripts/setup.sh"
    exit 1
fi

cd "$OPENPI_REPO"

echo "Computing norm stats for pi05_quantycat"
uv run scripts/compute_norm_stats.py --config-name pi05_quantycat

if [ ! -f "$NORM_STATS_PATH" ]; then
    echo "ERROR: expected norm stats not found at:"
    echo "  $NORM_STATS_PATH"
    exit 1
fi

echo ""
echo "Norm stats ready:"
echo "  $NORM_STATS_PATH"
