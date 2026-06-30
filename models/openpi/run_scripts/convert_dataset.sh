#!/usr/bin/env bash
# Convert a raw LeRobot v3.0 recording into the per-episode v2.1 layout that
# OpenPI expects. Run this once for every newly captured dataset before
# training on it with OpenPI.
#
# Usage from the repo root:
#   bash models/openpi/run_scripts/convert_dataset.sh --source <raw_v3_dataset> --target <output_v21_dataset>
#
# Any extra flags are forwarded to convert_lerobot_v21.py, e.g.:
#   bash models/openpi/run_scripts/convert_dataset.sh \
#       --source ~/quantycat-data/datasets/my_new_dataset \
#       --target ~/quantycat-data/datasets/my_new_dataset_openpi_v21 \
#       --episodes "0-59,70-87" --renumber --overwrite

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OPENPI_VENV="${OPENPI_VENV:-$REPO/.venvs/openpi}"
PYTHON="$OPENPI_VENV/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: openpi venv not found at: $OPENPI_VENV"
    echo "Run setup first:"
    echo "  bash models/openpi/run_scripts/setup.sh"
    exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ERROR: ffmpeg is not installed or not on PATH."
    exit 1
fi

"$PYTHON" "$SCRIPT_DIR/convert_lerobot_v21.py" "$@"
