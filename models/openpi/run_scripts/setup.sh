#!/usr/bin/env bash
# Set up the openpi environment used by the Quantycat SO-101 pi05 config.
#
# Usage from /home/caroline/quantycat-positronic:
#   bash models/openpi/run_scripts/setup.sh
#
# This script does not patch openpi. It assumes /home/caroline/openpi already
# contains the Quantycat config changes.

set -euo pipefail

OPENPI_REPO="${OPENPI_REPO:-/home/caroline/openpi}"

if [ ! -d "$OPENPI_REPO" ]; then
    echo "ERROR: openpi repo not found at: $OPENPI_REPO"
    echo "Clone or place openpi there before running setup."
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is not installed or not on PATH."
    echo ""
    echo "Install uv with one of:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  source \$HOME/.local/bin/env"
    echo ""
    echo "or:"
    echo "  python -m pip install uv"
    echo ""
    echo "Then rerun this script."
    exit 1
fi

cd "$OPENPI_REPO"

echo "Syncing openpi dependencies in $OPENPI_REPO"
GIT_LFS_SKIP_SMUDGE=1 uv sync

echo "Installing openpi editable package"
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

echo ""
echo "Setup complete."
echo "Next step:"
echo "  cd /home/caroline/quantycat-positronic"
echo "  bash models/openpi/run_scripts/preprocess.sh"
