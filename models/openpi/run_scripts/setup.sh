#!/usr/bin/env bash
# Set up the openpi environment used by the Quantycat SO-101 pi05 config.
#
# Usage from /home/caroline/Desktop/quantycat-positronic:
#   bash models/openpi/run_scripts/setup.sh
#
# This script does not patch openpi. It assumes the vendored OpenPI fork already
# contains the Quantycat config changes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OPENPI_REPO="${OPENPI_REPO:-$REPO/vendor/openpi}"

if [ ! -d "$OPENPI_REPO" ]; then
    echo "ERROR: openpi repo not found at: $OPENPI_REPO"
    echo "Clone or place openpi there before running setup."
    exit 1
fi

if command -v uv >/dev/null 2>&1; then
    UV_CMD=(uv)
elif python3 -m uv --version >/dev/null 2>&1; then
    UV_CMD=(python3 -m uv)
else
    echo "ERROR: uv is not installed or not runnable."
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
GIT_LFS_SKIP_SMUDGE=1 "${UV_CMD[@]}" sync

echo "Installing openpi editable package"
GIT_LFS_SKIP_SMUDGE=1 "${UV_CMD[@]}" pip install -e .

echo ""
echo "Setup complete."
echo "Next step:"
echo "  cd $REPO"
echo "  bash models/openpi/run_scripts/preprocess.sh"
