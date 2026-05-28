#!/usr/bin/env bash
# Install the lerobot environment for pi05 fine-tuning on the Quantycat SO-101 dataset.
#
# Usage from the repo root:
#   bash models/lerobot/run_scripts/setup.sh
#
# Environment variables:
#   LEROBOT_VENV   Where to create the venv  (default: vendor/lerobot/.venv)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LEROBOT_VENV="${LEROBOT_VENV:-$REPO/vendor/lerobot/.venv}"

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

echo "Creating lerobot venv at: $LEROBOT_VENV"
mkdir -p "$(dirname "$LEROBOT_VENV")"
"${UV_CMD[@]}" venv "$LEROBOT_VENV" --python 3.12

echo "Installing lerobot[pi]"
"${UV_CMD[@]}" pip install --python "$LEROBOT_VENV" "lerobot[pi]==0.5.1" peft

echo ""
echo "Setup complete. lerobot venv at: $LEROBOT_VENV"
echo "Next step:"
echo "  cd $REPO"
echo "  bash models/lerobot/run_scripts/training.sh"
