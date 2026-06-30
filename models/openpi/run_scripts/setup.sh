#!/usr/bin/env bash
# Clone (or update) upstream openpi, apply Quantycat patches, and install the
# environment at .venvs/openpi.
#
# Usage from the repo root:
#   bash models/openpi/run_scripts/setup.sh
#
# Run this again whenever:
#   - Setting up a new machine.
#   - Pulling a newer upstream openpi version (OPENPI_REF=<tag>).
#   - The patch files in vendor_patches/ have changed.
#
# Environment variables:
#   OPENPI_UPSTREAM   Official openpi git URL  (default: https://github.com/Physical-Intelligence/openpi)
#   OPENPI_REF        Branch, tag, or commit to pin  (default: main)
#   OPENPI_REPO       Where to clone  (default: vendor/openpi)
#   OPENPI_VENV       Where to create the venv  (default: .venvs/openpi)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OPENPI_REPO="${OPENPI_REPO:-$REPO/vendor/openpi}"
OPENPI_UPSTREAM="${OPENPI_UPSTREAM:-https://github.com/Physical-Intelligence/openpi}"
OPENPI_REF="${OPENPI_REF:-main}"
OPENPI_VENV="${OPENPI_VENV:-$REPO/.venvs/openpi}"
PATCHES_DIR="$REPO/models/openpi/vendor_patches"

# ── 1. Clone or update upstream openpi ───────────────────────────────────────
if [ ! -d "$OPENPI_REPO/.git" ]; then
    echo "Cloning openpi ($OPENPI_REF) from $OPENPI_UPSTREAM"
    git clone --branch "$OPENPI_REF" --depth 1 "$OPENPI_UPSTREAM" "$OPENPI_REPO"
else
    echo "Updating openpi at $OPENPI_REPO"
    git -C "$OPENPI_REPO" fetch origin "$OPENPI_REF"
    git -C "$OPENPI_REPO" reset --hard FETCH_HEAD
fi

# ── 2. Apply Quantycat patches ────────────────────────────────────────────────
echo "Applying Quantycat patches"
cp "$PATCHES_DIR/src/openpi/training/config.py"           "$OPENPI_REPO/src/openpi/training/config.py"
cp "$PATCHES_DIR/src/quantycat_training_config.py"        "$OPENPI_REPO/src/quantycat_training_config.py"
mkdir -p "$OPENPI_REPO/src/openpi/policies"
cp "$PATCHES_DIR/src/openpi/policies/quantycat_policy.py" "$OPENPI_REPO/src/openpi/policies/quantycat_policy.py"

# ── 3. Install dependencies into .venvs/openpi ────────────────────────────────
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

mkdir -p "$REPO/.venvs"
export UV_PROJECT_ENVIRONMENT="$OPENPI_VENV"

cd "$OPENPI_REPO"

echo "Syncing openpi dependencies into $OPENPI_VENV"
GIT_LFS_SKIP_SMUDGE=1 "${UV_CMD[@]}" sync

echo ""
echo "Setup complete. openpi venv at: $OPENPI_VENV"
echo "Next step:"
echo "  cd $REPO"
echo "  bash models/openpi/run_scripts/preprocess.sh"
