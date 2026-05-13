#!/usr/bin/env bash
# Set up the openpi environment for the SO-101 pipeline.
#
# What this does:
#   1. Verifies ~/openpi exists.
#   2. Creates (or reuses) a uv-managed venv inside ~/openpi.
#   3. Installs openpi and its dependencies via `uv sync`.
#   4. Symlinks so101_policy.py into openpi's policies package so that
#      the training infrastructure can import it.
#   5. Exports $PYTHON for use by the other run_scripts.
#
# Usage:
#   source models/openpi/run_scripts/setup.sh
#
# After sourcing, run:
#   bash models/openpi/run_scripts/preprocess.sh
#   bash models/openpi/run_scripts/finetune.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
OPENPI_REPO="$HOME/openpi"

_fail_setup() {
    echo "Setup failed."
    return 1 2>/dev/null || exit 1
}

# ── 1. Verify openpi repo ──────────────────────────────────────────────────
if [ ! -d "$OPENPI_REPO" ]; then
    echo "ERROR: openpi repo not found at $OPENPI_REPO"
    echo "  Clone it first:"
    echo "    git clone https://github.com/Physical-Intelligence/openpi ~/openpi"
    _fail_setup
fi

# ── 2. Verify uv ──────────────────────────────────────────────────────────
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found. Install it with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  source \$HOME/.local/bin/env"
    _fail_setup
fi

# ── 3. Create / sync the openpi venv ──────────────────────────────────────
INSTALL_FLAG="$OPENPI_REPO/.positronic_installed_$(uname)"

cd "$OPENPI_REPO"
if [ ! -f "$INSTALL_FLAG" ]; then
    echo "Installing openpi via uv sync (first time — may take a few minutes)…"
    uv sync && touch "$INSTALL_FLAG"
else
    echo "openpi already installed. Delete $INSTALL_FLAG to force reinstall."
fi

export PYTHON="$(uv run which python3)"
echo "Python: $PYTHON"

# ── 4. Symlink so101_policy.py into openpi's policies package ─────────────
SO101_POLICY_SRC="$MODEL_ROOT/training_config/so101_policy.py"
SO101_POLICY_DST="$OPENPI_REPO/src/openpi/policies/so101_policy.py"

if [ ! -e "$SO101_POLICY_DST" ]; then
    echo "Symlinking so101_policy.py into openpi policies…"
    ln -s "$SO101_POLICY_SRC" "$SO101_POLICY_DST"
    echo "  $SO101_POLICY_DST -> $SO101_POLICY_SRC"
else
    echo "so101_policy.py already linked."
fi

cd "$REPO_ROOT"
echo ""
echo "Setup complete. Activate the openpi environment with:"
echo "  source $OPENPI_REPO/.venv/bin/activate"
