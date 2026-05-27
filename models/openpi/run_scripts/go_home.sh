#!/usr/bin/env bash
set -euo pipefail

# Move the SO-101 back to the home/rest position.
# Pass extra arguments through to go_home.py, for example:
#   bash models/openpi/run_scripts/go_home.sh
#   bash models/openpi/run_scripts/go_home.sh --home 4 -85 92 67 6 0.4
#   bash models/openpi/run_scripts/go_home.sh --steps 80 --period 0.05

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO="${REPO:-$DEFAULT_REPO}"
OPENPI_REPO="${OPENPI_REPO:-$REPO/vendor/openpi}"
PYTHON="${PYTHON:-${OPENPI_REPO}/.venv/bin/python}"
VENDOR_ROOT="${VENDOR_ROOT:-${REPO}/_vendor}"

cd "$REPO"
if [ -d "$VENDOR_ROOT/scservo_sdk" ] && ! "$PYTHON" -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('scservo_sdk') else 1)"; then
    export PYTHONPATH="${VENDOR_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi
exec "$PYTHON" models/openpi/inference/go_home.py "$@"
