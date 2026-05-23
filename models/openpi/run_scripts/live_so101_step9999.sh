#!/usr/bin/env bash
set -euo pipefail

# Run the OpenPI pi0.5 LoRA step-9999 policy on the live SO-101 follower.
# Pass extra arguments through to inference.py, for example:
#   bash models/openpi/run_scripts/live_so101_step9999.sh --check-only
#   bash models/openpi/run_scripts/live_so101_step9999.sh --dry-run --max-steps 5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO="${REPO:-$DEFAULT_REPO}"
OPENPI_REPO="${OPENPI_REPO:-$REPO/vendor/openpi}"
PYTHON="${PYTHON:-${OPENPI_REPO}/.venv/bin/python}"
CONFIG="${CONFIG:-${REPO}/models/openpi/inference/inference_config.json}"
VENDOR_ROOT="${VENDOR_ROOT:-${REPO}/_vendor}"

cd "$REPO"
if [ -d "$VENDOR_ROOT/scservo_sdk" ] && ! "$PYTHON" -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('scservo_sdk') else 1)"; then
    export PYTHONPATH="${VENDOR_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi
exec "$PYTHON" models/openpi/inference/inference.py --deploy-config "$CONFIG" "$@"
