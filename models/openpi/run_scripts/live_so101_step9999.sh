#!/usr/bin/env bash
set -euo pipefail

# Run the OpenPI pi0.5 LoRA step-9999 policy on the live SO-101 follower.
# Pass extra arguments through to live_so101_openpi.py, for example:
#   bash models/openpi/run_scripts/live_so101_step9999.sh --check-only
#   bash models/openpi/run_scripts/live_so101_step9999.sh --dry-run --max-steps 5

REPO="${REPO:-/home/caroline/quantycat-positronic}"
OPENPI_REPO="${OPENPI_REPO:-/home/caroline/openpi}"
PYTHON="${PYTHON:-${OPENPI_REPO}/.venv/bin/python}"
CONFIG="${CONFIG:-${REPO}/models/openpi/deployment/pi05_lora_step9999_so101.json}"

cd "$REPO"
exec "$PYTHON" models/openpi/deployment/live_so101_openpi.py --deploy-config "$CONFIG" "$@"
