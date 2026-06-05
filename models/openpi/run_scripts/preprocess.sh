#!/usr/bin/env bash
# Compute quantile norm stats for the OpenPI training pipeline.
#
# OpenPI reads norm stats from the LeRobot dataset's meta/stats.json, so this
# just delegates to models/lerobot/preprocess.sh. Run that once and both
# pipelines are ready.
#
# Usage from the repo root:
#   bash models/openpi/run_scripts/preprocess.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"

exec bash "$REPO/models/lerobot/run_scripts/preprocess.sh" "$@"
