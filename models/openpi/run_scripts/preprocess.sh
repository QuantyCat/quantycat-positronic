#!/usr/bin/env bash
# Compute openpi normalisation statistics for the SO-101 dataset.
#
# Run after setting hf_repo_id in models/openpi/config.yaml.
#
# Usage:
#   source models/openpi/run_scripts/setup.sh
#   bash models/openpi/run_scripts/preprocess.sh

_fail_preprocess() {
    return 1 2>/dev/null || exit 1
}

if [ -z "$MODEL_ROOT" ] || [ -z "$PYTHON" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    MODEL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    if [ -z "$PYTHON" ]; then
        PYTHON="$(command -v python3)"
    fi
fi

if [ -z "$MODEL_ROOT" ] || [ -z "$PYTHON" ]; then
    echo "ERROR: source $MODEL_ROOT/run_scripts/setup.sh first"
    _fail_preprocess
fi

echo "--- Step 1: Compute openpi normalisation statistics ---"
$PYTHON "$MODEL_ROOT/preprocessing/step1_compute_norm_stats.py" || _fail_preprocess

echo ""
echo "Preprocessing complete."
