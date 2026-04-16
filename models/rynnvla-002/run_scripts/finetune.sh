#!/usr/bin/env bash

if [ -z "$MODEL_ROOT" ] || [ -z "$PYTHON" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    MODEL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    if [ -z "$PYTHON" ]; then
        PYTHON="$(command -v python3)"
    fi
fi

if [ -z "$MODEL_ROOT" ] || [ -z "$PYTHON" ]; then
    echo "ERROR: source $MODEL_ROOT/run_scripts/setup.sh first, or ensure python3 is on PATH"
    exit 1
fi

export PYTORCH_ALLOC_CONF=expandable_segments:True 

$PYTHON "$MODEL_ROOT/fine_tuning/finetune.py"
