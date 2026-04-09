#!/bin/bash
# visualization.sh — Generate a training dashboard HTML from a training run.
#
# Paths are read from config.yaml by default. Run with no args to use config defaults.
#
# Usage:
#   source models/rynnvla-002/run_scripts/setup.sh   # once per session
#   bash run_scripts/visualization.sh                 # uses config.yaml defaults
#   bash run_scripts/visualization.sh <run_dir>       # custom run directory
#   bash run_scripts/visualization.sh <run_dir> <output_path>

if [ -z "$MODEL_ROOT" ] || [ -z "$PYTHON" ]; then
    echo "ERROR: run 'source models/rynnvla-002/run_scripts/setup.sh' first"
    exit 1
fi

$PYTHON $MODEL_ROOT/visualization/generate_dashboard.py "$@"
