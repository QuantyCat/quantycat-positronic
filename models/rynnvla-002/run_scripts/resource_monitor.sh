#!/bin/bash
# resource_monitor.sh — Log CPU, RAM, and GPU usage to CSV during training.
#
# Output path is read from config.yaml (training_output/<task>_<robot>/resources.csv).
# Run this in the background before starting training, then kill it when done.
#
# Usage:
#   source models/rynnvla-002/run_scripts/setup.sh   # once per session
#   bash run_scripts/resource_monitor.sh &            # start in background
#   MONITOR_PID=$!
#   bash run_scripts/finetune.sh                      # run training
#   kill $MONITOR_PID                                 # stop monitor

if [ -z "$MODEL_ROOT" ] || [ -z "$PYTHON" ]; then
    echo "ERROR: run 'source models/rynnvla-002/run_scripts/setup.sh' first"
    exit 1
fi

$PYTHON $MODEL_ROOT/visualization/resource_monitor.py "$@"
