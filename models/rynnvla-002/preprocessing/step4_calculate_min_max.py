"""
Step 4 — Calculate action and state min/max normalization values.

Runs calculate_min_max_action.py and calculate_min_max_state.py and saves
results to $WORK_DIR/min_max_action.txt and $WORK_DIR/min_max_state.txt.
Skips if both files already exist.

Run from repo root:
    python3 models/rynnvla-002/preprocessing/calculate_min_max.py
"""

import os
import sys
import subprocess
import yaml

CONFIG_PATH = "models/rynnvla-002/config.yaml"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

work_dir = os.path.abspath(config["work_dir"])
training_data = os.path.join(work_dir, "training_data")
action_stats = os.path.join(work_dir, "min_max_action.txt")
state_stats = os.path.join(work_dir, "min_max_state.txt")

if os.path.exists(action_stats) and os.path.exists(state_stats):
    print("Min/max already calculated — skipping.")
    print(f"  Action stats: {action_stats}")
    print(f"  State stats:  {state_stats}")
    print("Delete these files to rerun.")
    sys.exit(0)

for script, output_file in [
    ("step4_calculate_min_max_action.py", action_stats),
    ("step4_calculate_min_max_state.py", state_stats),
]:
    print(f"Running {script}...")
    proc = subprocess.Popen(
        [sys.executable, os.path.join(SCRIPT_DIR, script), training_data],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines = []
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    proc.wait()
    if proc.returncode != 0:
        sys.exit(proc.returncode)
    with open(output_file, "w") as f:
        f.writelines(lines)
    print(f"  Saved to {output_file}")
