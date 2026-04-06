"""
Step 7 — Write the RynnVLA-002 training config YAML.

Points the training config at the record.json produced by merge_records.py.
Uses $HOME so the path is portable across machines.

Run from repo root:
    python3 models/rynnvla-002/preprocessing/update_train_config.py
"""

import os
import sys
import yaml

CONFIG_PATH = "models/rynnvla-002/config.yaml"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

work_dir    = os.path.abspath(config["work_dir"])
home        = os.path.expanduser("~")
record_json = os.path.join(work_dir, "tokens", "vla_data", "record.json")

if not os.path.exists(record_json):
    print(f"ERROR: record.json not found at {record_json}")
    print("Run merge_records.py first.")
    sys.exit(1)

# Store path relative to $HOME so it works on any machine
portable_path = record_json.replace(home, "$HOME")

rynnvla_repo = os.path.join(home, "RynnVLA-002", "rynnvla-002")
train_config = os.path.join(
    rynnvla_repo,
    "configs", "lerobot",
    "his_1_third_view_wrist_w_state_20_256_pretokenize.yaml"
)

with open(train_config, "w") as f:
    f.write(f"META:\n  - path: '{portable_path}'\n")

print(f"Updated {train_config}")
print(f"  path: {portable_path}")
