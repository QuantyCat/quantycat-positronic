"""
Step 7 — Write the RynnVLA-002 training/validation config YAML files.

Points the training configs at the record.json files produced by merge_records.py.
Uses $HOME so the path is portable across machines.

Run from repo root:
    python3 models/rynnvla-002/preprocessing/update_train_config.py
"""

import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "models/rynnvla-002/config.yaml"

with CONFIG_PATH.open() as f:
    config = yaml.safe_load(f)

work_dir = Path(config["work_dir"])
if not work_dir.is_absolute():
    work_dir = (REPO_ROOT / work_dir).resolve()
his = config["his"]
chunk_size = config["chunk_size"]
resolution = config["resolution"]
run_validation = bool(config.get("run_validation", False))
home = os.path.expanduser("~")
tokens_root = work_dir / "tokens" / "vla_data"

# Filename derived from config values — must match what finetune.py constructs
rynnvla_repo = REPO_ROOT / "vendor/rynnvla-002/rynnvla-002"
config_dir = rynnvla_repo / "configs" / "lerobot"
base_name = f"his_{his}_third_view_wrist_w_state_{chunk_size}_{resolution}_pretokenize"

split_to_name = {
    "train": f"{base_name}.yaml",
    "val_ind": f"{base_name}_val_ind.yaml",
    "val_ood": f"{base_name}_val_ood.yaml",
}

available = {}
for split_name in split_to_name:
    record_json = tokens_root / split_name / "record.json"
    if record_json.exists():
        available[split_name] = record_json

if "train" not in available:
    legacy_train = tokens_root / "record.json"
    if legacy_train.exists():
        available["train"] = legacy_train

if "train" not in available:
    print(f"ERROR: train record.json not found under {tokens_root}")
    print("Run merge_records.py first.")
    sys.exit(1)

if run_validation and "val_ind" not in available:
    print(f"ERROR: validation is enabled but val_ind record.json was not found under {tokens_root}")
    print("Run preprocessing again to build split validation artifacts.")
    sys.exit(1)

if "val_ood" not in available and "val_ind" in available:
    available["val_ood"] = available["val_ind"]

for split_name, config_name in split_to_name.items():
    if split_name != "train" and split_name not in available:
        continue
    portable_path = str(available[split_name]).replace(home, "$HOME")
    config_path = config_dir / config_name
    with config_path.open("w") as f:
        f.write(f"META:\n  - path: '{portable_path}'\n")
    print(f"Updated {config_path}")
    print(f"  path: {portable_path}")
