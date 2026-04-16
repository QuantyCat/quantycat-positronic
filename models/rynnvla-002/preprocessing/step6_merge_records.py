"""
Step 6 — Merge per-worker record files into per-split record.json files.

Combines the N-of-N-record.jsonl files written by pretokenize workers
into tokens/vla_data/<split>/record.json for training/evaluation.
Also writes tokens/vla_data/record.json as a compatibility alias to train.

Run from repo root:
    python3 models/rynnvla-002/preprocessing/merge_records.py
"""

import os
import sys
import glob
import json
import shutil
import yaml

CONFIG_PATH = "models/rynnvla-002/config.yaml"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

work_dir   = os.path.abspath(config["work_dir"])
root_dir = os.path.join(work_dir, "tokens", "vla_data")

if not os.path.isdir(root_dir):
    print(f"ERROR: token directory not found at {root_dir}")
    sys.exit(0)

split_dirs = [path for path in sorted(glob.glob(os.path.join(root_dir, "*"))) if os.path.isdir(path)]
if not split_dirs:
    split_dirs = [root_dir]

train_record_json = None

for split_dir in split_dirs:
    jsonl_files = sorted(glob.glob(os.path.join(split_dir, "*-record.jsonl")))
    if not jsonl_files:
        continue

    record_json = os.path.join(split_dir, "record.json")
    if os.path.exists(record_json):
        print(f"{record_json} already exists — skipping.")
    else:
        records = []
        for f in jsonl_files:
            with open(f) as fh:
                records.extend(json.loads(line) for line in fh if line.strip())
            print(f"  {os.path.basename(split_dir) or 'root'} / {os.path.basename(f)}: {len(records)} records so far")

        with open(record_json, "w") as fh:
            json.dump(records, fh, indent=2)

        print(f"Merged {len(records)} records → {record_json}")

    if os.path.basename(split_dir) == "train":
        train_record_json = record_json

if train_record_json:
    compat_record_json = os.path.join(root_dir, "record.json")
    shutil.copyfile(train_record_json, compat_record_json)
    print(f"Wrote compatibility train alias → {compat_record_json}")
