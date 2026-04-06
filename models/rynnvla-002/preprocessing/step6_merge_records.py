"""
Step 6 — Merge per-worker record files into a single record.json.

Combines the N-of-N-record.jsonl files written by pretokenize workers
into tokens/vla_data/record.json for training.
Skips if record.json already exists.

Run from repo root:
    python3 models/rynnvla-002/preprocessing/merge_records.py
"""

import os
import sys
import glob
import json
import yaml

CONFIG_PATH = "models/rynnvla-002/config.yaml"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

work_dir   = os.path.abspath(config["work_dir"])
output_dir = os.path.join(work_dir, "tokens", "vla_data")
record_json = os.path.join(output_dir, "record.json")

if os.path.exists(record_json):
    print(f"record.json already exists — skipping.")
    sys.exit(0)

jsonl_files = sorted(glob.glob(os.path.join(output_dir, "*-record.jsonl")))
if not jsonl_files:
    print(f"ERROR: no *-record.jsonl files found in {output_dir}")
    sys.exit(1)

records = []
for f in jsonl_files:
    with open(f) as fh:
        records.extend(json.loads(line) for line in fh if line.strip())
    print(f"  {os.path.basename(f)}: {len(records)} records so far")

with open(record_json, "w") as fh:
    json.dump(records, fh, indent=2)

print(f"Merged {len(records)} records → {record_json}")
