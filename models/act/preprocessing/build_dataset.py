"""
Stage 1 — Prepare ACT dataset (models/act/build_dataset.py)

ACT trains via lerobot-train which requires a LeRobot v3.0 dataset.
This script copies the input dataset into $WORK_DIR/dataset and upgrades
it to v3.0 if needed. The original input data is never modified.

Usage:
    python3 models/act/build_dataset.py \
        --input-dir $INPUT_DIR \
        --work-dir $WORK_DIR
"""

import argparse
import json
import os
import shutil
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--work-dir",  required=True)
    args = parser.parse_args()

    input_dir   = os.path.abspath(args.input_dir)
    work_dir    = os.path.abspath(args.work_dir)
    dataset_dir = os.path.join(work_dir, "dataset")

    with open(os.path.join(input_dir, "meta", "info.json")) as f:
        info = json.load(f)

    version = info.get("codebase_version", "unknown")
    print(f"Input dataset version: {version}")

    if os.path.exists(dataset_dir):
        print(f"ERROR: {dataset_dir} already exists. Delete it before rebuilding.")
        sys.exit(1)

    print(f"Copying input data to {dataset_dir}...")
    shutil.copytree(input_dir, dataset_dir)

    if version == "v3.0":
        print("Already v3.0 — nothing to do.")
        return

    if version != "v2.1":
        print(f"ERROR: unsupported dataset version '{version}'. Expected v2.1 or v3.0.")
        sys.exit(1)

    try:
        from lerobot.scripts.convert_dataset_v21_to_v30 import convert_dataset
    except ImportError:
        print("ERROR: lerobot is not installed. Install it with: pip install lerobot")
        sys.exit(1)

    print("Upgrading from v2.1 to v3.0...")
    convert_dataset(
        repo_id="dataset",
        root=dataset_dir,
        push_to_hub=False,
    )
    print(f"Done. Dataset ready at {dataset_dir}")


if __name__ == "__main__":
    main()
