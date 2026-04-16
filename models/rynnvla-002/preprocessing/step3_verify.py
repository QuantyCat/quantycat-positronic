"""
Verifier — checks preprocessing outputs for errors
models/rynnvla-002/preprocessing/verify.py

Checks:
  - training_data/ exists and has the expected episode structure
  - conversation JSON split files exist and are well-formed
  - All image, action, and state files referenced in the JSON exist on disk
  - Action chunk sizes match config
  - Image counts match config (his * 2 cameras)

Usage:
    python3 models/rynnvla-002/preprocessing/verify.py
"""

import json
import os
import random
import sys
from glob import glob

import yaml

CONFIG_PATH = "models/rynnvla-002/config.yaml"

def check(condition, msg):
    if not condition:
        print(f"  FAIL: {msg}")
        return False
    return True

def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    work_dir   = os.path.abspath(config["work_dir"])
    chunk_size = config["chunk_size"]
    his        = config["his"]
    resolution = config["resolution"]
    task_label = config["task_label"]

    errors = 0

    # --- Check training_data/ ---
    print("=== Step 1: training_data/ ===")
    training_data_dir = os.path.join(work_dir, "training_data")
    if not check(os.path.exists(training_data_dir), f"{training_data_dir} does not exist"):
        errors += 1
    else:
        task_dirs = [d for d in os.listdir(training_data_dir) if os.path.isdir(os.path.join(training_data_dir, d))]
        print(f"  Tasks found: {task_dirs}")
        episode_count = 0
        for task in task_dirs:
            task_path = os.path.join(training_data_dir, task)
            episodes = sorted(os.listdir(task_path))
            episode_count += len(episodes)
            for ep in episodes:
                ep_path = os.path.join(task_path, ep)
                for subdir in ["front_image", "wrist_image", "state", "abs_action"]:
                    if not check(os.path.isdir(os.path.join(ep_path, subdir)),
                                 f"{ep}/{subdir} missing"):
                        errors += 1
        print(f"  Episodes found: {episode_count}")

    # --- Check conversations JSON ---
    print("\n=== Step 2: conversations JSON ===")
    pattern = os.path.join(
        work_dir,
        "conversations",
        f"libero_{task_label}_his_{his}_*_img_state_abs_ck_1_{resolution}.json",
    )
    conv_files = sorted(glob(pattern))

    if not check(bool(conv_files), f"no conversation JSON files found matching {pattern}"):
        errors += 1
        print(f"\nTotal errors: {errors}")
        sys.exit(1 if errors else 0)
    total_conversations = 0
    total_samples_checked = 0
    missing_files = 0
    malformed = 0

    for conv_file in conv_files:
        with open(conv_file) as f:
            convs = json.load(f)

        print(f"  {os.path.basename(conv_file)}: {len(convs)} conversations")
        total_conversations += len(convs)

        sample = random.sample(convs, min(200, len(convs)))
        total_samples_checked += len(sample)

        for conv in sample:
            gpt_value = conv["conversations"][1]["value"]
            action_token_count = gpt_value.count("<|action|>")
            if not check(action_token_count == chunk_size,
                         f"entry has {action_token_count} action tokens, expected {chunk_size}"):
                malformed += 1

            expected_images = his * 2
            if not check(len(conv["image"]) == expected_images,
                         f"entry has {len(conv['image'])} images, expected {expected_images} (his={his} * 2 cameras)"):
                malformed += 1

            for img_path in conv["image"]:
                if not os.path.exists(img_path):
                    missing_files += 1
                    if missing_files <= 5:
                        print(f"  FAIL: missing image: {img_path}")

            if not check(len(conv["action"]) == chunk_size,
                         f"entry has {len(conv['action'])} action files, expected {chunk_size}"):
                malformed += 1
            for action_path in conv["action"]:
                if not os.path.exists(action_path):
                    missing_files += 1
                    if missing_files <= 5:
                        print(f"  FAIL: missing action: {action_path}")

            if not os.path.exists(conv["state"]):
                missing_files += 1
                if missing_files <= 5:
                    print(f"  FAIL: missing state: {conv['state']}")

    if missing_files > 5:
        print(f"  ... and {missing_files - 5} more missing files")

    errors += missing_files + malformed

    # --- Summary ---
    print("\n=== Summary ===")
    print(f"  Conversations:   {total_conversations}")
    print(f"  Sample checked:  {total_samples_checked}")
    print(f"  Missing files:   {missing_files}")
    print(f"  Malformed:       {malformed}")
    print(f"  Total errors:    {errors}")

    if errors == 0:
        print("\nAll checks passed.")
    else:
        print("\nErrors found — check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
