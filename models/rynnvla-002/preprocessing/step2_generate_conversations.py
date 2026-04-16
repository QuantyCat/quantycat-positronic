"""
Stage 2 — Generate conversation JSON from extracted training data
models/rynnvla-002/preprocessing/generate_conversations.py

Reads the per-episode folder structure produced by convert_lerobot.py and
generates split JSON files of conversations for RynnVLA-002.

Each timestep becomes one conversation:
    human: "What action should the robot take to <task>?" + <state> + <images>
    gpt:   <action> x chunk_size

Output:
    $WORK_DIR/conversations/
        libero_<task_label>_his_<his>_train_img_state_abs_ck_1_<resolution>.json
        libero_<task_label>_his_<his>_val_ind_img_state_abs_ck_1_<resolution>.json
P.S - these variables come from config.yaml

Usage:
    python3 models/rynnvla-002/preprocessing/generate_conversations.py

This file was generated from RynnVLA-002/rynnvla-002/lerobot_util/action_model_conv_generation_w_2_abs_state_all_data.py
"""

import copy
import json
import math
import os
import sys

import yaml
from tqdm import tqdm

CONFIG_PATH = "models/rynnvla-002/config.yaml"


def process_libero_data(
    input_dir: str,
    his: int,
    task_name_for_output: str,
    resolution: int,
    len_action: int,
    output_dir: str,
    run_validation: bool,
    val_ratio: float,
):
    """
    Processes robot trajectory data from a specified input directory to create conversational datasets.

    Expected Directory Structure:
    input_dir/
    ├── TASK_NAME/
    │   ├── episode_000000/
    │   │   ├── abs_action/
    │   │   ├── front_image/
    │   │   ├── wrist_image/
    │   │   └── state/
    │   └── episode_000001/
    │       └── ...
    └── ...
    """
    ACTION_CHUNK_PREDICTION_HORIZON = 1
    SUB_ACTIONS_PER_CHUNK = len_action

    split_convs = {"train": [], "val_ind": []}
    split_traj_count = {"train": 0, "val_ind": 0}

    os.makedirs(output_dir, exist_ok=True)

    print(f"Scanning for task directories in: {input_dir}")

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found at '{input_dir}'")
        return

    task_paths = [os.path.join(input_dir, d) for d in sorted(os.listdir(input_dir)) if os.path.isdir(os.path.join(input_dir, d))]

    if not task_paths:
        print(f"Error: No task subdirectories found in '{input_dir}'. Please check the directory structure.")
        return

    print(f"Found {len(task_paths)} task(s) to process.")
    print("-" * 30)
    print(f"Historical frames (his): {his}")
    print(f"Action chunk prediction horizon: {ACTION_CHUNK_PREDICTION_HORIZON}")
    print(f"Sub-actions per chunk: {SUB_ACTIONS_PER_CHUNK}")
    print(f"Output label: {task_name_for_output}")
    print(f"Resolution: {resolution}")
    print(f"Output directory: {output_dir}")
    print("-" * 30)

    for task_path in tqdm(task_paths, desc="Processing Tasks"):
        task_name_readable = os.path.basename(task_path).replace('_', ' ')

        if not os.path.isdir(task_path):
            continue

        trj_list = [name for name in sorted(os.listdir(task_path)) if os.path.isdir(os.path.join(task_path, name))]
        if not trj_list:
            continue

        val_count = 0
        if run_validation and len(trj_list) > 1:
            val_count = max(1, math.ceil(len(trj_list) * val_ratio))
            val_count = min(val_count, len(trj_list) - 1)
        split_start = len(trj_list) - val_count

        for trj_idx, trj in enumerate(tqdm(trj_list, desc=f"  - Trajectories for '{task_name_readable}'", leave=False)):
            trj_path = os.path.join(task_path, trj)
            split_name = "val_ind" if run_validation and trj_idx >= split_start else "train"

            if not os.path.isdir(trj_path):
                continue

            action_base_path = os.path.join(trj_path, 'abs_action')
            imgs_path = os.path.join(trj_path, 'front_image')
            imgs_path_w = os.path.join(trj_path, 'wrist_image')
            state_path = os.path.join(trj_path, 'state')

            if not all(os.path.exists(p) for p in [action_base_path, imgs_path, imgs_path_w, state_path]):
                print(f"    Warning: Missing required data directories in {trj_path}. Skipping.")
                continue

            try:
                action_dirs_raw = [d for d in os.listdir(action_base_path) if d.startswith('action_') and os.path.isdir(os.path.join(action_base_path, d))]
                img_files_raw = [f for f in os.listdir(imgs_path) if f.startswith('image_') and f.endswith('.png')]
                state_files_raw = [f for f in os.listdir(state_path) if f.startswith('state_') and f.endswith('.npy')]

                action_indices = sorted([int(d.split('_')[1]) for d in action_dirs_raw])
                img_indices = sorted([int(f.split('_')[1].split('.')[0]) for f in img_files_raw])
                state_indices = sorted([int(f.split('_')[1].split('.')[0]) for f in state_files_raw])
            except (ValueError, IndexError) as e:
                print(f"    Warning: Could not parse file/dir indices in {trj_path}. Error: {e}. Skipping.")
                continue

            common_indices = sorted(list(set(action_indices) & set(img_indices) & set(state_indices)))
            if not common_indices:
                print(f"    Warning: No common indices found for all data types in {trj_path}. Skipping.")
                continue

            img_list, img_list_w, action_list, state_list = [], [], [], []

            for idx in common_indices:
                img_file = os.path.join(imgs_path, f"image_{idx}.png")
                img_file_w = os.path.join(imgs_path_w, f"image_{idx}.png")
                action_dir = os.path.join(action_base_path, f"action_{idx}")
                state_file = os.path.join(state_path, f"state_{idx}.npy")

                if os.path.exists(img_file) and os.path.exists(img_file_w) and os.path.isdir(action_dir) and os.path.exists(state_file):
                    try:
                        sub_action_files_raw = [f for f in os.listdir(action_dir) if f.endswith('.npy')]
                        sub_action_files_sorted = sorted(sub_action_files_raw, key=lambda f: int(os.path.splitext(f)[0]))

                        if len(sub_action_files_sorted) == SUB_ACTIONS_PER_CHUNK:
                            sub_action_paths = [os.path.join(action_dir, f) for f in sub_action_files_sorted]
                            img_list.append(img_file)
                            img_list_w.append(img_file_w)
                            action_list.append(sub_action_paths)
                            state_list.append(state_file)
                    except (ValueError, FileNotFoundError) as e:
                        print(f"      Warning: Error processing action dir {action_dir}: {e}. Skipping index {idx}.")

            if not img_list or not action_list or not state_list:
                continue

            for j in range(len(action_list)):
                img_history_start_idx = max(0, j - his + 1)
                img_c = copy.deepcopy(img_list[img_history_start_idx : j + 1])
                img_c_w = copy.deepcopy(img_list_w[img_history_start_idx : j + 1])

                # Pad with the earliest available frame if history is shorter than his
                while len(img_c) < his:
                    img_c.insert(0, img_c[0])
                    img_c_w.insert(0, img_c_w[0])
                action_c = copy.deepcopy(action_list[j])
                state_c = copy.deepcopy(state_list[j])

                if len(action_c) != SUB_ACTIONS_PER_CHUNK:
                    continue

                conv = {
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"What action should the robot take to {task_name_readable}?" + "<|state|>" + "<|image|>" * len(img_c) * 2
                        },
                        {
                            "from": "gpt",
                            "value": "<|action|>" * SUB_ACTIONS_PER_CHUNK
                        },
                    ],
                    "image": img_c + img_c_w,
                    "action": action_c,
                    "state": state_c
                }
                split_convs[split_name].append(conv)

            split_traj_count[split_name] += 1

    print("-" * 30)
    print("Saving dataset...")

    for split_name, convs in split_convs.items():
        if split_name != "train" and not run_validation:
            continue
        output_path = os.path.join(
            output_dir,
            f"libero_{task_name_for_output}_his_{his}_{split_name}_img_state_abs_ck_{ACTION_CHUNK_PREDICTION_HORIZON}_{resolution}.json",
        )
        with open(output_path, "w") as f:
            json.dump(convs, f, indent=2)
        print(f"Saved {split_name} conversations to: {output_path}")

    print("\n--- Summary ---")
    print(f"Train trajectories: {split_traj_count['train']}")
    print(f"Train conversations: {len(split_convs['train'])}")
    if run_validation:
        print(f"Validation trajectories: {split_traj_count['val_ind']}")
        print(f"Validation conversations: {len(split_convs['val_ind'])}")


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    work_dir   = config.get("work_dir")
    his        = config.get("his")
    resolution = config.get("resolution")
    chunk_size = config.get("chunk_size")
    task_label = config.get("task_label")
    run_validation = bool(config.get("run_validation", False))
    val_ratio = float(config.get("val_ratio", 0.1))

    if not all([work_dir, his, resolution, chunk_size, task_label]):
        print(f"ERROR: work_dir, his, resolution, chunk_size, and task_label must be set in {CONFIG_PATH}")
        sys.exit(1)

    work_dir   = os.path.abspath(work_dir)
    input_dir  = os.path.join(work_dir, "training_data")
    output_dir = os.path.join(work_dir, "conversations")
    output_file = os.path.join(output_dir, f"libero_{task_label}_his_{his}_train_img_state_abs_ck_1_{resolution}.json")

    if os.path.exists(output_file):
        print(f"conversations already exist — skipping. Delete {output_file} to rerun.")
        return

    process_libero_data(
        input_dir=input_dir,
        his=his,
        task_name_for_output=task_label,
        resolution=resolution,
        len_action=chunk_size,
        output_dir=output_dir,
        run_validation=run_validation,
        val_ratio=val_ratio,
    )


if __name__ == "__main__":
    main()
