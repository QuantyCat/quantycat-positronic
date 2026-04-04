"""
Stage 1 — Convert LeRobot dataset to RynnVLA-002 format
models/rynnvla-002/preprocessing/convert_lerobot.py

Converts a LeRobot v2.1 dataset into the per-episode folder structure
that RynnVLA-002's conversation generation script expects.

Output structure:
    $WORK_DIR/training_data/
        TASK_NAME/
            episode_000000/
                front_image/image_0.png, image_1.png ...
                wrist_image/image_0.png, image_1.png ...
                state/state_0.npy                        shape (6,)
                abs_action/
                    action_0/0.npy ... 19.npy            relative actions, gripper absolute
                    action_1/0.npy ... 19.npy
                    ...
            episode_000001/
                ...

Usage:
    python3 models/rynnvla-002/preprocessing/convert_lerobot.py
"""

import json
import os
import shutil
import sys

import yaml

import numpy as np
from PIL import Image
from tqdm import tqdm

CONFIG_PATH = "models/rynnvla-002/config.yaml"

def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    input_dir  = config.get("input_dir")
    work_dir   = config.get("work_dir")
    CHUNK_SIZE = config.get("chunk_size")

    if not input_dir or not work_dir or not CHUNK_SIZE:
        print(f"ERROR: input_dir, work_dir, and chunk_size must be set in {CONFIG_PATH}")
        sys.exit(1)

    input_dir  = os.path.abspath(input_dir)
    work_dir   = os.path.abspath(work_dir)
    output_dir = os.path.join(work_dir, "training_data")

    if os.path.exists(output_dir):
        print(f"training_data/ already exists — skipping. Delete {output_dir} to rerun.")
        return

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as e:
        print(f"ERROR: failed to import from lerobot — {e}")
        sys.exit(1)

    # Read task name from dataset metadata
    tasks_path = os.path.join(input_dir, "meta", "tasks.jsonl")
    with open(tasks_path) as f:
        task = json.loads(f.readline())
    task_instruction = task["task"]
    task_name = task_instruction.replace(" ", "_")
    print(f"Task: {task_instruction}")

    # Copy input data to work dir — input is never modified
    dataset_dir = os.path.join(work_dir, "dataset")
    if os.path.exists(dataset_dir):
        print(f"ERROR: {dataset_dir} already exists. Delete it before rebuilding.")
        sys.exit(1)
    print(f"Copying input data to {dataset_dir}...")
    shutil.copytree(input_dir, dataset_dir)

    # Upgrade dataset from v2.1 to v3.0 if needed
    info_path = os.path.join(dataset_dir, "meta", "info.json")
    with open(info_path) as f:
        info = json.load(f)
    if info.get("codebase_version") == "v2.1":
        print("Dataset is v2.1 — upgrading to v3.0...")
        from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset
        convert_dataset(repo_id="dataset", root=dataset_dir, push_to_hub=False)
        print("Upgrade done.")

    # Load dataset
    print("Loading LeRobot dataset...")
    dataset = LeRobotDataset(None, root=dataset_dir, tolerance_s=1e-4)
    print(f"  {dataset.num_episodes} episodes, {len(dataset)} frames")

    skipped_chunks = 0
    total_chunks = 0

    def flush_episode(ep_idx, ep):
        """Write one episode worth of data to disk, then it can be GC'd."""
        nonlocal skipped_chunks, total_chunks

        states  = np.array(ep["states"])
        actions = np.array(ep["actions"])
        ep_len  = len(states)

        ep_dir         = os.path.join(output_dir, task_name, f"episode_{ep_idx:06d}")
        front_dir      = os.path.join(ep_dir, "front_image")
        wrist_dir      = os.path.join(ep_dir, "wrist_image")
        state_dir      = os.path.join(ep_dir, "state")
        abs_action_dir = os.path.join(ep_dir, "abs_action")

        os.makedirs(front_dir, exist_ok=True)
        os.makedirs(wrist_dir, exist_ok=True)
        os.makedirs(state_dir, exist_ok=True)

        for t in range(ep_len):
            Image.fromarray(ep["front_images"][t]).save(os.path.join(front_dir, f"image_{t}.png"))
            Image.fromarray(ep["wrist_images"][t]).save(os.path.join(wrist_dir, f"image_{t}.png"))
            np.save(os.path.join(state_dir, f"state_{t}.npy"), states[t])

            if t + CHUNK_SIZE > ep_len:
                continue

            total_chunks += 1
            action_chunk = actions[t : t + CHUNK_SIZE]

            # Relative actions: subtract current state; keep gripper absolute
            rel_chunk = action_chunk - states[t][np.newaxis, :]
            rel_chunk[:, -1] = action_chunk[:, -1]

            if np.sum(np.abs(rel_chunk)) == 0:
                skipped_chunks += 1
                continue

            chunk_dir = os.path.join(abs_action_dir, f"action_{t}")
            os.makedirs(chunk_dir, exist_ok=True)
            for j in range(CHUNK_SIZE):
                np.save(os.path.join(chunk_dir, f"{j}.npy"), rel_chunk[j])

    # Stream frames one at a time — only one episode is in RAM at once.
    # Frames are ordered by episode in the LeRobot dataset, so we detect
    # episode boundaries and flush each episode to disk before loading the next.
    current_ep_idx = None
    current_ep     = None

    for i in tqdm(range(len(dataset)), desc="Converting episodes"):
        frame  = dataset[i]
        ep_idx = frame["episode_index"].item()

        if ep_idx != current_ep_idx:
            if current_ep is not None:
                flush_episode(current_ep_idx, current_ep)
            current_ep_idx = ep_idx
            current_ep     = {"front_images": [], "wrist_images": [], "states": [], "actions": []}

        current_ep["front_images"].append((frame["observation.images.front"].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        current_ep["wrist_images"].append((frame["observation.images.wrist"].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        current_ep["states"].append(frame["observation.state"].numpy().astype(np.float32))
        current_ep["actions"].append(frame["action"].numpy().astype(np.float32))

    # Flush the last episode
    if current_ep is not None:
        flush_episode(current_ep_idx, current_ep)

    # Clean up intermediate copies — only training_data/ is needed going forward
    for leftover in [dataset_dir, os.path.join(work_dir, "dataset_old")]:
        if os.path.exists(leftover):
            print(f"Cleaning up {leftover}...")
            shutil.rmtree(leftover)

    print(f"\nDone.")
    print(f"  Training data: {output_dir}")
    print(f"  Action chunks written: {total_chunks - skipped_chunks}")
    print(f"  No-op chunks skipped:  {skipped_chunks}")


if __name__ == "__main__":
    main()
