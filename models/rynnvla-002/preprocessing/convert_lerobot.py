"""
Stage 1 — Convert LeRobot dataset to RynnVLA-002 format
models/rynnvla-002/preprocessing/convert_lerobot.py

Converts a LeRobot v2.1 dataset into the per-episode folder structure
that RynnVLA-002's conversation generation script expects.

Output structure:
    $WORK_DIR/extracted/
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
    source models/rynnvla-002/config.sh
    python3 models/rynnvla-002/preprocessing/convert_lerobot.py
"""

import json
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

def main():
    input_dir  = os.environ.get("INPUT_DIR")
    work_dir   = os.environ.get("WORK_DIR")
    chunk_size = os.environ.get("CHUNK_SIZE")

    if not input_dir or not work_dir or not chunk_size:
        print("ERROR: INPUT_DIR, WORK_DIR, and CHUNK_SIZE must be set. Run: source models/rynnvla-002/config.sh")
        sys.exit(1)

    CHUNK_SIZE = int(chunk_size)

    input_dir  = os.path.abspath(args.input_dir)
    work_dir   = os.path.abspath(args.work_dir)
    output_dir = os.path.join(work_dir, "extracted")

    if os.path.exists(output_dir):
        print(f"ERROR: {output_dir} already exists. Delete it before rebuilding.")
        sys.exit(1)

    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print("ERROR: lerobot is not installed. Run: pip install lerobot")
        sys.exit(1)

    # Read task name from dataset metadata
    tasks_path = os.path.join(input_dir, "meta", "tasks.jsonl")
    with open(tasks_path) as f:
        task = json.loads(f.readline())
    task_instruction = task["task"]
    task_name = task_instruction.replace(" ", "_")
    print(f"Task: {task_instruction}")

    # Load dataset
    print("Loading LeRobot dataset...")
    dataset = LeRobotDataset(None, root=input_dir, tolerance_s=1e-4)
    print(f"  {dataset.num_episodes} episodes, {len(dataset)} frames")

    skipped_chunks = 0
    total_chunks = 0

    for ep_idx in tqdm(range(dataset.num_episodes), desc="Episodes"):
        from_idx = dataset.episode_data_index["from"][ep_idx].item()
        to_idx   = dataset.episode_data_index["to"][ep_idx].item()
        ep_len   = to_idx - from_idx

        ep_dir         = os.path.join(output_dir, task_name, f"episode_{ep_idx:06d}")
        front_dir      = os.path.join(ep_dir, "front_image")
        wrist_dir      = os.path.join(ep_dir, "wrist_image")
        state_dir      = os.path.join(ep_dir, "state")
        abs_action_dir = os.path.join(ep_dir, "abs_action")

        os.makedirs(front_dir, exist_ok=True)
        os.makedirs(wrist_dir, exist_ok=True)
        os.makedirs(state_dir, exist_ok=True)

        # Collect all frames for this episode first
        states  = []
        actions = []
        front_images = []
        wrist_images = []

        for i in range(ep_len):
            frame = dataset[from_idx + i]
            # Images come as CHW float32 [0, 1] — convert to HWC uint8
            front = (frame["observation.images.front"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            wrist = (frame["observation.images.wrist"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            front_images.append(front)
            wrist_images.append(wrist)
            states.append(frame["observation.state"].numpy().astype(np.float32))
            actions.append(frame["action"].numpy().astype(np.float32))

        states  = np.array(states)   # (T, 6)
        actions = np.array(actions)  # (T, 6)

        for t in range(ep_len):
            # Save image and state for every timestep
            Image.fromarray(front_images[t]).save(os.path.join(front_dir, f"image_{t}.png"))
            Image.fromarray(wrist_images[t]).save(os.path.join(wrist_dir, f"image_{t}.png"))
            np.save(os.path.join(state_dir, f"state_{t}.npy"), states[t])

            # Action chunks require CHUNK_SIZE future steps
            if t + CHUNK_SIZE > ep_len:
                continue

            total_chunks += 1
            action_chunk = actions[t : t + CHUNK_SIZE]  # (20, 6)

            # Relative actions: subtract current state; keep gripper absolute
            rel_chunk = action_chunk - states[t][np.newaxis, :]
            rel_chunk[:, -1] = action_chunk[:, -1]

            # Skip no-op chunks
            if np.sum(np.abs(rel_chunk)) == 0:
                skipped_chunks += 1
                continue

            chunk_dir = os.path.join(abs_action_dir, f"action_{t}")
            os.makedirs(chunk_dir, exist_ok=True)
            for j in range(CHUNK_SIZE):
                np.save(os.path.join(chunk_dir, f"{j}.npy"), rel_chunk[j])

    print(f"\nDone.")
    print(f"  Extracted data: {output_dir}")
    print(f"  Action chunks written: {total_chunks - skipped_chunks}")
    print(f"  No-op chunks skipped:  {skipped_chunks}")


if __name__ == "__main__":
    main()
