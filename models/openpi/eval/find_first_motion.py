#!/usr/bin/env python3
"""Find the first frame where significant motion starts in each training episode.

Helps diagnose how long the demonstrator held still at the start of each episode.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATASET_ROOT = REPO / "my_data/input_data"
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--threshold-deg", type=float, default=2.0,
                        help="shoulder_lift delta threshold for 'first real motion'")
    args = parser.parse_args()

    data_dir = args.dataset_root / "data/chunk-000"
    episodes = sorted(data_dir.glob("episode_*.parquet"))
    print(f"Found {len(episodes)} episodes")
    print()
    print(f"{'ep':>4}  {'frames':>6}  {'dur_s':>6}  {'first_motion_fr':>15}  {'first_motion_s':>14}  state0_lift  action0_lift")
    print("-" * 90)

    for ep_path in episodes:
        df = pd.read_parquet(ep_path)
        n = len(df)
        ep_idx = int(ep_path.stem.split("_")[1])

        ts = df["timestamp"].values
        duration = ts[-1] - ts[0]

        states = np.stack([np.rad2deg(np.asarray(df.iloc[i]["observation.state"], dtype=np.float32)[:6]) for i in range(n)])
        actions = np.stack([np.rad2deg(np.asarray(df.iloc[i]["action"], dtype=np.float32)[:6]) for i in range(n)])
        deltas = actions.copy()
        deltas[:, :5] -= states[:, :5]

        # Find first frame where shoulder_lift delta exceeds threshold
        lift_delta = np.abs(deltas[:, 1])
        motion_frames = np.where(lift_delta > args.threshold_deg)[0]
        first_motion_frame = int(motion_frames[0]) if len(motion_frames) > 0 else -1
        first_motion_s = float(ts[first_motion_frame]) if first_motion_frame >= 0 else float("nan")

        state0_lift = float(states[0, 1])
        action0_lift = float(actions[0, 1])

        fm_str = str(first_motion_frame) if first_motion_frame >= 0 else "never"
        fms_str = f"{first_motion_s:.2f}s" if first_motion_frame >= 0 else "never"
        print(f"{ep_idx:>4}  {n:>6}  {duration:>6.1f}  {fm_str:>15}  {fms_str:>14}  {state0_lift:>11.2f}°  {action0_lift:>11.2f}°")


if __name__ == "__main__":
    main()
