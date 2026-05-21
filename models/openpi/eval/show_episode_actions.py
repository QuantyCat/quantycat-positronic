#!/usr/bin/env python3
"""Show ground-truth action deltas across frames of a training episode.

This is pure data inspection — no policy, no robot.
Helps answer: what motions did the demonstrator actually make?
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATASET_ROOT = REPO / "my_data/input_data"

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, default=7)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--stride", type=int, default=1, help="Print every N-th frame")
    args = parser.parse_args()

    parquet = args.dataset_root / "data/chunk-000" / f"episode_{args.episode:06d}.parquet"
    if not parquet.is_file():
        print(f"Not found: {parquet}", file=sys.stderr)
        return 1

    df = pd.read_parquet(parquet)
    print(f"Episode {args.episode}: {len(df)} frames")
    print()

    def fmt(vals):
        return "[" + ", ".join(f"{v:7.2f}" for v in vals) + "]"

    print("frame  " + "  ".join(f"{n[:9]:>9}" for n in JOINT_NAMES))
    print("       state_deg (shoulder_pan ... gripper)")
    print("-" * 90)

    for i in range(0, len(df), args.stride):
        row = df.iloc[i]
        state = np.asarray(row["observation.state"], dtype=np.float32)
        action = np.asarray(row["action"], dtype=np.float32)

        state_deg = np.rad2deg(state[:6])
        action_deg = np.rad2deg(action[:6])
        delta_deg = action_deg.copy()
        delta_deg[:5] -= state_deg[:5]  # gripper is absolute

        print(f"fr{i:03d}  state  {fmt(state_deg.tolist())}")
        print(f"       action {fmt(action_deg.tolist())}")
        print(f"       delta  {fmt(delta_deg.tolist())}")
        print()

    # Summary stats
    states = np.stack([np.rad2deg(np.asarray(df.iloc[i]["observation.state"], dtype=np.float32)[:6]) for i in range(len(df))])
    actions = np.stack([np.rad2deg(np.asarray(df.iloc[i]["action"], dtype=np.float32)[:6]) for i in range(len(df))])
    deltas = actions.copy()
    deltas[:, :5] -= states[:, :5]

    print("=" * 90)
    print("EPISODE SUMMARY (degrees)")
    print(f"  state range:   min={fmt(states.min(axis=0).tolist())}  max={fmt(states.max(axis=0).tolist())}")
    print(f"  delta range:   min={fmt(deltas.min(axis=0).tolist())}  max={fmt(deltas.max(axis=0).tolist())}")
    print(f"  max |delta|:       {fmt(np.abs(deltas).max(axis=0).tolist())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
