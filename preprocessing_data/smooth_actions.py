#!/usr/bin/env python3
"""Smooth action trajectories in a LeRobot dataset to reduce teleoperation jitter.

Applies a Gaussian filter along the time axis of the action column for each
episode. Gripper joint(s) are excluded by default because gripper open/close
events should remain sharp.

Videos are not modified — smoothing only affects the parquet action column.

Usage:
    python smooth_actions.py --src my_data/trimmed_nopause --dst my_data/smoothed

    # Tune smoothing strength:
    python smooth_actions.py --src my_data/trimmed_nopause --dst my_data/smoothed \
        --sigma 2.0

    # Also smooth the gripper joint:
    python smooth_actions.py --src my_data/trimmed_nopause --dst my_data/smoothed \
        --smooth-all-joints

    # Preview delta stats before vs after without writing:
    python smooth_actions.py --src my_data/trimmed_nopause --dst my_data/smoothed \
        --dry-run
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

FPS     = 30
CAMERAS = ["observation.images.front", "observation.images.wrist"]


def smooth_episode_actions(actions: np.ndarray, sigma: float,
                           smooth_all_joints: bool) -> np.ndarray:
    smoothed = actions.copy().astype(np.float64)
    n_joints = actions.shape[1]
    # Default: smooth all joints except the last (gripper)
    joints_to_smooth = range(n_joints) if smooth_all_joints else range(n_joints - 1)
    for j in joints_to_smooth:
        smoothed[:, j] = gaussian_filter1d(actions[:, j], sigma=sigma)
    return smoothed.astype(actions.dtype)


def compute_feature_stats(df, orig_image_stats):
    stats = {}
    for col in df.columns:
        if col in orig_image_stats:
            s = dict(orig_image_stats[col])
            s["count"] = [len(df)]
            stats[col] = s
            continue
        try:
            vals = df[col].values
            arr = (np.stack([np.asarray(v, dtype=np.float64) for v in vals])
                   if hasattr(vals[0], "__len__")
                   else vals.astype(np.float64)[:, None])
        except Exception:
            continue
        stats[col] = {
            "min":   arr.min(axis=0).tolist(),
            "max":   arr.max(axis=0).tolist(),
            "mean":  arr.mean(axis=0).tolist(),
            "std":   arr.std(axis=0).tolist(),
            "count": [len(arr)],
        }
    return stats


def delta_stats(actions):
    d = np.diff(actions, axis=0)
    return d.std(axis=0), np.abs(d).max(axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src",  type=Path, required=True)
    parser.add_argument("--dst",  type=Path, required=True)
    parser.add_argument("--sigma", type=float, default=1.5,
                        help="Gaussian filter sigma in frames (default: 1.5 ≈ 3-frame window)")
    parser.add_argument("--smooth-all-joints", action="store_true",
                        help="Also smooth the gripper joint (last joint). Off by default.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print before/after delta stats without writing")
    args = parser.parse_args()

    src_data = args.src / "data/chunk-000"
    src_meta = args.src / "meta"
    ep_paths = sorted(src_data.glob("episode_*.parquet"))

    print(f"Source:      {args.src}")
    print(f"Destination: {args.dst}")
    print(f"Sigma:       {args.sigma} frames")
    print(f"Gripper:     {'smoothed' if args.smooth_all_joints else 'untouched'}")
    print(f"Episodes:    {len(ep_paths)}")
    print()

    if args.dry_run:
        print(f"{'ep':>3}  {'joint':>5}  {'std_before':>10}  {'std_after':>9}  {'max_before':>10}  {'max_after':>9}")
        print("-" * 65)
        for p in ep_paths:
            ep = int(p.stem.split("_")[1])
            df = pd.read_parquet(p)
            actions = np.array([np.asarray(v) for v in df["action"].values])
            smoothed = smooth_episode_actions(actions, args.sigma, args.smooth_all_joints)
            std_b, max_b = delta_stats(actions)
            std_a, max_a = delta_stats(smoothed)
            for j in range(actions.shape[1]):
                label = f"j{j}{'*' if j == actions.shape[1]-1 and not args.smooth_all_joints else ''}"
                print(f"{ep:>3}  {label:>5}  {std_b[j]:>10.5f}  {std_a[j]:>9.5f}  "
                      f"{max_b[j]:>10.5f}  {max_a[j]:>9.5f}")
        return

    # Load original image stats to preserve them
    orig_stats_map = {}
    with open(src_meta / "episodes_stats.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            orig_stats_map[obj["episode_index"]] = {
                k: v for k, v in obj["stats"].items()
                if k in ("observation.images.front", "observation.images.wrist")
            }

    # Copy videos and meta structure as-is (smoothing doesn't touch video)
    for cam in CAMERAS:
        src_cam = args.src / "videos/chunk-000" / cam
        dst_cam = args.dst / "videos/chunk-000" / cam
        if src_cam.exists():
            shutil.copytree(src_cam, dst_cam, dirs_exist_ok=True)

    (args.dst / "meta").mkdir(parents=True, exist_ok=True)

    ep_meta_out, ep_stats_out = [], []
    global_offset = 0

    for p in ep_paths:
        ep = int(p.stem.split("_")[1])
        dst_pq = args.dst / "data/chunk-000" / p.name

        print(f"ep{ep:02d}  smoothing actions", end="", flush=True)

        df = pd.read_parquet(p)
        actions = np.array([np.asarray(v) for v in df["action"].values])
        smoothed = smooth_episode_actions(actions, args.sigma, args.smooth_all_joints)

        # Replace action column with smoothed values (keep same list-of-arrays format)
        df["action"] = [smoothed[i].tolist() for i in range(len(smoothed))]

        n = len(df)
        dst_pq.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst_pq, index=False)

        std_b, _ = delta_stats(actions)
        std_a, _ = delta_stats(smoothed)
        print(f" ({n} frames, arm delta std: {std_b[:5].mean():.5f} → {std_a[:5].mean():.5f}) ✓")

        ep_meta_out.append({"episode_index": ep,
                            "tasks": ["Put the screwdriver into the cup"],
                            "length": n})
        ep_stats_out.append({"episode_index": ep,
                             "stats": compute_feature_stats(df, orig_stats_map.get(ep, {}))})
        global_offset += n

    with open(src_meta / "info.json") as f:
        info = json.load(f)
    info["total_frames"] = global_offset
    with open(args.dst / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)
    with open(args.dst / "meta" / "episodes.jsonl", "w") as f:
        for ep in ep_meta_out:
            f.write(json.dumps(ep) + "\n")
    with open(args.dst / "meta" / "episodes_stats.jsonl", "w") as f:
        for ep in ep_stats_out:
            f.write(json.dumps(ep) + "\n")
    shutil.copy(src_meta / "tasks.jsonl", args.dst / "meta" / "tasks.jsonl")

    print(f"\nDone: {global_offset} frames → {args.dst}")


if __name__ == "__main__":
    main()
