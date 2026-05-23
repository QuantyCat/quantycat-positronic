#!/usr/bin/env python3
"""Remove intra-episode pauses from a LeRobot dataset.

A "pause" is any run of consecutive frames where the commanded arm speed
(L2 norm of action delta over joints 0-4) stays below --speed-threshold
for at least --min-pause-frames frames.

Fixes two problems common in real-world teleoperation recordings:
  - Demonstrator hesitating mid-motion
  - Post-task hovering before the recording was stopped
  - Residual hold phase not caught by trim_dataset.py

Timestamps are reassigned as uniform 1/fps intervals after removal,
so parquet and video stay in sync.

Usage:
    python remove_pauses.py --src my_data/trimmed --dst my_data/trimmed_nopause

    # Preview what would be removed without writing:
    python remove_pauses.py --src my_data/trimmed --dst my_data/trimmed_nopause --dry-run

    # Tune thresholds:
    python remove_pauses.py --src my_data/trimmed --dst my_data/trimmed_nopause \
        --speed-threshold 0.015 --min-pause-frames 20
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FPS     = 30
CAMERAS = ["observation.images.front", "observation.images.wrist"]


def find_keep_mask(actions: np.ndarray, speed_threshold: float,
                   min_pause_frames: int) -> np.ndarray:
    speed = np.zeros(len(actions))
    speed[1:] = np.linalg.norm(np.diff(actions[:, :5], axis=0), axis=1)

    keep = np.ones(len(actions), dtype=bool)
    i = 0
    while i < len(speed):
        if speed[i] < speed_threshold:
            j = i
            while j < len(speed) and speed[j] < speed_threshold:
                j += 1
            if j - i >= min_pause_frames:
                keep[i:j] = False
            i = j
        else:
            i += 1
    return keep


def process_parquet(src_path, dst_path, new_ep_idx, global_offset,
                    speed_threshold, min_pause_frames):
    df      = pd.read_parquet(src_path)
    actions = np.array([np.asarray(v) for v in df["action"].values])
    keep    = find_keep_mask(actions, speed_threshold, min_pause_frames)
    n_removed = int((~keep).sum())

    df = df[keep].reset_index(drop=True)
    n  = len(df)

    df["frame_index"]   = np.arange(n, dtype=np.int64)
    df["timestamp"]     = (np.arange(n) / FPS).astype(np.float32)
    df["episode_index"] = np.int64(new_ep_idx)
    df["index"]         = np.arange(global_offset, global_offset + n, dtype=np.int64)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst_path, index=False)
    return n, n_removed, keep


def build_select_expr(keep_mask: np.ndarray) -> str:
    intervals, in_run, start = [], False, 0
    for i, k in enumerate(keep_mask):
        if k and not in_run:
            start, in_run = i, True
        elif not k and in_run:
            intervals.append((start, i - 1))
            in_run = False
    if in_run:
        intervals.append((start, len(keep_mask) - 1))
    return "+".join(f"between(n,{s},{e})" for s, e in intervals)


def process_video(src_path, dst_path, keep_mask):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src_path),
        "-vf", f"select='{build_select_expr(keep_mask)}',setpts=N/{FPS}/TB",
        "-vsync", "cfr",
        "-r", str(FPS),
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-an",
        str(dst_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n  ffmpeg error:\n{result.stderr[-600:]}", file=sys.stderr)
        raise RuntimeError(f"ffmpeg failed for {src_path.name}")


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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src",  type=Path, required=True)
    parser.add_argument("--dst",  type=Path, required=True)
    parser.add_argument("--speed-threshold",  type=float, default=0.01,
                        help="L2 joint speed below which a frame counts as paused (default: 0.01 rad/frame)")
    parser.add_argument("--min-pause-frames", type=int,   default=15,
                        help="Minimum consecutive pause frames before removal (default: 15 = 0.5s)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    src_data  = args.src / "data/chunk-000"
    src_video = args.src / "videos/chunk-000"
    src_meta  = args.src / "meta"

    ep_paths = sorted(src_data.glob("episode_*.parquet"))

    print(f"Source:           {args.src}")
    print(f"Destination:      {args.dst}")
    print(f"Speed threshold:  {args.speed_threshold} rad/frame")
    print(f"Min pause length: {args.min_pause_frames} frames ({args.min_pause_frames/FPS:.2f}s)")
    print(f"Episodes:         {len(ep_paths)}")
    print()

    if args.dry_run:
        total_in = total_rm = 0
        print(f"{'ep':>3}  {'frames':>6}  {'removed':>7}  {'pct':>5}  pause_runs")
        print("-" * 65)
        for p in ep_paths:
            ep  = int(p.stem.split("_")[1])
            df  = pd.read_parquet(p)
            act = np.array([np.asarray(v) for v in df["action"].values])
            keep = find_keep_mask(act, args.speed_threshold, args.min_pause_frames)
            n_rm = int((~keep).sum())
            total_in += len(df); total_rm += n_rm
            runs, m, i = [], ~keep, 0
            while i < len(m):
                if m[i]:
                    j = i
                    while j < len(m) and m[j]: j += 1
                    runs.append(f"fr{i}-{j-1}({j-i})")
                    i = j
                else:
                    i += 1
            run_str = " ".join(runs[:4]) + (f" +{len(runs)-4}more" if len(runs) > 4 else "")
            print(f"{ep:>3}  {len(df):>6}  {n_rm:>7}  {n_rm/len(df):>4.0%}  {run_str}")
        print(f"\nTotal: {total_in} → {total_in-total_rm} frames "
              f"({total_rm} removed, {total_rm/total_in:.1%})")
        return

    orig_stats_map = {}
    with open(src_meta / "episodes_stats.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            orig_stats_map[obj["episode_index"]] = {
                k: v for k, v in obj["stats"].items()
                if k in ("observation.images.front", "observation.images.wrist")
            }

    for cam in CAMERAS:
        (args.dst / "videos/chunk-000" / cam).mkdir(parents=True, exist_ok=True)
    (args.dst / "meta").mkdir(parents=True, exist_ok=True)

    global_offset = 0
    ep_meta_out, ep_stats_out = [], []
    total_in = total_rm = 0

    for p in ep_paths:
        ep     = int(p.stem.split("_")[1])
        dst_pq = args.dst / "data/chunk-000" / p.name

        print(f"ep{ep:02d}  parquet", end="", flush=True)
        n, n_rm, keep = process_parquet(p, dst_pq, ep, global_offset,
                                        args.speed_threshold, args.min_pause_frames)
        print(f" ({n} frames, -{n_rm} paused)", end="", flush=True)

        for cam in CAMERAS:
            src_vid = src_video / cam / f"episode_{ep:06d}.mp4"
            dst_vid = args.dst / "videos/chunk-000" / cam / f"episode_{ep:06d}.mp4"
            print(f"  {cam.split('.')[-1]}...", end="", flush=True)
            process_video(src_vid, dst_vid, keep)

        print(" ✓")

        df = pd.read_parquet(dst_pq)
        ep_meta_out.append({"episode_index": ep,
                            "tasks": ["Put the screwdriver into the cup"],
                            "length": n})
        ep_stats_out.append({"episode_index": ep,
                             "stats": compute_feature_stats(df, orig_stats_map.get(ep, {}))})
        global_offset += n
        total_in += n + n_rm
        total_rm += n_rm

    with open(src_meta / "info.json") as f:
        info = json.load(f)
    info["total_frames"] = global_offset
    dst_meta = args.dst / "meta"
    with open(dst_meta / "info.json", "w") as f:
        json.dump(info, f, indent=4)
    with open(dst_meta / "episodes.jsonl", "w") as f:
        for ep in ep_meta_out: f.write(json.dumps(ep) + "\n")
    with open(dst_meta / "episodes_stats.jsonl", "w") as f:
        for ep in ep_stats_out: f.write(json.dumps(ep) + "\n")
    shutil.copy(src_meta / "tasks.jsonl", dst_meta / "tasks.jsonl")

    print(f"\nDone: {global_offset} frames kept, {total_rm} removed "
          f"({total_rm/total_in:.1%}) → {args.dst}")


if __name__ == "__main__":
    main()
