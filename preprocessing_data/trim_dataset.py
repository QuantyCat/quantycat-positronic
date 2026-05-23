#!/usr/bin/env python3
"""Trim the start of every episode and optionally remove bad episodes.

Typical use: remove a countdown/hold phase recorded before demonstrations begin,
and drop any known-bad episodes (failed grasps, wrong trajectories, etc.).

Usage:
    python trim_dataset.py --src my_data/input_data --dst my_data/trimmed \
        --trim-frames 165

    # Also remove known bad episodes:
    python trim_dataset.py --src my_data/input_data --dst my_data/trimmed \
        --trim-frames 165 --remove-episodes 45 12

    # Dry run first (shows plan, no writes):
    python trim_dataset.py --src my_data/input_data --dst my_data/trimmed \
        --trim-frames 165 --dry-run

Notes:
    - Episodes are re-indexed contiguously after removal (ep46→45, etc.)
    - Parquet frame_index, timestamp, episode_index, and global index are all reset
    - Videos are trimmed frame-exact and re-encoded as H.264
    - All meta files (info.json, episodes.jsonl, episodes_stats.jsonl, tasks.jsonl)
      are regenerated
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


def trim_parquet(src_path, dst_path, new_ep_idx, global_offset, trim_frames):
    df = pd.read_parquet(src_path)
    if trim_frames >= len(df):
        raise ValueError(
            f"Cannot trim {trim_frames} frames from {src_path.name} "
            f"(only {len(df)} frames)"
        )
    df = df.iloc[trim_frames:].reset_index(drop=True)
    n  = len(df)

    df["frame_index"]   = np.arange(n, dtype=np.int64)
    df["timestamp"]     = (np.arange(n) / FPS).astype(np.float32)
    df["episode_index"] = np.int64(new_ep_idx)
    df["index"]         = np.arange(global_offset, global_offset + n, dtype=np.int64)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst_path, index=False)
    return n, df


def trim_video(src_path, dst_path, trim_frames):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src_path),
        "-vf", f"select='gte(n,{trim_frames})',setpts=PTS-STARTPTS",
        "-vsync", "0",
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
    parser.add_argument("--src",  type=Path, required=True,  help="Source dataset root")
    parser.add_argument("--dst",  type=Path, required=True,  help="Output dataset root")
    parser.add_argument("--trim-frames", type=int, default=0,
                        help="Frames to cut from the start of every episode (default: 0)")
    parser.add_argument("--remove-episodes", type=int, nargs="*", default=[],
                        metavar="N", help="Episode indices to drop entirely")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    remove_set = set(args.remove_episodes)
    src_data  = args.src / "data/chunk-000"
    src_video = args.src / "videos/chunk-000"
    src_meta  = args.src / "meta"

    all_eps   = sorted(int(p.stem.split("_")[1]) for p in src_data.glob("episode_*.parquet"))
    keep_eps  = [i for i in all_eps if i not in remove_set]

    print(f"Source:          {args.src}")
    print(f"Destination:     {args.dst}")
    print(f"Trim frames:     {args.trim_frames} ({args.trim_frames/FPS:.2f}s)")
    print(f"Remove episodes: {sorted(remove_set) or 'none'}")
    print(f"Episodes:        {len(all_eps)} in → {len(keep_eps)} out")
    print()

    if args.dry_run:
        print("DRY RUN — episode mapping:")
        for new_idx, orig_idx in enumerate(keep_eps):
            flag = " (re-indexed)" if orig_idx != new_idx else ""
            orig_len = len(pd.read_parquet(src_data / f"episode_{orig_idx:06d}.parquet"))
            print(f"  ep{orig_idx:02d} → ep{new_idx:02d}  {orig_len} → {orig_len - args.trim_frames} frames{flag}")
        return

    # Preserve original image stats (pixel values not in parquet)
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

    for new_idx, orig_idx in enumerate(keep_eps):
        src_pq = src_data / f"episode_{orig_idx:06d}.parquet"
        dst_pq = args.dst / "data/chunk-000" / f"episode_{new_idx:06d}.parquet"

        print(f"ep{orig_idx:02d}→{new_idx:02d}  parquet", end="", flush=True)
        n, df = trim_parquet(src_pq, dst_pq, new_idx, global_offset, args.trim_frames)
        print(f" ({n} frames)", end="", flush=True)

        for cam in CAMERAS:
            src_vid = src_video / cam / f"episode_{orig_idx:06d}.mp4"
            dst_vid = args.dst / "videos/chunk-000" / cam / f"episode_{new_idx:06d}.mp4"
            print(f"  {cam.split('.')[-1]}...", end="", flush=True)
            trim_video(src_vid, dst_vid, args.trim_frames)

        print(" ✓")

        ep_meta_out.append({"episode_index": new_idx,
                            "tasks": ["Put the screwdriver into the cup"],
                            "length": n})
        ep_stats_out.append({"episode_index": new_idx,
                             "stats": compute_feature_stats(df, orig_stats_map.get(orig_idx, {}))})
        global_offset += n

    # Regenerate meta
    with open(src_meta / "info.json") as f:
        info = json.load(f)
    info.update({"total_episodes": len(keep_eps),
                 "total_frames":   global_offset,
                 "total_videos":   len(keep_eps) * len(CAMERAS),
                 "splits":         {"train": f"0:{len(keep_eps)}"}})
    for cam in CAMERAS:
        info["features"][cam]["info"]["video.codec"] = "h264"

    dst_meta = args.dst / "meta"
    with open(dst_meta / "info.json", "w") as f:
        json.dump(info, f, indent=4)
    with open(dst_meta / "episodes.jsonl", "w") as f:
        for ep in ep_meta_out:
            f.write(json.dumps(ep) + "\n")
    with open(dst_meta / "episodes_stats.jsonl", "w") as f:
        for ep in ep_stats_out:
            f.write(json.dumps(ep) + "\n")
    shutil.copy(src_meta / "tasks.jsonl", dst_meta / "tasks.jsonl")

    print(f"\nDone: {len(keep_eps)} episodes, {global_offset} frames → {args.dst}")


if __name__ == "__main__":
    main()
