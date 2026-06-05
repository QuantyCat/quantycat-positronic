#!/usr/bin/env python3
"""Convert a LeRobot dataset between v2.1 and v3.0 formats.

v2.1: one parquet file per episode  → data/chunk-000/episode_000000.parquet
v3.0: all episodes in one file      → data/chunk-000/file-000.parquet

Usage:
    # v2.1 → v3.0 (consolidate):
    python convert_dataset_format.py --src my_data/clean_v2 --dst my_data/clean_v3 --to v3

    # v3.0 → v2.1 (split):
    python convert_dataset_format.py --src my_data/clean_v3 --dst my_data/clean_v2 --to v2

    # Preview without writing:
    python convert_dataset_format.py --src my_data/clean_v2 --dst my_data/clean_v3 --to v3 --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def detect_version(src: Path) -> str:
    with open(src / "meta/info.json") as f:
        info = json.load(f)
    return info["codebase_version"]


def _write_stats_json(dst: Path, src_meta: Path) -> None:
    """Aggregate per-episode stats into meta/stats.json for lerobot 0.5.1+."""
    import json as _json
    import numpy as np

    stats_jsonl = src_meta / "episodes_stats.jsonl"
    if not stats_jsonl.exists():
        return

    ep_stats = []
    with open(stats_jsonl) as f:
        for line in f:
            if line.strip():
                ep_stats.append(_json.loads(line)["stats"])
    if not ep_stats:
        return

    global_stats: dict = {}
    for key in ep_stats[0]:
        counts = np.array([float(s[key]["count"][0]) for s in ep_stats])
        means  = np.array([s[key]["mean"] for s in ep_stats], dtype=np.float64)
        stds   = np.array([s[key]["std"]  for s in ep_stats], dtype=np.float64)
        mins   = np.array([s[key]["min"]  for s in ep_stats], dtype=np.float64)
        maxs   = np.array([s[key]["max"]  for s in ep_stats], dtype=np.float64)
        w = counts / counts.sum()
        global_mean = (w[:, None] * means).sum(axis=0)
        global_std  = np.sqrt(np.maximum(
            (w[:, None] * (stds**2 + (means - global_mean)**2)).sum(axis=0), 0
        ))
        global_stats[key] = {
            "min":   mins.min(axis=0).tolist(),
            "max":   maxs.max(axis=0).tolist(),
            "mean":  global_mean.tolist(),
            "std":   global_std.tolist(),
            "count": [int(counts.sum())],
        }

    with open(dst / "meta" / "stats.json", "w") as f:
        _json.dump(global_stats, f, indent=2)


def _write_episodes_parquet(dst: Path, src_meta: Path, combined: "pd.DataFrame",
                            video_keys: list[str]) -> None:
    """Write meta/episodes/chunk-000/file-000.parquet for lerobot 0.5.1+."""
    # Build per-episode lengths from the consolidated DataFrame
    ep_lengths = combined.groupby("episode_index").size().to_dict()

    # Load tasks per episode from episodes.jsonl
    tasks_by_ep: dict[int, list[str]] = {}
    ep_jsonl = src_meta / "episodes.jsonl"
    if ep_jsonl.exists():
        with open(ep_jsonl) as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    tasks_by_ep[obj["episode_index"]] = obj.get("tasks", [])

    from_idx = 0
    rows = []
    for ep_idx in sorted(ep_lengths):
        length = ep_lengths[ep_idx]
        row: dict = {
            "episode_index": ep_idx,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": from_idx,
            "dataset_to_index": from_idx + length,
            "tasks": tasks_by_ep.get(ep_idx, []),
            "length": length,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for vk in video_keys:
            row[f"videos/{vk}/chunk_index"] = 0
            row[f"videos/{vk}/file_index"] = ep_idx
        rows.append(row)
        from_idx += length

    ep_dir = dst / "meta/episodes/chunk-000"
    ep_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(ep_dir / "file-000.parquet", index=False)


def convert_v2_to_v3(src: Path, dst: Path, dry_run: bool) -> None:
    src_data = src / "data/chunk-000"
    ep_paths = sorted(src_data.glob("episode_*.parquet"))

    if not ep_paths:
        raise FileNotFoundError(f"No episode parquets found in {src_data}")

    print(f"Source (v2.1):   {src}")
    print(f"Destination (v3.0): {dst}")
    print(f"Episodes:        {len(ep_paths)}")

    dfs = [pd.read_parquet(p) for p in ep_paths]
    combined = pd.concat(dfs, ignore_index=True)
    total_frames = len(combined)

    print(f"Total frames:    {total_frames}")

    if dry_run:
        print("\nDRY RUN — no files written.")
        return

    dst_data = dst / "data/chunk-000"
    dst_data.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(dst_data / "file-000.parquet", index=False)

    # Copy videos as-is
    src_videos = src / "videos/chunk-000"
    if src_videos.exists():
        dst_videos = dst / "videos/chunk-000"
        shutil.copytree(src_videos, dst_videos, dirs_exist_ok=True)

    # Update meta
    dst_meta = dst / "meta"
    dst_meta.mkdir(parents=True, exist_ok=True)

    with open(src / "meta/info.json") as f:
        info = json.load(f)
    video_keys = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]
    info["codebase_version"] = "v3.0"
    info["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    info["video_path"] = "videos/chunk-{chunk_index:03d}/{video_key}/episode_{file_index:06d}.mp4"
    info["total_frames"] = total_frames
    info.pop("total_videos", None)
    info.pop("total_chunks", None)
    with open(dst_meta / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    for fname in ("episodes.jsonl", "tasks.jsonl", "tasks.parquet", "episodes_stats.jsonl"):
        src_f = src / "meta" / fname
        if src_f.exists():
            shutil.copy2(src_f, dst_meta / fname)

    # Generate tasks.parquet from tasks.jsonl if parquet is absent
    tasks_parquet = dst_meta / "tasks.parquet"
    if not tasks_parquet.exists():
        tasks_jsonl = dst_meta / "tasks.jsonl"
        if tasks_jsonl.exists():
            import json as _json
            rows = [_json.loads(l) for l in tasks_jsonl.read_text().splitlines() if l.strip()]
            pd.DataFrame(rows).set_index("task_index").to_parquet(tasks_parquet)

    _write_episodes_parquet(dst, src / "meta", combined, video_keys)
    _write_stats_json(dst, src / "meta")

    print(f"\nDone → {dst}")


def convert_v3_to_v2(src: Path, dst: Path, dry_run: bool) -> None:
    src_parquets = sorted((src / "data").glob("chunk-*/file-*.parquet"))

    if not src_parquets:
        raise FileNotFoundError(f"No file-*.parquet found under {src / 'data'}")

    df = pd.concat([pd.read_parquet(p) for p in src_parquets], ignore_index=True)
    episodes = sorted(df["episode_index"].unique())

    print(f"Source (v3.0):      {src}")
    print(f"Destination (v2.1): {dst}")
    print(f"Episodes:           {len(episodes)}")
    print(f"Total frames:       {len(df)}")

    if dry_run:
        print("\nDRY RUN — no files written.")
        return

    dst_data = dst / "data/chunk-000"
    dst_data.mkdir(parents=True, exist_ok=True)

    for ep in episodes:
        ep_df = df[df["episode_index"] == ep].sort_values("frame_index")
        ep_df.to_parquet(dst_data / f"episode_{ep:06d}.parquet", index=False)

    # Copy videos as-is
    src_videos = src / "videos/chunk-000"
    if src_videos.exists():
        dst_videos = dst / "videos/chunk-000"
        shutil.copytree(src_videos, dst_videos, dirs_exist_ok=True)

    # Update meta
    dst_meta = dst / "meta"
    dst_meta.mkdir(parents=True, exist_ok=True)

    with open(src / "meta/info.json") as f:
        info = json.load(f)
    info["codebase_version"] = "v2.1"
    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    # v2.1 video_path uses episode_chunk/episode_index, not chunk_index/file_index
    info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    with open(dst_meta / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    for fname in ("episodes.jsonl", "tasks.jsonl", "tasks.parquet", "episodes_stats.jsonl"):
        src_f = src / "meta" / fname
        if src_f.exists():
            shutil.copy2(src_f, dst_meta / fname)

    print(f"\nDone → {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="Source dataset root")
    parser.add_argument("--dst", type=Path, required=True, help="Destination dataset root")
    parser.add_argument("--to", choices=["v2", "v3"], required=True,
                        help="Target format: v2 (one file per episode) or v3 (one consolidated file)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    src_version = detect_version(args.src)
    print(f"Detected source version: {src_version}\n")

    if args.to == "v3":
        if src_version != "v2.1":
            print(f"WARNING: source is already {src_version}, expected v2.1")
        convert_v2_to_v3(args.src, args.dst, args.dry_run)
    else:
        if src_version != "v3.0":
            print(f"WARNING: source is already {src_version}, expected v3.0")
        convert_v3_to_v2(args.src, args.dst, args.dry_run)


if __name__ == "__main__":
    main()
