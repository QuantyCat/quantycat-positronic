#!/usr/bin/env python3
"""Create episode-level train/heldout LeRobot dataset splits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import shutil
from typing import Any

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DEFAULT_SRC = REPO / "my_data/clean_input_data_achieved_delta"
DEFAULT_TRAIN = REPO / "my_data/clean_input_data_achieved_delta_train39"
DEFAULT_HELDOUT = REPO / "my_data/clean_input_data_achieved_delta_heldout10"
VIDEO_KEYS = ("observation.images.front", "observation.images.wrist")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--train-dst", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--heldout-dst", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--heldout-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument(
        "--copy-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How to copy parquet/video files into the split datasets.",
    )
    return parser.parse_args()


def _episode_index(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _copy_or_link(src: Path, dst: Path, copy_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == "hardlink":
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def _feature_stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    values = np.asarray(values)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return {
        "min": np.min(values, axis=0).astype(float).tolist(),
        "max": np.max(values, axis=0).astype(float).tolist(),
        "mean": np.mean(values, axis=0).astype(float).tolist(),
        "std": np.std(values, axis=0).astype(float).tolist(),
        "count": [int(values.shape[0])],
    }


def _episode_stats(df: pd.DataFrame) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for column in df.columns:
        first = df[column].iloc[0]
        if isinstance(first, np.ndarray):
            values = np.stack(df[column].to_numpy())
        else:
            values = df[column].to_numpy()
        stats[column] = _feature_stats(values)
    return stats


def _write_split(
    *,
    src: Path,
    dst: Path,
    original_episode_indices: list[int],
    copy_mode: str,
    split_name: str,
) -> list[dict[str, int]]:
    if dst.exists():
        raise FileExistsError(f"destination already exists: {dst}")

    (dst / "data/chunk-000").mkdir(parents=True)
    (dst / "meta").mkdir(parents=True)
    for video_key in VIDEO_KEYS:
        (dst / "videos/chunk-000" / video_key).mkdir(parents=True)

    tasks_src = src / "meta/tasks.jsonl"
    if tasks_src.is_file():
        _copy_or_link(tasks_src, dst / "meta/tasks.jsonl", copy_mode)

    mapping: list[dict[str, int]] = []
    episodes_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    global_index = 0

    for new_index, original_index in enumerate(original_episode_indices):
        src_parquet = src / "data/chunk-000" / f"episode_{original_index:06d}.parquet"
        if not src_parquet.is_file():
            raise FileNotFoundError(src_parquet)
        df = pd.read_parquet(src_parquet).copy()
        length = int(len(df))
        df["episode_index"] = new_index
        df["frame_index"] = np.arange(length, dtype=np.int64)
        df["index"] = np.arange(global_index, global_index + length, dtype=np.int64)
        global_index += length

        dst_parquet = dst / "data/chunk-000" / f"episode_{new_index:06d}.parquet"
        df.to_parquet(dst_parquet, index=False)

        for video_key in VIDEO_KEYS:
            src_video = src / "videos/chunk-000" / video_key / f"episode_{original_index:06d}.mp4"
            if not src_video.is_file():
                raise FileNotFoundError(src_video)
            dst_video = dst / "videos/chunk-000" / video_key / f"episode_{new_index:06d}.mp4"
            _copy_or_link(src_video, dst_video, copy_mode)

        episodes_rows.append(
            {
                "episode_index": new_index,
                "tasks": ["Put the screwdriver into the cup"],
                "length": length,
            }
        )
        stats_rows.append({"episode_index": new_index, "stats": _episode_stats(df)})
        mapping.append({"new_episode_index": new_index, "original_episode_index": original_index, "length": length})

    _write_jsonl(dst / "meta/episodes.jsonl", episodes_rows)
    _write_jsonl(dst / "meta/episodes_stats.jsonl", stats_rows)

    info = _load_json(src / "meta/info.json")
    info["total_episodes"] = len(original_episode_indices)
    info["total_frames"] = int(global_index)
    info["total_videos"] = int(len(original_episode_indices) * len(VIDEO_KEYS))
    info["splits"] = {"train": f"0:{len(original_episode_indices)}"}
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")

    manifest = {
        "source_dataset": str(src),
        "destination_dataset": str(dst),
        "split_name": split_name,
        "episode_count": len(original_episode_indices),
        "total_frames": int(global_index),
        "mapping": mapping,
    }
    report_dir = dst / "split_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# {split_name} Split",
        "",
        f"Source: `{src}`",
        f"Destination: `{dst}`",
        f"Episodes: {len(original_episode_indices)}",
        f"Frames: {global_index}",
        "",
        "| new episode | original episode | frames |",
        "|---:|---:|---:|",
    ]
    for row in mapping:
        lines.append(f"| {row['new_episode_index']} | {row['original_episode_index']} | {row['length']} |")
    (report_dir / "manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return mapping


def main() -> int:
    args = _parse_args()
    src = args.src.expanduser().resolve()
    train_dst = args.train_dst.expanduser().resolve()
    heldout_dst = args.heldout_dst.expanduser().resolve()

    if not src.is_dir():
        raise FileNotFoundError(src)
    if args.heldout_count <= 0:
        raise ValueError("--heldout-count must be > 0")

    episode_indices = sorted(_episode_index(path) for path in (src / "data/chunk-000").glob("episode_*.parquet"))
    if args.heldout_count >= len(episode_indices):
        raise ValueError("--heldout-count must be smaller than the episode count")

    rng = random.Random(args.seed)
    heldout = sorted(rng.sample(episode_indices, args.heldout_count))
    train = [episode for episode in episode_indices if episode not in set(heldout)]

    train_mapping = _write_split(
        src=src,
        dst=train_dst,
        original_episode_indices=train,
        copy_mode=args.copy_mode,
        split_name="train",
    )
    heldout_mapping = _write_split(
        src=src,
        dst=heldout_dst,
        original_episode_indices=heldout,
        copy_mode=args.copy_mode,
        split_name="heldout",
    )

    combined = {
        "source_dataset": str(src),
        "seed": args.seed,
        "heldout_count": args.heldout_count,
        "train_dataset": str(train_dst),
        "heldout_dataset": str(heldout_dst),
        "train_original_episodes": train,
        "heldout_original_episodes": heldout,
        "train_mapping": train_mapping,
        "heldout_mapping": heldout_mapping,
    }
    split_report = src / "split_report_train39_heldout10_seed20260525.json"
    split_report.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")

    print(f"source: {src}")
    print(f"train: {train_dst} episodes={len(train)}")
    print(f"heldout: {heldout_dst} episodes={len(heldout)}")
    print(f"heldout original episodes: {heldout}")
    print(f"split report: {split_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
