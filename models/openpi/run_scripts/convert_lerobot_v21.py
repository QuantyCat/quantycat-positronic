#!/usr/bin/env python3
"""Convert a LeRobot v3.0 dataset into the per-episode v2.1 layout OpenPI expects.

Usage:
    python models/openpi/data/convert_lerobot_v21.py \
        --source ~/quantycat-data/datasets/<raw_v3_dataset> \
        --target ~/quantycat-data/datasets/<output_v21_dataset>
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def _episode_stats(row: pd.Series) -> dict[str, Any]:
    stats: dict[str, dict[str, Any]] = {}
    for col, value in row.items():
        if not col.startswith("stats/"):
            continue
        _, feature, stat_name = col.split("/", 2)
        if feature.startswith("observation.images."):
            continue
        stats.setdefault(feature, {})[stat_name] = value
    return stats


def _rewrite_task(value: Any, task_text: str | None) -> Any:
    if task_text is None:
        return value
    if isinstance(value, list):
        return [task_text for _ in value]
    return task_text


def _task_list(value: Any, task_text: str | None) -> list[str]:
    if task_text is not None:
        return [task_text]
    if isinstance(value, str):
        return [value]
    return [str(task) for task in value]


def _parse_episode_filter(value: str | None) -> list[int] | None:
    if value is None:
        return None
    episodes: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            episodes.extend(range(int(start), int(end) + 1))
        else:
            episodes.append(int(part))
    return sorted(dict.fromkeys(episodes))


def _split_tables(
    source: Path,
    target: Path,
    task_text: str | None,
    episode_filter: list[int] | None,
    renumber: bool,
) -> pd.DataFrame:
    info_path = source / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    episodes = pd.read_parquet(source / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    data = pd.read_parquet(source / "data" / "chunk-000" / "file-000.parquet")
    tasks = pd.read_parquet(source / "meta" / "tasks.parquet").reset_index()
    if episode_filter is not None:
        episodes = episodes[episodes["episode_index"].isin(episode_filter)].copy()
        data = data[data["episode_index"].isin(episode_filter)].copy()
    episodes = episodes.sort_values("episode_index").reset_index(drop=True)
    episode_map = {
        int(row["episode_index"]): index if renumber else int(row["episode_index"])
        for index, row in episodes.iterrows()
    }

    v21_info = dict(info)
    v21_info["total_episodes"] = int(len(episodes))
    v21_info["total_frames"] = int(episodes["length"].sum())
    v21_info["splits"] = {"train": f"0:{len(episodes)}"}
    v21_info["codebase_version"] = "v2.1"
    v21_info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    v21_info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    v21_info["total_chunks"] = 1
    v21_info["total_videos"] = len([key for key, ft in info["features"].items() if ft["dtype"] == "video"])
    v21_info.pop("data_files_size_in_mb", None)
    v21_info.pop("video_files_size_in_mb", None)

    (target / "meta").mkdir(parents=True, exist_ok=True)
    (target / "meta" / "info.json").write_text(json.dumps(v21_info, indent=4) + "\n")
    shutil.copy2(source / "meta" / "stats.json", target / "meta" / "stats.json")

    _write_jsonl(
        target / "meta" / "tasks.jsonl",
        [
            {"task_index": int(row["task_index"]), "task": str(_rewrite_task(row["task"], task_text))}
            for _, row in tasks.sort_values("task_index").iterrows()
        ],
    )
    _write_jsonl(
        target / "meta" / "episodes.jsonl",
        [
            {
                "episode_index": int(episode_map[int(row["episode_index"])]),
                "tasks": _task_list(row["tasks"], task_text),
                "length": int(row["length"]),
                "source_episode_index": int(row["episode_index"]),
            }
            for _, row in episodes.sort_values("episode_index").iterrows()
        ],
    )
    _write_jsonl(
        target / "meta" / "episodes_stats.jsonl",
        [
            {
                "episode_index": int(episode_map[int(row["episode_index"])]),
                "source_episode_index": int(row["episode_index"]),
                "stats": _episode_stats(row),
            }
            for _, row in episodes.sort_values("episode_index").iterrows()
        ],
    )

    data_root = target / "data" / "chunk-000"
    data_root.mkdir(parents=True, exist_ok=True)
    next_index = 0
    converted_episodes: list[pd.Series] = []
    dataset_from_index = 0
    for _, row in episodes.sort_values("episode_index").iterrows():
        source_episode = int(row["episode_index"])
        target_episode = int(episode_map[source_episode])
        ep_df = data[data["episode_index"] == source_episode].copy()
        ep_df["episode_index"] = target_episode
        ep_df["index"] = range(next_index, next_index + len(ep_df))
        next_index += len(ep_df)
        ep_df.to_parquet(data_root / f"episode_{target_episode:06d}.parquet", index=False)

        converted = row.copy()
        converted["source_episode_index"] = source_episode
        converted["source_dataset_from_index"] = int(row["dataset_from_index"])
        converted["source_dataset_to_index"] = int(row["dataset_to_index"])
        converted["episode_index"] = target_episode
        converted["dataset_from_index"] = dataset_from_index
        dataset_from_index += int(row["length"])
        converted["dataset_to_index"] = dataset_from_index
        converted_episodes.append(converted)

    return pd.DataFrame(converted_episodes)


def _split_videos(source: Path, target: Path, episodes: pd.DataFrame, fps: int) -> None:
    video_keys = ("observation.images.front", "observation.images.wrist")
    total = len(episodes)

    for video_key in video_keys:
        source_video = source / "videos" / video_key / "chunk-000" / "file-000.mp4"
        target_dir = target / "videos" / "chunk-000" / video_key
        target_dir.mkdir(parents=True, exist_ok=True)

        for i, (_, row) in enumerate(episodes.sort_values("episode_index").iterrows(), start=1):
            episode_index = int(row["episode_index"])
            start_frame = int(row.get("source_dataset_from_index", row["dataset_from_index"]))
            end_frame = int(row.get("source_dataset_to_index", row["dataset_to_index"]))
            output_path = target_dir / f"episode_{episode_index:06d}.mp4"
            print(f"[{video_key}] {i}/{total} episode_{episode_index:06d}", flush=True)
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source_video),
                    "-vf",
                    f"trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    "-bf",
                    "0",
                    "-pix_fmt",
                    "yuv420p",
                    "-r",
                    str(fps),
                    str(output_path),
                ],
                check=True,
            )

        expected = len(episodes)
        actual = len(list(target_dir.glob("episode_*.mp4")))
        if actual != expected:
            raise RuntimeError(f"{video_key}: expected {expected} split videos, found {actual}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="LeRobot v3.0 dataset to convert.")
    parser.add_argument("--target", type=Path, required=True, help="Output path for the v2.1 dataset.")
    parser.add_argument("--task-text", default=None, help="Override task text in generated tasks/episodes metadata.")
    parser.add_argument(
        "--episodes",
        default=None,
        help="Optional comma-separated episode filter. Ranges are inclusive, e.g. 0-59,70-87.",
    )
    parser.add_argument("--renumber", action="store_true", help="Renumber filtered episodes to contiguous ids from 0.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.target.exists():
        if not args.overwrite:
            raise SystemExit(f"Target exists: {args.target}. Use --overwrite to replace it.")
        shutil.rmtree(args.target)

    info = json.loads((args.source / "meta" / "info.json").read_text())
    print(f"Converting {args.source} -> {args.target}")
    print("Splitting episode tables...")
    episodes = _split_tables(args.source, args.target, args.task_text, _parse_episode_filter(args.episodes), args.renumber)
    print(f"Splitting {len(episodes)} episodes x 2 cameras into per-episode videos...")
    _split_videos(args.source, args.target, episodes, fps=int(info["fps"]))
    print(f"Created {args.target}")


if __name__ == "__main__":
    main()
