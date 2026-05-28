#!/usr/bin/env python3
"""Build a preserved OpenPI/LeRobot dataset variant with achieved arm deltas.

OpenPI's Quantycat config expects raw dataset actions to be absolute joint
targets. During training it applies DeltaActions, subtracting the current
observation state from every future action in the horizon. To train on achieved
motion instead of commanded targets, this script rewrites arm-joint actions as
the next achieved state:

    raw_action[k, 0:5] = observation.state[k + 1, 0:5]

Then OpenPI's horizon transform gives:

    transformed_action[t, h, 0:5] = state[t + h + 1, 0:5] - state[t, 0:5]

The gripper action is left in the original absolute-action convention.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DEFAULT_SRC = REPO / "my_data/clean_input_data"
DEFAULT_DST = REPO / "my_data/clean_input_data_achieved_delta"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    parser.add_argument(
        "--copy-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How to copy unchanged files before replacing parquet files.",
    )
    return parser.parse_args()


def _copy_dataset(src: Path, dst: Path, copy_mode: str) -> None:
    if not src.is_dir():
        raise FileNotFoundError(src)
    if dst.exists():
        raise FileExistsError(f"destination already exists: {dst}")

    if copy_mode == "hardlink":
        shutil.copytree(src, dst, copy_function=os.link)
    else:
        shutil.copytree(src, dst)


def _stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    return {
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
        "count": [int(values.shape[0])],
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _rewrite_episode(path: Path) -> dict[str, Any]:
    df = pd.read_parquet(path)
    states = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    old_actions = np.stack(df["action"].to_numpy()).astype(np.float32)

    new_actions = old_actions.copy()
    if len(states) > 1:
        new_actions[:-1, :5] = states[1:, :5]
        new_actions[-1, :5] = states[-1, :5]
    else:
        new_actions[:, :5] = states[:, :5]

    df = df.copy()
    df["action"] = list(new_actions)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)

    old_delta = old_actions[:, :5] - states[:, :5]
    new_delta = new_actions[:, :5] - states[:, :5]
    return {
        "episode_index": int(df["episode_index"].iloc[0]),
        "frames": int(len(df)),
        "old_action_abs_mean_deg": np.rad2deg(np.abs(old_delta).mean(axis=0)).astype(float).tolist(),
        "new_action_abs_mean_deg": np.rad2deg(np.abs(new_delta).mean(axis=0)).astype(float).tolist(),
        "old_j2_abs_mean_deg": float(np.rad2deg(np.abs(old_delta[:, 2]).mean())),
        "new_j2_abs_mean_deg": float(np.rad2deg(np.abs(new_delta[:, 2]).mean())),
        "old_j2_abs_p90_deg": float(np.rad2deg(np.percentile(np.abs(old_delta[:, 2]), 90))),
        "new_j2_abs_p90_deg": float(np.rad2deg(np.percentile(np.abs(new_delta[:, 2]), 90))),
        "new_action_stats": _stats(new_actions),
    }


def _update_episode_stats(dst: Path, summaries: list[dict[str, Any]]) -> None:
    stats_path = dst / "meta/episodes_stats.jsonl"
    if not stats_path.is_file():
        return

    by_episode = {row["episode_index"]: row for row in summaries}
    rows = _load_jsonl(stats_path)
    for row in rows:
        episode_index = int(row["episode_index"])
        if episode_index in by_episode:
            row["stats"]["action"] = by_episode[episode_index]["new_action_stats"]
    _write_jsonl(stats_path, rows)


def _write_report(dst: Path, src: Path, summaries: list[dict[str, Any]]) -> None:
    report_dir = dst / "achieved_delta_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    old_j2 = np.asarray([row["old_j2_abs_mean_deg"] for row in summaries], dtype=np.float32)
    new_j2 = np.asarray([row["new_j2_abs_mean_deg"] for row in summaries], dtype=np.float32)
    old_p90 = np.asarray([row["old_j2_abs_p90_deg"] for row in summaries], dtype=np.float32)
    new_p90 = np.asarray([row["new_j2_abs_p90_deg"] for row in summaries], dtype=np.float32)

    payload = {
        "source_dataset": str(src),
        "destination_dataset": str(dst),
        "episode_count": len(summaries),
        "convention": {
            "arm_joints_0_4": "raw action is next achieved observation.state; OpenPI DeltaActions converts to achieved horizon deltas",
            "gripper_5": "unchanged original absolute action",
        },
        "j2_summary_deg": {
            "old_abs_mean_episode_mean": float(old_j2.mean()),
            "new_abs_mean_episode_mean": float(new_j2.mean()),
            "old_abs_p90_episode_mean": float(old_p90.mean()),
            "new_abs_p90_episode_mean": float(new_p90.mean()),
        },
        "episodes": [
            {key: value for key, value in row.items() if key != "new_action_stats"}
            for row in sorted(summaries, key=lambda item: item["episode_index"])
        ],
    }

    json_path = report_dir / "summary.json"
    md_path = report_dir / "summary.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Achieved-Delta OpenPI Dataset",
        "",
        f"Source: `{src}`",
        f"Destination: `{dst}`",
        "",
        "Arm joint actions were rewritten to the next achieved state. Gripper actions were left unchanged.",
        "",
        "## J2 Delta Magnitudes",
        "",
        "| metric | old commanded-target labels | new achieved-state labels |",
        "|---|---:|---:|",
        f"| episode mean abs j2 delta | {old_j2.mean():.3f} deg | {new_j2.mean():.3f} deg |",
        f"| episode mean p90 abs j2 delta | {old_p90.mean():.3f} deg | {new_p90.mean():.3f} deg |",
        "",
        "## Largest Old-vs-New J2 Changes",
        "",
        "| episode | frames | old mean abs j2 | new mean abs j2 | old p90 abs j2 | new p90 abs j2 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    largest = sorted(summaries, key=lambda row: row["old_j2_abs_mean_deg"] - row["new_j2_abs_mean_deg"], reverse=True)[:15]
    for row in largest:
        lines.append(
            "| "
            f"{row['episode_index']} | {row['frames']} | "
            f"{row['old_j2_abs_mean_deg']:.3f} | {row['new_j2_abs_mean_deg']:.3f} | "
            f"{row['old_j2_abs_p90_deg']:.3f} | {row['new_j2_abs_p90_deg']:.3f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().resolve()

    _copy_dataset(src, dst, args.copy_mode)

    parquet_paths = sorted((dst / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"no parquet files found under {dst / 'data'}")

    summaries = [_rewrite_episode(path) for path in parquet_paths]
    _update_episode_stats(dst, summaries)
    _write_report(dst, src, summaries)

    old_j2 = np.mean([row["old_j2_abs_mean_deg"] for row in summaries])
    new_j2 = np.mean([row["new_j2_abs_mean_deg"] for row in summaries])
    print(f"wrote dataset: {dst}")
    print(f"episodes: {len(summaries)}")
    print(f"j2 mean abs delta: old={old_j2:.3f} deg new={new_j2:.3f} deg")
    print(f"report: {dst / 'achieved_delta_report/summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
