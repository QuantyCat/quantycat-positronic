#!/usr/bin/env python3
"""
Audit RynnVLA action min/max normalization against saved training chunks.

This is CPU-only and checks:
  - stats file min/max vs empirical min/max
  - normalize -> unnormalize round-trip
  - where raw zero maps in normalized [-1, 1] space
  - what normalized zero unnormalizes to
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

JOINT_LABELS = ("joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "gripper")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit action normalization stats.")
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=Path("my_data/training_pipeline/training_data/Put_the_screwdriver_into_the_cup"),
    )
    parser.add_argument(
        "--stats-file",
        type=Path,
        default=Path("my_data/training_pipeline/min_max_action.txt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_output/screwdriver_so101/data_analysis/action_stats_audit/summary.json"),
    )
    parser.add_argument("--sample-limit", type=int, default=1000)
    return parser.parse_args()


def _parse_stats_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mins: list[float] = []
    maxs: list[float] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if "|" not in line:
                continue
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
            if len(nums) < 5:
                continue
            mins.append(float(nums[1]))
            maxs.append(float(nums[2]))
    if len(mins) != len(JOINT_LABELS):
        raise SystemExit(f"expected {len(JOINT_LABELS)} stat rows in {path}, got {len(mins)}")
    return np.asarray(mins, dtype=np.float32), np.asarray(maxs, dtype=np.float32)


def _iter_action_files(task_dir: Path) -> list[Path]:
    return sorted(task_dir.glob("episode_*/abs_action/action_*/*.npy"))


def _load_actions(paths: list[Path]) -> np.ndarray:
    actions = []
    for path in paths:
        arr = np.load(path).astype(np.float32)
        if arr.shape == (len(JOINT_LABELS),):
            actions.append(arr)
    if not actions:
        raise SystemExit("no valid action files found")
    return np.stack(actions, axis=0)


def _norm(action: np.ndarray, min_values: np.ndarray, max_values: np.ndarray) -> np.ndarray:
    return np.clip(2.0 * (action - min_values) / (max_values - min_values + 1e-8) - 1.0, -1.0, 1.0)


def _unnorm(norm_action: np.ndarray, min_values: np.ndarray, max_values: np.ndarray) -> np.ndarray:
    clipped = np.clip(norm_action, -1.0, 1.0)
    return np.clip((clipped + 1.0) / 2.0 * (max_values - min_values + 1e-8) + min_values, min_values, max_values)


def _joint_dict(values: np.ndarray) -> dict[str, float]:
    return {name: float(values[i]) for i, name in enumerate(JOINT_LABELS)}


def main() -> None:
    args = _parse_args()
    stats_min, stats_max = _parse_stats_file(args.stats_file)
    all_paths = _iter_action_files(args.task_dir)
    if not all_paths:
        raise SystemExit(f"no action files found under {args.task_dir}")

    actions = _load_actions(all_paths)
    sample = actions[: max(1, min(args.sample_limit, len(actions)))]
    normed = _norm(sample, stats_min, stats_max)
    recovered = _unnorm(normed, stats_min, stats_max)
    roundtrip_abs = np.abs(recovered - sample)

    zero = np.zeros(len(JOINT_LABELS), dtype=np.float32)
    norm_zero = _norm(zero, stats_min, stats_max)
    midpoint = _unnorm(zero, stats_min, stats_max)

    report: dict[str, Any] = {
        "task_dir": str(args.task_dir.resolve()),
        "stats_file": str(args.stats_file.resolve()),
        "num_action_files": len(all_paths),
        "stats_min": _joint_dict(stats_min),
        "stats_max": _joint_dict(stats_max),
        "empirical_min": _joint_dict(np.min(actions, axis=0)),
        "empirical_max": _joint_dict(np.max(actions, axis=0)),
        "empirical_mean": _joint_dict(np.mean(actions, axis=0)),
        "empirical_median": _joint_dict(np.median(actions, axis=0)),
        "roundtrip_max_abs_error": float(np.max(roundtrip_abs)),
        "roundtrip_mean_abs_error": float(np.mean(roundtrip_abs)),
        "raw_zero_as_normalized": _joint_dict(norm_zero),
        "normalized_zero_as_raw_midpoint": _joint_dict(midpoint),
        "normalized_empirical_mean": _joint_dict(np.mean(_norm(actions, stats_min, stats_max), axis=0)),
        "normalized_empirical_median": _joint_dict(np.median(_norm(actions, stats_min, stats_max), axis=0)),
    }

    print("Action Stats Audit")
    print(f"actions={len(all_paths)}")
    print(f"roundtrip_max_abs_error={report['roundtrip_max_abs_error']:.8f}")
    print("raw zero as normalized [-1,1]:")
    for name, value in report["raw_zero_as_normalized"].items():
        print(f"  {name:<8} {value:.4f}")
    print("normalized zero as raw midpoint:")
    for name, value in report["normalized_zero_as_raw_midpoint"].items():
        print(f"  {name:<8} {value:.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
