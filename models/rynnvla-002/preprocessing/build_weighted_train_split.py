#!/usr/bin/env python3
"""
Build a rebalanced training conversation file from chunk-level action weights.

This implements the weighting scheme requested for low-information action chunks:
  final_weight = motion_band_weight * gripper_bonus * repeated_chunk_factor

Because the downstream training path in this repo does not consume per-example
loss weights directly, this script materializes the weighting as a rebalanced
train split via stochastic resampling with replacement-free duplication:

  normalized_weight = final_weight / mean(final_weight)
  copies = floor(normalized_weight) + Bernoulli(frac(normalized_weight))

Expected dataset size stays close to the original train split.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

CONFIG_PATH = Path("models/rynnvla-002/config.yaml")


def _parse_args() -> argparse.Namespace:
    config = {}
    if CONFIG_PATH.is_file():
        with CONFIG_PATH.open(encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    work_dir = Path(config.get("work_dir", "my_data/training_pipeline"))
    label = str(config.get("task_label", "screwdriver"))
    his = int(config.get("his", 1))
    resolution = int(config.get("resolution", 256))
    robot = str(config.get("robot", "so101"))
    parser = argparse.ArgumentParser(description="Build weighted/rebalanced train conversations from action chunks.")
    parser.add_argument(
        "--input",
        type=Path,
        default=work_dir / "conversations" / f"libero_{label}_his_{his}_train_img_state_abs_ck_1_{resolution}.json",
        help="Input train conversation JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=work_dir / "conversations" / f"libero_{label}_his_{his}_train_img_state_abs_ck_1_{resolution}_weighted.json",
        help="Output rebalanced train conversation JSON.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(config.get("training_output", "training_output")) / f"{label}_{robot}" / "weighting_report" / "summary.json",
        help="Where to write a JSON summary of weights and resampling.",
    )
    parser.add_argument(
        "--repeat-eps",
        type=float,
        default=1e-6,
        help="Tolerance used to decide whether all sub-actions in a chunk are identical.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for stochastic resampling.",
    )
    return parser.parse_args()


def _load_chunk(paths: list[str]) -> np.ndarray:
    return np.stack([np.load(path).astype(np.float32) for path in paths], axis=0)


def _load_state(path: str) -> np.ndarray:
    return np.load(path).astype(np.float32)


def _motion_band_weight(arm_motion: float) -> float:
    if arm_motion < 0.018:
        return 0.15
    if arm_motion < 0.036:
        return 0.5
    if arm_motion < 0.049:
        return 1.0
    if arm_motion < 0.060:
        return 1.5
    return 2.0


def _gripper_bonus(grip_motion: float) -> float:
    if grip_motion >= 0.08:
        return 2.0
    if grip_motion >= 0.02:
        return 1.5
    return 1.0


def _repeated_chunk_factor(chunk: np.ndarray, eps: float) -> float:
    if float(np.max(np.abs(chunk - chunk[0:1]))) < eps:
        return 0.25
    return 1.0


def _weight_record(conv: dict[str, Any], eps: float) -> dict[str, Any]:
    chunk = _load_chunk(conv["action"])
    state = _load_state(conv["state"])
    arm_motion = float(np.mean(np.abs(chunk[:, 0:5])))
    grip_change_within_chunk = float(np.max(np.abs(np.diff(chunk[:, 5])))) if chunk.shape[0] > 1 else 0.0
    grip_offset_from_state = float(np.max(np.abs(chunk[:, 5] - state[5])))
    grip_motion = max(grip_change_within_chunk, grip_offset_from_state)
    motion_weight = _motion_band_weight(arm_motion)
    grip_weight = _gripper_bonus(grip_motion)
    repeat_weight = _repeated_chunk_factor(chunk, eps)
    final_weight = motion_weight * grip_weight * repeat_weight
    return {
        "arm_motion": arm_motion,
        "grip_motion": grip_motion,
        "grip_change_within_chunk": grip_change_within_chunk,
        "grip_offset_from_state": grip_offset_from_state,
        "motion_band_weight": motion_weight,
        "gripper_bonus": grip_weight,
        "repeated_chunk_factor": repeat_weight,
        "final_weight": final_weight,
    }


def _episode_and_step(conv: dict[str, Any]) -> tuple[str, int]:
    action0 = Path(conv["action"][0])
    episode = action0.parents[2].name
    step = int(action0.parents[0].name.split("_")[1])
    return episode, step


def _resample(convs: list[dict[str, Any]], weights: np.ndarray, rng: np.random.Generator) -> tuple[list[dict[str, Any]], list[int]]:
    normalized = weights / float(np.mean(weights))
    base = np.floor(normalized).astype(np.int32)
    frac = normalized - base
    extra = (rng.random(len(weights)) < frac).astype(np.int32)
    copies = base + extra

    out: list[dict[str, Any]] = []
    index_map: list[int] = []
    for idx, n_copies in enumerate(copies.tolist()):
        if n_copies <= 0:
            continue
        for _ in range(n_copies):
            out.append(copy.deepcopy(convs[idx]))
            index_map.append(idx)
    return out, index_map


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    with args.input.open(encoding="utf-8") as f:
        convs: list[dict[str, Any]] = json.load(f)

    metrics = [_weight_record(conv, args.repeat_eps) for conv in convs]
    weights = np.asarray([row["final_weight"] for row in metrics], dtype=np.float32)
    rebalanced, index_map = _resample(convs, weights, rng)

    for idx, conv in enumerate(rebalanced):
        source_idx = index_map[idx]
        conv["sample_weight_metadata"] = metrics[source_idx]
        conv["sample_weight_metadata"]["source_index"] = source_idx

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(rebalanced, f, indent=2)

    episodes = {}
    for idx, row in enumerate(metrics):
        episode, _step = _episode_and_step(convs[idx])
        episodes.setdefault(episode, {"count": 0, "weight_sum": 0.0})
        episodes[episode]["count"] += 1
        episodes[episode]["weight_sum"] += row["final_weight"]

    summary = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "seed": args.seed,
        "original_examples": len(convs),
        "rebalanced_examples": len(rebalanced),
        "weight_mean": float(np.mean(weights)),
        "weight_median": float(np.median(weights)),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "arm_motion_mean": float(np.mean([row["arm_motion"] for row in metrics])),
        "grip_motion_mean": float(np.mean([row["grip_motion"] for row in metrics])),
        "band_counts": {
            "lt_0.018": int(sum(row["motion_band_weight"] == 0.15 for row in metrics)),
            "0.018_to_0.036": int(sum(row["motion_band_weight"] == 0.5 for row in metrics)),
            "0.036_to_0.049": int(sum(row["motion_band_weight"] == 1.0 for row in metrics)),
            "0.049_to_0.060": int(sum(row["motion_band_weight"] == 1.5 for row in metrics)),
            "ge_0.060": int(sum(row["motion_band_weight"] == 2.0 for row in metrics)),
        },
        "gripper_bonus_counts": {
            "1.0": int(sum(row["gripper_bonus"] == 1.0 for row in metrics)),
            "1.5": int(sum(row["gripper_bonus"] == 1.5 for row in metrics)),
            "2.0": int(sum(row["gripper_bonus"] == 2.0 for row in metrics)),
        },
        "repeated_chunk_count": int(sum(row["repeated_chunk_factor"] == 0.25 for row in metrics)),
        "top_episodes_by_weight_mean": sorted(
            (
                {
                    "episode": episode,
                    "mean_weight": values["weight_sum"] / values["count"],
                    "num_chunks": values["count"],
                }
                for episode, values in episodes.items()
            ),
            key=lambda row: row["mean_weight"],
            reverse=True,
        )[:10],
    }

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"wrote {args.output}")
    print(f"wrote {args.summary}")
    print(f"original_examples={len(convs)} rebalanced_examples={len(rebalanced)}")
    print(f"weight_mean={summary['weight_mean']:.4f} weight_median={summary['weight_median']:.4f}")


if __name__ == "__main__":
    main()
