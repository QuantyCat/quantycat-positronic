#!/usr/bin/env python3
"""Build a train split weighted toward deployment first-action joint motion."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

CONFIG_PATH = Path("models/rynnvla-002/config.yaml")


def _load_config() -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        return {}
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_args() -> argparse.Namespace:
    config = _load_config()
    work_dir = Path(config.get("work_dir", "my_data/training_pipeline"))
    label = str(config.get("task_label", "screwdriver"))
    his = int(config.get("his", 1))
    resolution = int(config.get("resolution", 256))
    robot = str(config.get("robot", "so101"))
    default_input = work_dir / "conversations" / f"libero_{label}_his_{his}_train_img_state_abs_ck_1_{resolution}.json"
    default_output = work_dir / "conversations" / f"libero_{label}_his_{his}_train_img_state_abs_ck_1_{resolution}_h1j3_weighted.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(config.get("training_output", "training_output"))
        / f"{label}_{robot}"
        / "weighting_report"
        / "h1j3_weighted_summary.json",
    )
    parser.add_argument("--action-stats", type=Path, default=work_dir / "min_max_action.txt")
    parser.add_argument("--joint", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat-eps", type=float, default=1e-6)
    return parser.parse_args()


def _load_chunk(paths: list[str]) -> np.ndarray:
    return np.stack([np.load(path).astype(np.float32) for path in paths], axis=0)


def _load_zero_norm(path: Path, action_dim: int) -> np.ndarray:
    mins: list[float] = []
    maxs: list[float] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.startswith("Dim "):
                continue
            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 3:
                continue
            mins.append(float(parts[1]))
            maxs.append(float(parts[2]))
            if len(mins) == action_dim:
                break
    if len(mins) != action_dim:
        raise ValueError(f"expected {action_dim} action stat rows in {path}, got {len(mins)}")
    return np.asarray(
        [2.0 * (0.0 - mn) / max(mx - mn, 1e-8) - 1.0 for mn, mx in zip(mins, maxs)],
        dtype=np.float32,
    ), np.asarray(mins, dtype=np.float32), np.asarray(maxs, dtype=np.float32)


def _normalize_action(action: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    return np.clip(2.0 * (action - mins) / (maxs - mins + 1e-8) - 1.0, -1.0, 1.0)


def _h1_joint_weight(centered_abs: float) -> float:
    if centered_abs < 0.08:
        return 0.5
    if centered_abs < 0.14:
        return 1.0
    if centered_abs < 0.18:
        return 2.0
    if centered_abs < 0.24:
        return 3.0
    return 4.0


def _arm_motion_weight(arm_motion: float) -> float:
    if arm_motion < 0.018:
        return 0.5
    if arm_motion < 0.036:
        return 0.8
    if arm_motion < 0.049:
        return 1.0
    if arm_motion < 0.060:
        return 1.1
    return 1.25


def _repeated_chunk_factor(chunk: np.ndarray, eps: float) -> float:
    if float(np.max(np.abs(chunk - chunk[0:1]))) < eps:
        return 0.25
    return 1.0


def _weight_record(
    conv: dict[str, Any],
    zero_norm: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    joint: int,
    repeat_eps: float,
) -> dict[str, Any]:
    chunk = _load_chunk(conv["action"])
    norm_chunk = _normalize_action(chunk, mins, maxs)
    centered_h1_joint = float(norm_chunk[0, joint] - zero_norm[joint])
    centered_h1_joint_abs = abs(centered_h1_joint)
    arm_motion = float(np.mean(np.abs(chunk[:, 0:5])))
    h1_weight = _h1_joint_weight(centered_h1_joint_abs)
    arm_weight = _arm_motion_weight(arm_motion)
    repeat_weight = _repeated_chunk_factor(chunk, repeat_eps)
    final_weight = h1_weight * arm_weight * repeat_weight
    return {
        "joint": joint,
        "h1_joint_centered": centered_h1_joint,
        "h1_joint_centered_abs": centered_h1_joint_abs,
        "arm_motion": arm_motion,
        "h1_joint_weight": h1_weight,
        "arm_motion_weight": arm_weight,
        "repeated_chunk_factor": repeat_weight,
        "final_weight": final_weight,
    }


def _resample(
    convs: list[dict[str, Any]],
    weights: np.ndarray,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[int], np.ndarray]:
    normalized = weights / float(np.mean(weights))
    base = np.floor(normalized).astype(np.int32)
    frac = normalized - base
    copies = base + (rng.random(len(weights)) < frac).astype(np.int32)
    out: list[dict[str, Any]] = []
    index_map: list[int] = []
    for idx, n_copies in enumerate(copies.tolist()):
        for _ in range(n_copies):
            out.append(copy.deepcopy(convs[idx]))
            index_map.append(idx)
    return out, index_map, copies


def _bucket_counts(values: np.ndarray) -> dict[str, int]:
    return {
        "lt_0.08": int(np.sum(values < 0.08)),
        "0.08_to_0.14": int(np.sum((values >= 0.08) & (values < 0.14))),
        "0.14_to_0.18": int(np.sum((values >= 0.14) & (values < 0.18))),
        "0.18_to_0.24": int(np.sum((values >= 0.18) & (values < 0.24))),
        "ge_0.24": int(np.sum(values >= 0.24)),
    }


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)
    with args.input.open(encoding="utf-8") as f:
        convs: list[dict[str, Any]] = json.load(f)
    if not convs:
        raise ValueError(f"no conversations found in {args.input}")

    first_chunk = _load_chunk(convs[0]["action"])
    action_dim = int(first_chunk.shape[1])
    zero_norm, mins, maxs = _load_zero_norm(args.action_stats, action_dim)
    metrics = [
        _weight_record(conv, zero_norm, mins, maxs, args.joint, args.repeat_eps)
        for conv in convs
    ]
    weights = np.asarray([row["final_weight"] for row in metrics], dtype=np.float32)
    rebalanced, index_map, copies = _resample(convs, weights, rng)
    for out_idx, conv in enumerate(rebalanced):
        source_idx = index_map[out_idx]
        conv["sample_weight_metadata"] = dict(metrics[source_idx])
        conv["sample_weight_metadata"]["source_index"] = source_idx

    source_h1_abs = np.asarray([row["h1_joint_centered_abs"] for row in metrics], dtype=np.float32)
    sampled_h1_abs = source_h1_abs[index_map] if index_map else np.asarray([], dtype=np.float32)
    summary = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "action_stats": str(args.action_stats.resolve()),
        "seed": args.seed,
        "joint": args.joint,
        "zero_norm": zero_norm.tolist(),
        "original_examples": len(convs),
        "rebalanced_examples": len(rebalanced),
        "weight_mean": float(np.mean(weights)),
        "weight_median": float(np.median(weights)),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "copies_min": int(np.min(copies)),
        "copies_max": int(np.max(copies)),
        "source_h1_joint_centered_abs_mean": float(np.mean(source_h1_abs)),
        "sampled_h1_joint_centered_abs_mean": float(np.mean(sampled_h1_abs)),
        "source_h1_joint_centered_abs_p90": float(np.percentile(source_h1_abs, 90)),
        "sampled_h1_joint_centered_abs_p90": float(np.percentile(sampled_h1_abs, 90)),
        "source_h1_joint_centered_abs_ge_0.18_frac": float(np.mean(source_h1_abs >= 0.18)),
        "sampled_h1_joint_centered_abs_ge_0.18_frac": float(np.mean(sampled_h1_abs >= 0.18)),
        "source_h1_joint_centered_abs_buckets": _bucket_counts(source_h1_abs),
        "sampled_h1_joint_centered_abs_buckets": _bucket_counts(sampled_h1_abs),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(rebalanced, f, indent=2)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"wrote {args.output}")
    print(f"wrote {args.summary}")
    print(f"original_examples={len(convs)} rebalanced_examples={len(rebalanced)}")
    print(
        "h1_joint_centered_abs_ge_0.18 "
        f"source={summary['source_h1_joint_centered_abs_ge_0.18_frac']:.3f} "
        f"sampled={summary['sampled_h1_joint_centered_abs_ge_0.18_frac']:.3f}"
    )


if __name__ == "__main__":
    main()
