#!/usr/bin/env python3
"""Build a train split balanced for H1 centered motion on joints 1, 2, and 3.

This is a hypothesis-test split for the teleop fine-tuning workflow:
if targeted data reweighting fixed joint 3, then a bucketed H1 weighting scheme
over joints 1/2/3 should improve the weaker joints without sacrificing joint 3.

The weighting is materialized by stochastic resampling because the downstream
trainer consumes conversation records, not per-example sample weights.
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
DEFAULT_JOINT_PRIORITIES = {1: 1.6, 2: 1.6, 3: 1.0}
J3_RESTORE_JOINT_PRIORITIES = {1: 1.45, 2: 1.45, 3: 1.8}


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
    default_output = work_dir / "conversations" / f"libero_{label}_his_{his}_train_img_state_abs_ck_1_{resolution}_h1j123_balanced.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(config.get("training_output", "training_output"))
        / f"{label}_{robot}"
        / "weighting_report"
        / "h1j123_balanced_summary.json",
    )
    parser.add_argument("--action-stats", type=Path, default=work_dir / "min_max_action.txt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sign-eps", type=float, default=0.01)
    parser.add_argument("--repeat-eps", type=float, default=1e-6)
    parser.add_argument("--max-weight", type=float, default=7.0)
    parser.add_argument("--min-weight", type=float, default=0.10)
    parser.add_argument(
        "--recipe",
        choices=("balanced", "j3restore"),
        default="balanced",
        help="Weighting recipe. j3restore gives j3 high-motion examples additive pressure.",
    )
    parser.add_argument(
        "--joint-priorities",
        default=None,
        help="Comma-separated joint:priority overrides, for example '1:1.45,2:1.45,3:1.8'.",
    )
    return parser.parse_args()


def _parse_joint_priorities(raw: str | None, recipe: str) -> dict[int, float]:
    if raw is None:
        return dict(J3_RESTORE_JOINT_PRIORITIES if recipe == "j3restore" else DEFAULT_JOINT_PRIORITIES)
    result: dict[int, float] = {}
    for item in raw.split(","):
        joint_raw, priority_raw = item.split(":", 1)
        result[int(joint_raw.strip())] = float(priority_raw.strip())
    if not result:
        raise ValueError("--joint-priorities cannot be empty")
    return result


def _load_chunk(paths: list[str]) -> np.ndarray:
    return np.stack([np.load(path).astype(np.float32) for path in paths], axis=0)


def _load_zero_norm(path: Path, action_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    mins_arr = np.asarray(mins, dtype=np.float32)
    maxs_arr = np.asarray(maxs, dtype=np.float32)
    zero_norm = 2.0 * (0.0 - mins_arr) / np.maximum(maxs_arr - mins_arr, 1e-8) - 1.0
    return zero_norm.astype(np.float32), mins_arr, maxs_arr


def _normalize_action(action: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    return np.clip(2.0 * (action - mins) / (maxs - mins + 1e-8) - 1.0, -1.0, 1.0)


def _magnitude_bucket(centered_abs: float) -> str:
    if centered_abs <= 0.01:
        return "quiet"
    if centered_abs < 0.04:
        return "tiny"
    if centered_abs < 0.08:
        return "small"
    if centered_abs < 0.14:
        return "medium"
    if centered_abs < 0.18:
        return "large"
    if centered_abs < 0.24:
        return "xlarge"
    return "huge"


def _magnitude_weight(centered_abs: float) -> float:
    bucket = _magnitude_bucket(centered_abs)
    return {
        "quiet": 0.20,
        "tiny": 0.60,
        "small": 1.00,
        "medium": 1.60,
        "large": 2.30,
        "xlarge": 3.20,
        "huge": 4.20,
    }[bucket]


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


def _direction_weights(
    centered_h1: np.ndarray,
    sign_eps: float,
    joint_priorities: dict[int, float],
) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for joint in joint_priorities:
        values = centered_h1[:, joint]
        pos = int(np.sum(values > sign_eps))
        neg = int(np.sum(values < -sign_eps))
        max_count = max(pos, neg, 1)
        result[joint] = {
            "pos": min(max_count / max(pos, 1), 2.5),
            "neg": min(max_count / max(neg, 1), 2.5),
            "quiet": 0.35,
        }
    return result


def _direction(value: float, sign_eps: float) -> str:
    if value > sign_eps:
        return "pos"
    if value < -sign_eps:
        return "neg"
    return "quiet"


def _weight_record(
    conv: dict[str, Any],
    zero_norm: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    direction_weights: dict[int, dict[str, float]],
    joint_priorities: dict[int, float],
    recipe: str,
    sign_eps: float,
    repeat_eps: float,
    min_weight: float,
    max_weight: float,
) -> dict[str, Any]:
    chunk = _load_chunk(conv["action"])
    norm_chunk = _normalize_action(chunk, mins, maxs)
    centered = norm_chunk - zero_norm.reshape(1, -1)
    h1_centered = centered[0]
    arm_motion = float(np.mean(np.abs(chunk[:, 0:5])))

    joint_scores: dict[str, dict[str, float | str]] = {}
    weighted_scores = []
    max_score = 0.0
    priority_sum = 0.0
    priority_scores_by_joint: dict[int, float] = {}
    for joint, priority in joint_priorities.items():
        value = float(h1_centered[joint])
        centered_abs = abs(value)
        direction = _direction(value, sign_eps)
        score = _magnitude_weight(centered_abs) * direction_weights[joint][direction]
        priority_score = priority * score
        weighted_scores.append(priority_score)
        priority_scores_by_joint[joint] = priority_score
        max_score = max(max_score, priority_score)
        priority_sum += priority
        joint_scores[f"joint_{joint}"] = {
            "h1_centered": value,
            "h1_centered_abs": centered_abs,
            "direction": direction,
            "magnitude_bucket": _magnitude_bucket(centered_abs),
            "direction_weight": direction_weights[joint][direction],
            "score": score,
            "priority": priority,
            "priority_score": priority_score,
        }

    mean_score = float(sum(weighted_scores) / max(priority_sum, 1e-8))
    if recipe == "j3restore":
        # Keep broad j1/j2 coverage while making strong j3 examples win even
        # when they are paired with quiet or conflicting j1/j2 labels.
        j3_score = priority_scores_by_joint.get(3, 0.0)
        bucket_signal = 0.35 * mean_score + 0.35 * max_score + 0.30 * j3_score
        j3_abs = abs(float(h1_centered[3])) if h1_centered.shape[0] > 3 else 0.0
        if j3_abs >= 0.14:
            bucket_signal *= 1.15
        elif j3_abs >= 0.08:
            bucket_signal *= 1.08
    else:
        bucket_signal = 0.5 * mean_score + 0.5 * max_score
    motion_weight = _arm_motion_weight(arm_motion)
    repeat_weight = _repeated_chunk_factor(chunk, repeat_eps)
    final_weight = float(np.clip(bucket_signal * motion_weight * repeat_weight, min_weight, max_weight))
    return {
        "joint_scores": joint_scores,
        "arm_motion": arm_motion,
        "arm_motion_weight": motion_weight,
        "repeated_chunk_factor": repeat_weight,
        "bucket_signal": bucket_signal,
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


def _joint_distribution(
    centered_h1: np.ndarray,
    joint_priorities: dict[int, float],
    index_map: list[int] | None = None,
) -> dict[str, Any]:
    values_by_source = centered_h1 if index_map is None else centered_h1[index_map]
    result: dict[str, Any] = {}
    for joint in joint_priorities:
        values = values_by_source[:, joint]
        active = np.abs(values) > 0.01
        pos = int(np.sum(values[active] > 0))
        neg = int(np.sum(values[active] < 0))
        abs_values = np.abs(values)
        result[f"joint_{joint}"] = {
            "active_frac": float(np.mean(active)),
            "pos": pos,
            "neg": neg,
            "pos_frac": float(pos / max(pos + neg, 1)),
            "centered_abs_mean": float(np.mean(abs_values)),
            "centered_abs_p90": float(np.percentile(abs_values, 90)),
            "magnitude_buckets": {
                bucket: int(sum(_magnitude_bucket(float(v)) == bucket for v in abs_values))
                for bucket in ("quiet", "tiny", "small", "medium", "large", "xlarge", "huge")
            },
        }
    return result


def main() -> None:
    args = _parse_args()
    joint_priorities = _parse_joint_priorities(args.joint_priorities, args.recipe)
    rng = np.random.default_rng(args.seed)
    with args.input.open(encoding="utf-8") as f:
        convs: list[dict[str, Any]] = json.load(f)
    if not convs:
        raise ValueError(f"no conversations found in {args.input}")

    chunks = [_load_chunk(conv["action"]) for conv in convs]
    action_dim = int(chunks[0].shape[1])
    zero_norm, mins, maxs = _load_zero_norm(args.action_stats, action_dim)
    centered_h1 = np.stack(
        [_normalize_action(chunk, mins, maxs)[0] - zero_norm for chunk in chunks],
        axis=0,
    )
    dir_weights = _direction_weights(centered_h1, args.sign_eps, joint_priorities)
    metrics = [
        _weight_record(
            conv,
            zero_norm,
            mins,
            maxs,
            dir_weights,
            joint_priorities,
            args.recipe,
            args.sign_eps,
            args.repeat_eps,
            args.min_weight,
            args.max_weight,
        )
        for conv in convs
    ]
    weights = np.asarray([row["final_weight"] for row in metrics], dtype=np.float32)
    rebalanced, index_map, copies = _resample(convs, weights, rng)
    for out_idx, conv in enumerate(rebalanced):
        source_idx = index_map[out_idx]
        conv["sample_weight_metadata"] = dict(metrics[source_idx])
        conv["sample_weight_metadata"]["source_index"] = source_idx

    summary = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "action_stats": str(args.action_stats.resolve()),
        "seed": args.seed,
        "recipe": args.recipe,
        "sign_eps": args.sign_eps,
        "zero_norm": zero_norm.tolist(),
        "joint_priorities": joint_priorities,
        "direction_weights": dir_weights,
        "original_examples": len(convs),
        "rebalanced_examples": len(rebalanced),
        "unique_source_examples": len(set(index_map)),
        "weight_mean": float(np.mean(weights)),
        "weight_median": float(np.median(weights)),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "copies_min": int(np.min(copies)),
        "copies_max": int(np.max(copies)),
        "source_h1_centered_distribution": _joint_distribution(centered_h1, joint_priorities),
        "sampled_h1_centered_distribution": _joint_distribution(centered_h1, joint_priorities, index_map),
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
    for joint in ("joint_1", "joint_2", "joint_3"):
        source = summary["source_h1_centered_distribution"][joint]
        sampled = summary["sampled_h1_centered_distribution"][joint]
        print(
            f"{joint}: pos_frac {source['pos_frac']:.3f}->{sampled['pos_frac']:.3f} "
            f"abs_mean {source['centered_abs_mean']:.4f}->{sampled['centered_abs_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
