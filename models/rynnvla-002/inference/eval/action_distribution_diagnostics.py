#!/usr/bin/env python3
"""
CPU-only diagnostics for low-motion RynnVLA action behavior.

This script scans saved RynnVLA-style action chunks:

    task_dir/episode_XXXXXX/abs_action/action_T/0.npy ... N.npy

It summarizes the target distribution, compares optional predicted live deltas
against the dataset, and prints high-motion windows suitable for batch eval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

JOINT_LABELS = ("joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "gripper")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose low-motion action chunk distributions.")
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=Path("my_data/training_pipeline/training_data/Put_the_screwdriver_into_the_cup"),
        help="Directory containing episode_* folders.",
    )
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--static-threshold",
        type=float,
        default=0.01,
        help="Mean absolute arm motion below this is counted as quasi-static.",
    )
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=0.01,
        help="Absolute gripper movement below this is counted as near-zero.",
    )
    parser.add_argument(
        "--pred-json",
        type=Path,
        default=None,
        help="Optional JSON file containing predicted chunks or deltas to compare.",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=Path("training_output/screwdriver_so101/action_distribution_diagnostics/summary.json"),
        help="Where to write the JSON summary.",
    )
    return parser.parse_args()


def _load_chunk(action_dir: Path) -> np.ndarray:
    files = sorted(action_dir.glob("*.npy"), key=lambda p: int(p.stem))
    if not files:
        raise ValueError(f"no .npy files under {action_dir}")
    return np.asarray([np.load(p).astype(np.float32) for p in files], dtype=np.float32)


def _iter_action_dirs(task_dir: Path) -> list[Path]:
    episodes = sorted(p for p in task_dir.iterdir() if p.is_dir() and p.name.startswith("episode_"))
    action_dirs: list[Path] = []
    for episode in episodes:
        root = episode / "abs_action"
        if not root.is_dir():
            continue
        action_dirs.extend(sorted(root.glob("action_*"), key=lambda p: int(p.name.split("_")[1])))
    return [p for p in action_dirs if p.is_dir()]


def _load_all_chunks(task_dir: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    chunks: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    for action_dir in _iter_action_dirs(task_dir):
        try:
            chunk = _load_chunk(action_dir)
        except Exception as exc:
            print(f"warning: skipping {action_dir}: {exc}")
            continue
        if chunk.ndim != 2 or chunk.shape[1] != len(JOINT_LABELS):
            print(f"warning: skipping {action_dir}: expected shape (T, 6), got {chunk.shape}")
            continue
        chunks.append(chunk)
        meta.append(
            {
                "episode": action_dir.parents[1].name,
                "step": int(action_dir.name.split("_")[1]),
                "path": str(action_dir),
            }
        )
    if not chunks:
        raise SystemExit(f"no chunks found under {task_dir}")
    return np.stack(chunks, axis=0), meta


def _percentiles(values: np.ndarray, qs: tuple[float, ...] = (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)) -> dict[str, float]:
    return {f"p{q:g}": float(np.percentile(values, q)) for q in qs}


def _joint_dict(values: np.ndarray) -> dict[str, float]:
    return {name: float(values[i]) for i, name in enumerate(JOINT_LABELS[: len(values)])}


def _summarize(chunks: np.ndarray, static_threshold: float, gripper_threshold: float) -> dict[str, Any]:
    flat = chunks.reshape(-1, chunks.shape[-1])
    arm_abs_step = np.mean(np.abs(chunks[:, :, :-1]), axis=(1, 2))
    all_abs_step = np.mean(np.abs(chunks), axis=(1, 2))
    chunk_max_arm = np.max(np.abs(chunks[:, :, :-1]), axis=(1, 2))
    within_chunk_max_change = np.max(np.abs(np.diff(chunks, axis=1)), axis=(1, 2))
    repeated_exact = np.all(np.isclose(chunks, chunks[:, :1, :], atol=0.0, rtol=0.0), axis=(1, 2))
    gripper_abs = np.abs(flat[:, -1])

    active = arm_abs_step >= static_threshold
    high_motion = arm_abs_step >= np.percentile(arm_abs_step, 75)

    return {
        "num_chunks": int(chunks.shape[0]),
        "chunk_size": int(chunks.shape[1]),
        "action_dim": int(chunks.shape[2]),
        "mean_abs_by_joint": _joint_dict(np.mean(np.abs(flat), axis=0)),
        "near_zero_fraction_by_joint_abs_lt_0.01": _joint_dict(np.mean(np.abs(flat) < 0.01, axis=0)),
        "arm_mean_abs_per_chunk_percentiles": _percentiles(arm_abs_step),
        "all_mean_abs_per_chunk_percentiles": _percentiles(all_abs_step),
        "arm_max_abs_per_chunk_percentiles": _percentiles(chunk_max_arm),
        "within_chunk_max_change_percentiles": _percentiles(within_chunk_max_change),
        "exactly_repeated_chunk_fraction": float(np.mean(repeated_exact)),
        "quasi_static_chunk_fraction": float(np.mean(arm_abs_step < static_threshold)),
        "active_chunk_fraction": float(np.mean(active)),
        "top_quartile_motion_fraction": float(np.mean(high_motion)),
        "gripper_abs_percentiles": _percentiles(gripper_abs),
        "gripper_near_zero_fraction": float(np.mean(gripper_abs < gripper_threshold)),
    }


def _top_windows(chunks: np.ndarray, meta: list[dict[str, Any]], window_size: int, top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_episode: dict[str, list[int]] = {}
    for idx, row in enumerate(meta):
        by_episode.setdefault(row["episode"], []).append(idx)

    scores = np.mean(np.abs(chunks[:, :, :-1]), axis=(1, 2))
    joint_means = np.mean(np.abs(chunks), axis=1)
    for episode, indices in by_episode.items():
        indices = sorted(indices, key=lambda i: meta[i]["step"])
        if len(indices) < window_size:
            continue
        for start_i in range(0, len(indices) - window_size + 1):
            ids = indices[start_i : start_i + window_size]
            rows.append(
                {
                    "episode": episode,
                    "start_step": int(meta[ids[0]]["step"]),
                    "end_step": int(meta[ids[-1]]["step"]),
                    "score": float(np.mean(scores[ids])),
                    "joint_means": _joint_dict(np.mean(joint_means[ids], axis=0)),
                }
            )
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:top_k]


def _load_pred_json(path: Path) -> np.ndarray:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        for key in ("predicted_actions", "predictions", "deltas", "actions"):
            if key in data:
                data = data[key]
                break
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3 or arr.shape[-1] != len(JOINT_LABELS):
        raise SystemExit(f"--pred-json must resolve to shape (N,T,6), (T,6), or (6,), got {arr.shape}")
    return arr


def _compare_predictions(chunks: np.ndarray, pred_chunks: np.ndarray) -> dict[str, Any]:
    train_arm = np.mean(np.abs(chunks[:, :, :-1]), axis=(1, 2))
    pred_arm = np.mean(np.abs(pred_chunks[:, :, :-1]), axis=(1, 2))
    train_flat = chunks.reshape(-1, chunks.shape[-1])
    pred_flat = pred_chunks.reshape(-1, pred_chunks.shape[-1])
    return {
        "pred_chunk_count": int(pred_chunks.shape[0]),
        "pred_mean_abs_by_joint": _joint_dict(np.mean(np.abs(pred_flat), axis=0)),
        "pred_arm_mean_abs_percentiles": _percentiles(pred_arm),
        "pred_arm_mean_abs_as_train_percentile": [
            float(np.mean(train_arm <= value) * 100.0) for value in pred_arm[:50]
        ],
        "pred_over_train_mean_abs_ratio_by_joint": _joint_dict(
            np.mean(np.abs(pred_flat), axis=0) / (np.mean(np.abs(train_flat), axis=0) + 1e-8)
        ),
    }


def _print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("Action Distribution Diagnostics")
    print(f"chunks={summary['num_chunks']}  chunk_size={summary['chunk_size']}  action_dim={summary['action_dim']}")
    print()
    print("Mean absolute target by joint:")
    for name, value in summary["mean_abs_by_joint"].items():
        print(f"  {name:<8} {value:.6f}")
    print()
    print("Arm mean-abs per chunk percentiles:")
    for key, value in summary["arm_mean_abs_per_chunk_percentiles"].items():
        print(f"  {key:<5} {value:.6f}")
    print()
    print(f"quasi_static_fraction={summary['quasi_static_chunk_fraction']:.3f}")
    print(f"exactly_repeated_chunk_fraction={summary['exactly_repeated_chunk_fraction']:.3f}")
    print(f"gripper_near_zero_fraction={summary['gripper_near_zero_fraction']:.3f}")
    print()
    print("Top high-motion eval windows:")
    for row in report["top_windows"]:
        joint_text = " ".join(f"{k}={v:.4f}" for k, v in row["joint_means"].items())
        print(f"  {row['episode']} steps {row['start_step']}-{row['end_step']} score={row['score']:.4f} {joint_text}")
    if "prediction_comparison" in report:
        print()
        print("Prediction comparison:")
        comp = report["prediction_comparison"]
        print("  pred mean abs by joint:")
        for name, value in comp["pred_mean_abs_by_joint"].items():
            print(f"    {name:<8} {value:.6f}")
        print("  pred/train mean abs ratio by joint:")
        for name, value in comp["pred_over_train_mean_abs_ratio_by_joint"].items():
            print(f"    {name:<8} {value:.3f}")


def main() -> None:
    args = _parse_args()
    chunks, meta = _load_all_chunks(args.task_dir.resolve())
    report: dict[str, Any] = {
        "task_dir": str(args.task_dir.resolve()),
        "summary": _summarize(chunks, args.static_threshold, args.gripper_threshold),
        "top_windows": _top_windows(chunks, meta, args.window_size, args.top_k),
    }
    if args.pred_json:
        report["prediction_comparison"] = _compare_predictions(chunks, _load_pred_json(args.pred_json))
    _print_summary(report)
    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(report, indent=2))
        print()
        print(f"wrote {args.save_json}")


if __name__ == "__main__":
    main()
