#!/usr/bin/env python3
"""
Rank high-motion windows from saved action chunks and print batch-eval commands.

This helper scans saved `abs_action/action_<step>/<chunk_pos>.npy` files and finds
episode windows with the largest average action magnitude. It is useful for picking
harder evaluation slices than low-motion near-static regions.

The saved `abs_action` files are already target deltas for joints 0-4, with an
absolute gripper target. Do not subtract state when ranking these chunks.

Examples:

  python models/rynnvla-002/eval/data_analysis/find_high_motion_windows.py \
    --episode my_data/training_pipeline/training_data/Put_the_screwdriver_into_the_cup/episode_000025

  python models/rynnvla-002/eval/data_analysis/find_high_motion_windows.py \
    --task-dir my_data/training_pipeline/training_data/Put_the_screwdriver_into_the_cup \
    --metric arm \
    --top-k 10
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model_eval"))
import episode_step_eval as step_eval

_DEFAULT_CONFIG = "models/rynnvla-002/config.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find high-motion evaluation windows from saved action chunks.")
    parser.add_argument("--episode", type=str, default=None, help="Path to one episode directory.")
    parser.add_argument("--task-dir", type=str, default=None, help="Path to a task directory containing episode_* folders.")
    parser.add_argument(
        "--metric",
        choices=("all", "arm", "gripper"),
        default="arm",
        help="Motion score to rank by: all joints, arm joints only, or gripper only. Ignored when --joint is set.",
    )
    parser.add_argument("--joint", type=int, default=None, help="Rank windows by a single joint index (0-4). Overrides --metric.")
    parser.add_argument("--window-size", type=int, default=50, help="Number of steps per ranked window.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top windows to print.")
    parser.add_argument("--save-windows", type=str, default=None, help="Save selected windows to this JSON path.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint override for printed commands.")
    parser.add_argument("--positronic-config", type=str, default=_DEFAULT_CONFIG)
    parser.add_argument(
        "--rynnvla-repo",
        type=str,
        default=os.environ.get("RYNNVLA_REPO", ""),
        help="Optional path to the RynnVLA python root for printed commands.",
    )
    return parser.parse_args()


def _load_chunk(action_dir: Path) -> np.ndarray:
    files = sorted(
        [p for p in action_dir.iterdir() if p.is_file() and p.suffix == ".npy"],
        key=lambda p: int(p.stem),
    )
    if not files:
        raise ValueError(f"no action files found in {action_dir}")
    return np.asarray([np.load(p).astype(np.float32) for p in files], dtype=np.float32)


def _sorted_action_dirs(action_root: Path) -> list[Path]:
    return sorted(
        [p for p in action_root.iterdir() if p.is_dir() and p.name.startswith("action_")],
        key=lambda p: int(p.name.split("_")[1]),
    )


def _load_episode_chunks(episode_dir: Path) -> np.ndarray:
    action_root = episode_dir / "abs_action"
    if not action_root.is_dir():
        raise FileNotFoundError(f"missing directory: {action_root}")
    action_dirs = _sorted_action_dirs(action_root)
    if not action_dirs:
        raise ValueError(f"no action directories found in {action_root}")
    return np.stack([_load_chunk(d) for d in action_dirs], axis=0)


def _step_scores(chunks: np.ndarray, metric: str, joint: int | None = None) -> np.ndarray:
    if joint is not None:
        return np.mean(np.abs(chunks[:, :, joint]), axis=1)
    if metric == "all":
        data = chunks
    elif metric == "arm":
        data = chunks[:, :, :-1]
    elif metric == "gripper":
        data = chunks[:, :, -1:]
    else:
        raise ValueError(f"unknown metric: {metric}")
    return np.mean(np.abs(data), axis=(1, 2))


def _joint_means(chunks: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(chunks), axis=1)


def _scan_episode(episode_dir: Path, metric: str, window_size: int, joint: int | None = None) -> list[dict[str, Any]]:
    chunks = _load_episode_chunks(episode_dir)
    if len(chunks) < window_size:
        return []

    scores = _step_scores(chunks, metric, joint)
    joint_means = _joint_means(chunks)
    rows: list[dict[str, Any]] = []
    for start in range(0, len(chunks) - window_size + 1):
        stop = start + window_size
        sl = slice(start, stop)
        rows.append(
            {
                "episode_dir": episode_dir,
                "episode_name": episode_dir.name,
                "start": start,
                "end": stop - 1,
                "score": float(scores[sl].mean()),
                "joint_means": joint_means[sl].mean(axis=0),
            }
        )
    return rows


def _resolve_task_dir(args: argparse.Namespace) -> Path:
    if args.task_dir:
        return Path(args.task_dir).expanduser().resolve()
    if args.episode:
        return Path(args.episode).expanduser().resolve().parent
    raise ValueError("Pass either --episode or --task-dir")


def _resolve_episode_dirs(args: argparse.Namespace) -> list[Path]:
    if args.episode:
        return [Path(args.episode).expanduser().resolve()]
    task_dir = Path(args.task_dir).expanduser().resolve()
    return sorted([p for p in task_dir.iterdir() if p.is_dir() and p.name.startswith("episode_")])


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _batch_eval_command(
    row: dict[str, Any],
    root: Path,
    ckpt_path: Path,
    rynnvla_repo: str,
) -> str:
    rel_episode = os.path.relpath(row["episode_dir"], root)
    parts = [
        "bash models/rynnvla-002/run_scripts/model_eval.sh",
        f"--episode {_shell_quote(rel_episode)}",
        f"--checkpoint {_shell_quote(str(ckpt_path))}",
        f"--start-step {row['start']}",
        f"--max-steps {row['end'] - row['start'] + 1}",
    ]
    if rynnvla_repo:
        parts.append(f"--rynnvla-repo {_shell_quote(rynnvla_repo)}")
    return " \\\n  ".join(parts)


def main() -> None:
    args = _parse_args()
    root = step_eval._repo_root()
    os.chdir(root)

    if not args.episode and not args.task_dir:
        raise SystemExit("Pass either --episode PATH or --task-dir PATH")
    if args.episode and args.task_dir:
        raise SystemExit("Pass only one of --episode or --task-dir")
    if args.window_size <= 0:
        raise SystemExit("--window-size must be > 0")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be > 0")

    cfg_path = Path(args.positronic_config)
    if not cfg_path.is_absolute():
        cfg_path = (root / cfg_path).resolve()
    cfg = step_eval._load_positronic_config(cfg_path)
    ckpt_path = step_eval._resolve_checkpoint(args, cfg)

    episode_dirs = _resolve_episode_dirs(args)
    if not episode_dirs:
        raise SystemExit("No episode directories found")

    rows: list[dict[str, Any]] = []
    for episode_dir in episode_dirs:
        rows.extend(_scan_episode(episode_dir, args.metric, args.window_size, args.joint))
    if not rows:
        raise SystemExit("No windows found; check --window-size and episode/task paths")

    rows.sort(key=lambda row: row["score"], reverse=True)
    seen_episodes: set[str] = set()
    top_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["episode_name"] not in seen_episodes:
            top_rows.append(row)
            seen_episodes.add(row["episode_name"])
        if len(top_rows) >= args.top_k:
            break

    if args.joint is not None:
        rank_metric = f"mean_abs_joint_{args.joint}_over_chunk_and_window"
    else:
        rank_metric = f"mean_abs_{args.metric}_over_chunk_and_window"

    scope = "episode" if args.episode else "task"
    print(
        f"Top {len(top_rows)} {scope} windows by metric={rank_metric} "
        f"(window_size={args.window_size})"
    )
    print()

    for idx, row in enumerate(top_rows, 1):
        joint_text = " ".join(f"j{joint_idx}={value:.4f}" for joint_idx, value in enumerate(row["joint_means"]))
        print(
            f"{idx:2d}. {row['episode_name']} steps {row['start']}-{row['end']}  "
            f"score={row['score']:.4f}  {joint_text}"
        )
        print(_batch_eval_command(row, root, ckpt_path, args.rynnvla_repo.strip()))
        print()

    if args.save_windows:
        import json as _json
        score_key = f"joint_{args.joint}_window_score" if args.joint is not None else "window_score"
        windows_out = [
            {
                "episode": str(row["episode_dir"]),
                "episode_name": row["episode_name"],
                "start_step": row["start"],
                "end_step": row["end"],
                "max_steps": args.window_size,
                score_key: row["score"],
            }
            for row in top_rows
        ]
        save_path = Path(args.save_windows)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            _json.dump({"window_size": args.window_size, "rank_metric": rank_metric, "windows": windows_out}, f, indent=2)
        print(f"saved {len(windows_out)} windows to {save_path}")


if __name__ == "__main__":
    main()
