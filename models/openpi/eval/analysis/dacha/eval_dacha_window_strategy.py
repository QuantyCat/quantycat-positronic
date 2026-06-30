#!/usr/bin/env python3
"""Evaluate an OpenPI checkpoint on random or evenly selected Dacha windows."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _parse_episodes(value: str) -> list[int]:
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
    return list(dict.fromkeys(episodes))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--episodes", required=True, help="Comma-separated episode ids or inclusive ranges.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--strategy", choices=("even", "random"), default="even")
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--windows-per-episode", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--sign-eps", type=float, default=0.01)
    parser.add_argument("--save-traces", action="store_true")
    return parser.parse_args()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(message: str) -> None:
    print(f"[{_stamp()}] {message}", flush=True)


def _load_quantycat_config(script_dir: Path) -> None:
    positronic_repo = Path(os.environ.get("QUANTYCAT_POSITRONIC_REPO", "/home/caroline/quantycat-positronic"))
    config_path = positronic_repo / "models/openpi/vendor_patches/src/quantycat_training_config.py"
    spec = importlib.util.spec_from_file_location("quantycat_training_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load config module from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["quantycat_training_config"] = module


def _import_relative_eval(script_dir: Path):
    positronic_repo = Path(os.environ.get("QUANTYCAT_POSITRONIC_REPO", "/home/caroline/quantycat-positronic"))
    eval_dir = positronic_repo / "models/openpi/eval/core_evals"
    sys.path.insert(0, str(eval_dir))
    _load_quantycat_config(script_dir)
    import relative_eval

    relative_eval.PROMPT = os.environ.get(
        "DACHA_EVAL_PROMPT",
        "pick up the orange wire and put it in the cup",
    )
    return relative_eval


def _window_starts(max_start: int, count: int, strategy: str, rng: np.random.Generator) -> list[int]:
    if max_start < 0:
        return []
    count = max(1, min(count, max_start + 1))
    if strategy == "even":
        if count == 1:
            return [max_start // 2]
        return sorted({int(round(v)) for v in np.linspace(0, max_start, count)})
    return sorted(rng.choice(np.arange(max_start + 1), size=count, replace=False).astype(int).tolist())


def _select_windows(episodes: list[Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    windows: list[dict[str, Any]] = []
    for ep in episodes:
        score_count = len(ep.actions) - args.action_horizon + 1
        max_start = score_count - args.window_size
        for start in _window_starts(max_start, args.windows_per_episode, args.strategy, rng):
            windows.append(
                {
                    "episode_index": ep.index,
                    "episode": str(ep.parquet),
                    "start_step": int(start),
                    "end_step": int(start + args.window_size - 1),
                    "max_steps": args.window_size,
                    "selection_strategy": args.strategy,
                }
            )
    if args.strategy == "random":
        rng.shuffle(windows)
    if args.max_windows > 0:
        windows = windows[: args.max_windows]
    return sorted(windows, key=lambda row: (int(row["episode_index"]), int(row["start_step"])))


def main() -> int:
    args = _parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()

    script_dir = Path(__file__).resolve().parent
    relative_eval = _import_relative_eval(script_dir)

    episode_filter = _parse_episodes(args.episodes)
    episodes = relative_eval._load_episodes(args.dataset_root, episode_filter=episode_filter)
    episodes_by_index = {ep.index: ep for ep in episodes}
    windows = _select_windows(episodes, args)
    if not windows:
        raise RuntimeError("No eval windows selected.")

    label_dir = args.output_root / args.label
    label_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / "selected_windows.json").write_text(
        json.dumps(
            {
                "dataset_root": str(args.dataset_root),
                "episodes": episode_filter,
                "strategy": args.strategy,
                "window_size": args.window_size,
                "action_horizon": args.action_horizon,
                "windows": windows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _log(f"loaded {len(episodes)} episodes from {args.dataset_root}")
    _log(f"selected {len(windows)} {args.strategy} windows")
    _log("computing commanded-delta action bounds")
    min_values, max_values = relative_eval._action_bounds(episodes, args.action_horizon)
    _log(f"loading OpenPI policy from {args.checkpoint}")
    policy = relative_eval._load_policy(args)
    _log("OpenPI policy loaded")

    sys.path.insert(0, str(Path(os.environ.get("QUANTYCAT_POSITRONIC_REPO", "/home/caroline/quantycat-positronic")) / "models/openpi/eval/libs"))
    import episode_batch_eval as batch_eval
    import episode_step_eval as step_eval

    all_pred: list[np.ndarray] = []
    all_gt: list[np.ndarray] = []
    cases: list[dict[str, Any]] = []
    trace_case_ids: list[str] = []
    trace_case_starts: list[int] = []

    for case_index, window in enumerate(windows, start=1):
        ep = episodes_by_index[int(window["episode_index"])]
        steps = list(range(int(window["start_step"]), int(window["end_step"]) + 1))
        _log(f"case {case_index}/{len(windows)} episode_{ep.index:06d}: steps {steps[0]}-{steps[-1]}")
        pred_steps: list[np.ndarray] = []
        gt_steps: list[np.ndarray] = []
        for idx, step in enumerate(steps, start=1):
            gt = relative_eval._gt_chunk(ep, step, args.action_horizon)
            pred = relative_eval._policy_delta(policy, ep, step, args.action_horizon)
            pred_steps.append(pred)
            gt_steps.append(gt)
            if idx == 1 or idx == len(steps) or idx % 10 == 0:
                metrics = step_eval._metrics(pred, gt)
                _log(f"  [{idx}/{len(steps)}] step={step} mean_abs={metrics['mean_abs']:.6f}")

        pred_arr = np.asarray(pred_steps, dtype=np.float32)
        gt_arr = np.asarray(gt_steps, dtype=np.float32)
        summary = batch_eval._summarize(pred_arr, gt_arr, args.sign_eps, min_values, max_values)
        focus = relative_eval._joint_focus(summary)
        relative_eval._print_focus(f"episode_{ep.index:06d} window focus", focus)
        all_pred.append(pred_arr)
        all_gt.append(gt_arr)
        trace_case_ids.append(f"episode_{ep.index:06d}")
        trace_case_starts.append(steps[0])
        cases.append(
            {
                "episode_index": ep.index,
                "episode": str(ep.parquet),
                "start_step": steps[0],
                "end_step": steps[-1],
                "step_count": len(steps),
                "summary": summary,
                "focus_joints": focus,
            }
        )

    aggregate_pred = np.concatenate(all_pred, axis=0)
    aggregate_gt = np.concatenate(all_gt, axis=0)
    aggregate_summary = batch_eval._summarize(aggregate_pred, aggregate_gt, args.sign_eps, min_values, max_values)
    aggregate_focus = relative_eval._joint_focus(aggregate_summary)
    relative_eval._print_focus("aggregate window focus", aggregate_focus)

    payload = {
        "checkpoint": str(args.checkpoint),
        "config_name": args.config_name,
        "dataset_root": str(args.dataset_root),
        "created_at": _stamp(),
        "selection_strategy": args.strategy,
        "window_size": args.window_size,
        "action_horizon": args.action_horizon,
        "case_count": len(cases),
        "total_step_count": int(aggregate_pred.shape[0]),
        "chunk_size": int(aggregate_pred.shape[1]),
        "action_dim": int(aggregate_pred.shape[2]),
        "aggregate_summary": aggregate_summary,
        "aggregate_focus_joints": aggregate_focus,
        "cases": cases,
    }
    out = label_dir / "window_strategy_summary.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _log(f"wrote summary: {out}")

    if args.save_traces:
        trace_path = label_dir / "window_strategy_traces.npz"
        np.savez_compressed(
            trace_path,
            pred=np.asarray(all_pred, dtype=np.float32),
            gt=np.asarray(all_gt, dtype=np.float32),
            case_id=np.asarray(trace_case_ids),
            case_start_step=np.asarray(trace_case_starts, dtype=np.int32),
        )
        _log(f"saved trace arrays: {trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
