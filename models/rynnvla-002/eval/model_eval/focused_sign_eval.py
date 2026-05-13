#!/usr/bin/env python3
"""Evaluate selected high-motion windows and summarize joint sign behavior.

Ground-truth actions are loaded directly from saved `abs_action` chunks. Those
files already contain target deltas for joints 0-4, with an absolute gripper
target. Evaluation must not subtract the current state from them.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

import episode_batch_eval as batch_eval
import episode_step_eval as step_eval


_FOCUS_JOINTS = ("joint_0", "joint_1", "joint_2", "joint_3", "joint_4")


def _parse_case(value: str) -> tuple[Path, int, int]:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("case must be EPISODE_PATH:START_STEP:MAX_STEPS")
    episode = Path(parts[0]).expanduser()
    try:
        start = int(parts[1])
        count = int(parts[2])
    except ValueError as e:
        raise argparse.ArgumentTypeError("START_STEP and MAX_STEPS must be integers") from e
    if start < 0 or count <= 0:
        raise argparse.ArgumentTypeError("START_STEP must be >= 0 and MAX_STEPS must be > 0")
    return episode, start, count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=_parse_case, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--positronic-config", default="models/rynnvla-002/config.yaml")
    parser.add_argument("--rynnvla-repo", default=os.environ.get("RYNNVLA_REPO", ""))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sign-eps", type=float, default=0.01)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--deterministic-crop", action="store_true")
    parser.add_argument("--save-traces", action="store_true", help="Write per-step prediction/GT arrays for offline audits.")
    return parser.parse_args()


def _joint_focus(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for joint in _FOCUS_JOINTS:
        convention = summary["convention_checks"][joint]
        norm_convention = summary["normalized_convention_checks"][joint]
        result[joint] = {
            "raw_sign_agreement": summary["per_joint"]["sign_agreement"][joint],
            "raw_sign_count": summary["per_joint"]["sign_count"][joint],
            "raw_same_corr": convention["same_joint_corr"],
            "raw_negated_corr": convention["negated_same_joint_corr"],
            "raw_fit_slope": convention["fit_same_joint"]["slope"],
            "normalized_mae": summary["normalized_per_joint"]["mean_abs_error"][joint],
            "normalized_sign_agreement": summary["normalized_per_joint"]["sign_agreement"][joint],
            "normalized_centered_sign_agreement": summary["normalized_centered_per_joint"]["sign_agreement"][joint],
            "normalized_centered_sign_center": summary["normalized_centered_per_joint"]["sign_center"][joint],
            "normalized_same_corr": norm_convention["same_joint_corr"],
            "normalized_negated_corr": norm_convention["negated_same_joint_corr"],
            "normalized_fit_slope": norm_convention["fit_same_joint"]["slope"],
        }
    return result


def _print_focus(title: str, focus: dict[str, dict[str, Any]]) -> None:
    def _fmt(value: Any) -> str:
        return "nan" if value is None else f"{value:.3f}"

    def _fmt_count(value: Any) -> str:
        return "0" if value is None else str(value)

    print(f"\n{title}")
    for joint in _FOCUS_JOINTS:
        row = focus[joint]
        print(
            f"  {joint}: sign={_fmt(row['raw_sign_agreement'])} "
            f"centered_norm_sign={_fmt(row['normalized_centered_sign_agreement'])} "
            f"n={_fmt_count(row['raw_sign_count'])} corr={_fmt(row['raw_same_corr'])} "
            f"neg_corr={_fmt(row['raw_negated_corr'])} "
            f"slope={_fmt(row['raw_fit_slope'])} norm_mae={_fmt(row['normalized_mae'])}"
        )


def main() -> None:
    args = _parse_args()
    root = step_eval._repo_root()
    os.chdir(root)

    cfg_path = Path(args.positronic_config)
    if not cfg_path.is_absolute():
        cfg_path = (root / cfg_path).resolve()
    cfg = step_eval._load_positronic_config(cfg_path)
    work_dir = batch_eval._configure_env(root, cfg)

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        training_output = Path(cfg.get("training_output", "training_output"))
        if not training_output.is_absolute():
            training_output = (root / training_output).resolve()
        output_dir = training_output / f"{cfg['task_label']}_{cfg['robot']}" / "model_eval" / ckpt_path.name / "focused_high_motion_joint_sign"
    output_dir.mkdir(parents=True, exist_ok=True)

    step_eval._ensure_rynnvla_on_path(args.rynnvla_repo.strip() or None)
    solver_args = batch_eval._make_solver_args(args, cfg, ckpt_path, output_dir)
    print(f"Loading Solver from {ckpt_path} ...")
    solver = batch_eval._load_solver(args.rynnvla_repo, solver_args)
    min_values, max_values = batch_eval._action_bounds()

    all_pred: list[np.ndarray] = []
    all_gt: list[np.ndarray] = []
    cases_out: list[dict[str, Any]] = []
    trace_case_ids: list[str] = []
    trace_case_starts: list[int] = []
    his = int(cfg["his"])

    for episode_path, start, count in args.case:
        episode_dir = episode_path if episode_path.is_absolute() else (root / episode_path).resolve()
        action_dirs = step_eval._sorted_action_dirs(episode_dir / "abs_action")
        last_step = len(action_dirs) - 1
        steps = list(range(start, min(start + count, last_step + 1)))
        if not steps:
            raise ValueError(f"No valid steps for {episode_dir} starting at {start}")

        prompt = step_eval._extract_task_from_episode(episode_dir)
        pred_steps = []
        gt_steps = []
        print(f"\nEvaluating {episode_dir.name}: steps {steps[0]}-{steps[-1]}")
        for index, step in enumerate(steps, start=1):
            sample = step_eval._build_sample(episode_dir, step=step, his=his)
            step_eval._reset_solver_history(solver, sample)
            pred = solver.get_action_wrist_action_head_state(
                front_image=sample["front_current"],
                wrist_image=sample["wrist_current"],
                state=sample["state"],
                prompt=prompt,
            )
            pred = np.asarray(pred, dtype=np.float32)
            gt = np.asarray(sample["gt_action"], dtype=np.float32)
            if pred.shape != gt.shape:
                raise ValueError(f"{episode_dir.name} step {step}: pred shape {pred.shape} != gt shape {gt.shape}")
            pred_steps.append(pred)
            gt_steps.append(gt)
            if index == 1 or index == len(steps) or index % 10 == 0:
                metrics = step_eval._metrics(pred, gt)
                print(f"  [{index}/{len(steps)}] step={step} mean_abs={metrics['mean_abs']:.6f}")

        pred_arr = np.asarray(pred_steps, dtype=np.float32)
        gt_arr = np.asarray(gt_steps, dtype=np.float32)
        summary = batch_eval._summarize(pred_arr, gt_arr, args.sign_eps, min_values, max_values)
        focus = _joint_focus(summary)
        _print_focus(f"{episode_dir.name} focus", focus)
        all_pred.append(pred_arr)
        all_gt.append(gt_arr)
        trace_case_ids.append(episode_dir.name)
        trace_case_starts.append(steps[0])
        cases_out.append(
            {
                "episode": str(episode_dir),
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
    aggregate_focus = _joint_focus(aggregate_summary)
    _print_focus("Aggregate focus", aggregate_focus)

    out_path = output_dir / "focused_high_motion_joint_sign.json"
    payload = {
        "checkpoint": str(ckpt_path),
        "positronic_config": str(cfg_path),
        "work_dir": str(work_dir),
        "action_convention": "saved target deltas for joints 0-4; gripper absolute; no eval-time state subtraction",
        "sign_eps": args.sign_eps,
        "case_count": len(cases_out),
        "total_step_count": int(aggregate_pred.shape[0]),
        "chunk_size": int(aggregate_pred.shape[1]),
        "action_dim": int(aggregate_pred.shape[2]),
        "focus_joints": list(_FOCUS_JOINTS),
        "aggregate_summary": aggregate_summary,
        "aggregate_focus_joints": aggregate_focus,
        "cases": cases_out,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved focused report to {out_path}")

    if args.save_traces:
        trace_path = output_dir / "focused_high_motion_traces.npz"
        np.savez_compressed(
            trace_path,
            pred=np.asarray(all_pred, dtype=np.float32),
            gt=np.asarray(all_gt, dtype=np.float32),
            case_id=np.asarray(trace_case_ids),
            case_start_step=np.asarray(trace_case_starts, dtype=np.int32),
        )
        print(f"Saved trace arrays to {trace_path}")


if __name__ == "__main__":
    main()
