#!/usr/bin/env python3
"""
Compare multiple checkpoints on the same episode slice.

This wraps episode_batch_eval.py logic so you can run the same episode/step window
through several checkpoints and compare the summary metrics side by side.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

import episode_batch_eval as batch_eval
import episode_step_eval as step_eval

_DEFAULT_CONFIG = "models/rynnvla-002/config.yaml"
_JOINT_LABELS = ("joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "gripper")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple checkpoints on one episode slice.")
    parser.add_argument("--episode", type=str, required=True, help="Path to one episode directory.")
    parser.add_argument("--checkpoint", type=str, action="append", required=True, help="Checkpoint directory. Pass multiple times.")
    parser.add_argument("--positronic-config", type=str, default=_DEFAULT_CONFIG)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--deterministic-crop", action="store_true")
    parser.add_argument(
        "--rynnvla-repo",
        type=str,
        default=os.environ.get("RYNNVLA_REPO", ""),
        help="If set, prepend this directory to sys.path.",
    )
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--sign-eps", type=float, default=1e-6)
    parser.add_argument("--save-json", action="store_true")
    return parser.parse_args()


def _evaluate_checkpoint(
    checkpoint: Path,
    episode_dir: Path,
    steps: list[int],
    prompt: str,
    his: int,
    cfg: dict[str, Any],
    cli_args: argparse.Namespace,
    min_values: np.ndarray,
    max_values: np.ndarray,
) -> dict[str, Any]:
    solver_args = batch_eval._make_solver_args(cli_args, cfg, checkpoint)
    Path(solver_args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"\nLoading Solver from {checkpoint} ...")
    solver = batch_eval._load_solver(cli_args.rynnvla_repo, solver_args)

    pred_steps = []
    gt_steps = []
    per_step = []

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
            raise ValueError(f"{checkpoint}: step {step}: pred shape {pred.shape} != gt shape {gt.shape}")
        pred_steps.append(pred)
        gt_steps.append(gt)
        step_metrics = step_eval._metrics(pred, gt)
        per_step.append({"step": step, "metrics": step_metrics})
        print(f"[{index}/{len(steps)}] {checkpoint.name} step={step} mean_abs={step_metrics['mean_abs']:.6f}")

    pred_arr = np.asarray(pred_steps, dtype=np.float32)
    gt_arr = np.asarray(gt_steps, dtype=np.float32)
    summary = batch_eval._summarize(pred_arr, gt_arr, cli_args.sign_eps, min_values, max_values)
    del solver
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_name": checkpoint.name,
        "summary": summary,
        "per_step": per_step,
    }


def _print_comparison(results: list[dict[str, Any]]) -> None:
    print("\nCheckpoint comparison")
    for result in results:
        summary = result["summary"]
        print(f"\n{result['checkpoint_name']}")
        print(
            "  overall:"
            f" mean_abs={summary['overall']['mean_abs']:.6f}"
            f" magnitude_ratio={summary['overall']['magnitude_ratio']:.6f}"
            f" l2={summary['overall']['l2']:.6f}"
        )
        print(
            "  joints:"
            + "".join(
                f" {joint}_bias={summary['per_joint']['mean_signed_bias'][joint]:.6f}"
                f" {joint}_mae={summary['per_joint']['mean_abs_error'][joint]:.6f}"
                f" {joint}_sign={summary['per_joint']['sign_agreement'][joint]:.6f}"
                for joint in _JOINT_LABELS
            )
        )
        print("  convention:")
        for joint in _JOINT_LABELS:
            check = summary["convention_checks"][joint]
            print(
                f"    {joint}:"
                f" same_corr={check['same_joint_corr']}"
                f" neg_same_corr={check['negated_same_joint_corr']}"
                f" best_match={check['best_match']}"
            )
        print("  linear_fit:")
        for joint in _JOINT_LABELS:
            check = summary["convention_checks"][joint]
            fit_same = check["fit_same_joint"]
            fit_neg = check["fit_negated_same_joint"]
            print(
                f"    {joint} same:"
                f" slope={fit_same['slope']}"
                f" intercept={fit_same['intercept']}"
                f" r2={fit_same['r2']}"
            )
            print(
                f"    {joint} negated:"
                f" slope={fit_neg['slope']}"
                f" intercept={fit_neg['intercept']}"
                f" r2={fit_neg['r2']}"
            )


def main() -> None:
    args = _parse_args()
    root = step_eval._repo_root()
    os.chdir(root)

    cfg_path = Path(args.positronic_config)
    if not cfg_path.is_absolute():
        cfg_path = (root / cfg_path).resolve()
    cfg = step_eval._load_positronic_config(cfg_path)
    batch_eval._configure_env(root, cfg)
    step_eval._ensure_rynnvla_on_path(args.rynnvla_repo.strip() or None)

    episode_dir = Path(args.episode).expanduser().resolve()
    action_dirs = step_eval._sorted_action_dirs(episode_dir / "abs_action")
    last_step = len(action_dirs) - 1
    steps = list(range(args.start_step, last_step + 1, args.stride))[: args.max_steps]
    if not steps:
        raise ValueError("No evaluation steps selected")

    his = int(cfg["his"])
    prompt = args.prompt if args.prompt is not None else step_eval._extract_task_from_episode(episode_dir)
    min_values, max_values = batch_eval._action_bounds()

    print(f"episode={episode_dir}")
    print(f"prompt={prompt}")
    print(f"steps={steps}")

    checkpoints = [Path(p).expanduser().resolve() for p in args.checkpoint]
    results = [
        _evaluate_checkpoint(
            checkpoint=checkpoint,
            episode_dir=episode_dir,
            steps=steps,
            prompt=prompt,
            his=his,
            cfg=cfg,
            cli_args=args,
            min_values=min_values,
            max_values=max_values,
        )
        for checkpoint in checkpoints
    ]

    _print_comparison(results)

    if args.save_json:
        out_dir = checkpoints[0] / "episode_batch_eval_logs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{episode_dir.name}_compare_{steps[0]:06d}_to_{steps[-1]:06d}.json"
        payload = {
            "episode": str(episode_dir),
            "steps": steps,
            "prompt": prompt,
            "results": results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved JSON report to {out_path}")


if __name__ == "__main__":
    main()
