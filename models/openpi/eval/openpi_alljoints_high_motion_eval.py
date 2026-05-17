#!/usr/bin/env python3
"""Run RynnVLA-style high-motion evals for a Quantycat OpenPI checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path("/home/caroline/quantycat-positronic")
OPENPI_REPO = Path("/home/caroline/openpi")
RYNNVLA_REPO = Path("/home/caroline/RynnVLA-002/rynnvla-002")
RYNN_EVAL_DIR = REPO / "models/rynnvla-002/eval/model_eval"
CONFIG = REPO / "models/rynnvla-002/config.yaml"
FIND_WINDOWS_SCRIPT = REPO / "models/rynnvla-002/eval/data_analysis/find_high_motion_windows.py"
TASK_DIR = REPO / "my_data/training_pipeline/training_data/Put_the_screwdriver_into_the_cup"
RYNN_PYTHON = "/home/caroline/miniconda3/envs/rynnvla002/bin/python"
DEFAULT_ROOT = REPO / "eval_output/screwdriver_so101/model_eval"
DEFAULT_CHECKPOINT = (
    REPO
    / "my_data/training_pipeline/openpi/checkpoints/pi05_quantycat_lora/"
    "screwdriver_so101_pi05_h20_lora_20260516_pdt/4999"
)
DEFAULT_LABEL = "openpi_pi05_h20_lora_step4999"
PROMPT = "Put the screwdriver into the cup"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config-name", default="pi05_quantycat_lora")
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--joints", default="0,1,2,3,4")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--sign-eps", type=float, default=0.01)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--save-traces", action="store_true")
    return parser.parse_args()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(message: str) -> None:
    print(f"[{_stamp()}] {message}", flush=True)


def _run(cmd: list[str], *, cwd: Path = REPO) -> None:
    _log("$ " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {' '.join(cmd)}")


def _find_windows(joint: int, output_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    windows_json = output_dir / "selected_windows.json"
    if windows_json.exists() and not args.force:
        return json.loads(windows_json.read_text(encoding="utf-8"))["windows"]

    cmd = [
        RYNN_PYTHON,
        str(FIND_WINDOWS_SCRIPT),
        "--task-dir",
        str(TASK_DIR),
        "--joint",
        str(joint),
        "--top-k",
        str(args.top_k),
        "--window-size",
        str(args.window_size),
        "--save-windows",
        str(windows_json),
        "--positronic-config",
        str(CONFIG),
    ]
    _run(cmd)
    return json.loads(windows_json.read_text(encoding="utf-8"))["windows"]


def _joint_focus(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    focus_joints = ("joint_0", "joint_1", "joint_2", "joint_3", "joint_4")
    result: dict[str, dict[str, Any]] = {}
    for joint in focus_joints:
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
    def fmt(value: Any) -> str:
        return "nan" if value is None else f"{value:.3f}"

    print(f"\n{title}", flush=True)
    for joint in ("joint_0", "joint_1", "joint_2", "joint_3", "joint_4"):
        row = focus[joint]
        print(
            f"  {joint}: sign={fmt(row['raw_sign_agreement'])} "
            f"corr={fmt(row['raw_same_corr'])} neg_corr={fmt(row['raw_negated_corr'])} "
            f"slope={fmt(row['raw_fit_slope'])} norm_mae={fmt(row['normalized_mae'])}",
            flush=True,
        )


def _load_policy(args: argparse.Namespace):
    sys.path.insert(0, str(OPENPI_REPO / "src"))
    os.chdir(OPENPI_REPO)

    from openpi.policies import policy_config
    from openpi.training import config as openpi_config

    train_config = openpi_config.get_config(args.config_name)
    return policy_config.create_trained_policy(
        train_config,
        args.checkpoint,
        sample_kwargs={"num_steps": args.sample_steps},
        default_prompt=PROMPT,
    )


def _pi_delta_actions(policy, sample: dict[str, Any], horizon: int) -> np.ndarray:
    obs = {
        "observation/images/front": sample["front_current"],
        "observation/images/wrist": sample["wrist_current"],
        "observation/state": sample["state"],
        "prompt": PROMPT,
    }
    pred_abs = np.asarray(policy.infer(obs)["actions"], dtype=np.float32)
    pred = pred_abs[:horizon, :6].copy()
    pred[:, :5] -= sample["state"][:5].reshape(1, 5)
    return pred


def _eval_joint(joint: int, output_dir: Path, windows: list[dict[str, Any]], policy, args: argparse.Namespace) -> Path:
    report = output_dir / "focused_high_motion_joint_sign.json"
    if report.exists() and not args.force:
        _log(f"report already exists for j{joint}: {report}")
        return report
    if report.exists():
        report.unlink()

    import episode_batch_eval as batch_eval
    import episode_step_eval as step_eval

    min_values, max_values = batch_eval._action_bounds()
    all_pred: list[np.ndarray] = []
    all_gt: list[np.ndarray] = []
    cases_out: list[dict[str, Any]] = []
    trace_case_ids: list[str] = []
    trace_case_starts: list[int] = []

    for case_index, window in enumerate(windows, start=1):
        episode_dir = Path(window["episode"]).expanduser().resolve()
        action_dirs = step_eval._sorted_action_dirs(episode_dir / "abs_action")
        last_step = len(action_dirs) - 1
        steps = list(range(window["start_step"], min(window["start_step"] + window["max_steps"], last_step + 1)))
        if not steps:
            raise ValueError(f"No valid steps for {episode_dir} starting at {window['start_step']}")

        pred_steps: list[np.ndarray] = []
        gt_steps: list[np.ndarray] = []
        _log(f"j{joint} case {case_index}/{len(windows)} {episode_dir.name}: steps {steps[0]}-{steps[-1]}")
        for idx, step in enumerate(steps, start=1):
            sample = step_eval._build_sample(episode_dir, step=step, his=1)
            gt = np.asarray(sample["gt_action"], dtype=np.float32)
            pred = _pi_delta_actions(policy, sample, horizon=gt.shape[0])
            if pred.shape != gt.shape:
                raise ValueError(f"{episode_dir.name} step {step}: pred shape {pred.shape} != gt shape {gt.shape}")
            pred_steps.append(pred)
            gt_steps.append(gt)
            if idx == 1 or idx == len(steps) or idx % 10 == 0:
                metrics = step_eval._metrics(pred, gt)
                _log(f"  [{idx}/{len(steps)}] step={step} mean_abs={metrics['mean_abs']:.6f}")

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
    _print_focus(f"j{joint} aggregate focus", aggregate_focus)

    payload = {
        "checkpoint": str(args.checkpoint),
        "config_name": args.config_name,
        "action_convention": (
            "OpenPI policy output is absolute; eval subtracts current state from joints 0-4 "
            "before comparing to saved target deltas; gripper remains absolute"
        ),
        "sign_eps": args.sign_eps,
        "case_count": len(cases_out),
        "total_step_count": int(aggregate_pred.shape[0]),
        "chunk_size": int(aggregate_pred.shape[1]),
        "action_dim": int(aggregate_pred.shape[2]),
        "focus_joints": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4"],
        "aggregate_summary": aggregate_summary,
        "aggregate_focus_joints": aggregate_focus,
        "cases": cases_out,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _log(f"saved focused report: {report}")

    if args.save_traces:
        trace_path = output_dir / "focused_high_motion_traces.npz"
        np.savez_compressed(
            trace_path,
            pred=np.asarray(all_pred, dtype=np.float32),
            gt=np.asarray(all_gt, dtype=np.float32),
            case_id=np.asarray(trace_case_ids),
            case_start_step=np.asarray(trace_case_starts, dtype=np.int32),
        )
        _log(f"saved trace arrays: {trace_path}")
    return report


def _summarize(label_dir: Path, reports: dict[int, Path], checkpoint: Path, config_name: str) -> Path:
    rows: dict[str, Any] = {}
    for ranked_joint, report in sorted(reports.items()):
        data = json.loads(report.read_text(encoding="utf-8"))
        focus = data["aggregate_focus_joints"]
        rows[f"broad_j{ranked_joint}_high_motion_top50"] = {
            "report": str(report),
            "case_count": data["case_count"],
            "total_step_count": data["total_step_count"],
            "diagonal_joint": f"joint_{ranked_joint}",
            "diagonal_metrics": focus.get(f"joint_{ranked_joint}", {}),
            "all_focus_joints": focus,
        }
    summary = {
        "checkpoint": str(checkpoint),
        "config_name": config_name,
        "created_at": _stamp(),
        "coverage": rows,
    }
    out = label_dir / "all_joint_focused_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


def main() -> int:
    args = _parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not (args.checkpoint / "params").is_dir():
        raise FileNotFoundError(args.checkpoint / "params")

    sys.path.insert(0, str(RYNNVLA_REPO))
    sys.path.insert(0, str(RYNN_EVAL_DIR))
    import episode_batch_eval as batch_eval
    import episode_step_eval as step_eval

    root = step_eval._repo_root()
    cfg = step_eval._load_positronic_config(CONFIG)
    batch_eval._configure_env(root, cfg)

    joints = [int(item.strip()) for item in args.joints.split(",") if item.strip()]
    label_dir = args.output_root.expanduser().resolve() / args.label
    label_dir.mkdir(parents=True, exist_ok=True)

    _log(f"loading OpenPI policy from {args.checkpoint}")
    policy = _load_policy(args)
    _log("OpenPI policy loaded")

    reports: dict[int, Path] = {}
    for joint in joints:
        output_dir = label_dir / f"broad_j{joint}_high_motion_top{args.top_k}"
        output_dir.mkdir(parents=True, exist_ok=True)
        _log(f"j{joint}: finding high-motion windows")
        windows = _find_windows(joint, output_dir, args)
        _log(f"j{joint}: running focused eval on {len(windows)} windows")
        reports[joint] = _eval_joint(joint, output_dir, windows, policy, args)

    summary = _summarize(label_dir, reports, args.checkpoint, args.config_name)
    _log(f"wrote all-joint eval summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
