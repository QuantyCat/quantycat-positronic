#!/usr/bin/env python3
"""Run focused high-motion eval coverage for joints 0-4 on one checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "models/rynnvla-002/config.yaml"
EVAL_SCRIPT = REPO / "eval/rynnvla_002/focused_sign_eval.py"
FIND_WINDOWS_SCRIPT = REPO / "eval/rynnvla_002/data_analysis/find_high_motion_windows.py"
TASK_DIR = REPO / "my_data/training_pipeline/training_data/Put_the_screwdriver_into_the_cup"
PYTHON = "/home/caroline/miniconda3/envs/rynnvla002/bin/python"
RYNNVLA_REPO = str(REPO / "vendor/rynnvla-002/rynnvla-002")
DEFAULT_ROOT = REPO / "eval_output/screwdriver_so101/model_eval"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--joints", default="0,1,2,3,4")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(message: str) -> None:
    print(f"[{_stamp()}] {message}", flush=True)


def _run(cmd: list[str], *, cwd: Path = REPO, stdout_path: Path | None = None) -> None:
    _log("$ " + " ".join(cmd))
    if stdout_path is None:
        result = subprocess.run(cmd, cwd=cwd)
    else:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("a", encoding="utf-8") as f:
            result = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {' '.join(cmd)}")


def _find_windows(joint: int, output_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    windows_json = output_dir / "selected_windows.json"
    if windows_json.exists() and not args.force:
        return json.loads(windows_json.read_text(encoding="utf-8"))["windows"]

    cmd = [
        PYTHON,
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


def _eval_joint(joint: int, output_dir: Path, windows: list[dict[str, Any]], args: argparse.Namespace) -> Path:
    report = output_dir / "focused_high_motion_joint_sign.json"
    if report.exists() and not args.force:
        _log(f"report already exists for j{joint}: {report}")
        return report
    if report.exists():
        report.unlink()

    cmd = [
        PYTHON,
        str(EVAL_SCRIPT),
        "--checkpoint",
        str(args.checkpoint),
        "--positronic-config",
        str(CONFIG),
        "--rynnvla-repo",
        RYNNVLA_REPO,
        "--output-dir",
        str(output_dir),
        "--sign-eps",
        "0.01",
        "--gpu",
        str(args.gpu),
        "--deterministic-crop",
        "--save-traces",
    ]
    for window in windows:
        cmd.extend(["--case", f"{window['episode']}:{window['start_step']}:{window['max_steps']}"])
    _run(cmd, stdout_path=output_dir / "focused_eval_stdout.log")
    return report


def _summarize(label_dir: Path, reports: dict[int, Path]) -> Path:
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
        "checkpoint": str(next(iter(json.loads(p.read_text(encoding="utf-8"))["checkpoint"] for p in reports.values()), "")),
        "created_at": _stamp(),
        "coverage": rows,
    }
    out = label_dir / "all_joint_focused_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


def main() -> int:
    args = _parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not (checkpoint / "model.safetensors").exists():
        raise FileNotFoundError(checkpoint / "model.safetensors")
    args.checkpoint = checkpoint

    joints = [int(item.strip()) for item in args.joints.split(",") if item.strip()]
    label_dir = args.output_root.expanduser().resolve() / args.label
    reports: dict[int, Path] = {}
    for joint in joints:
        output_dir = label_dir / f"broad_j{joint}_high_motion_top{args.top_k}"
        output_dir.mkdir(parents=True, exist_ok=True)
        _log(f"j{joint}: finding high-motion windows")
        windows = _find_windows(joint, output_dir, args)
        _log(f"j{joint}: running focused eval on {len(windows)} windows")
        reports[joint] = _eval_joint(joint, output_dir, windows, args)

    summary = _summarize(label_dir, reports)
    _log(f"wrote all-joint eval summary: {summary}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
