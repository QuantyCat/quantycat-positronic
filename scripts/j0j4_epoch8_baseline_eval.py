#!/usr/bin/env python3
"""Baseline eval for j0 and j4 on the h1j123_j3restore epoch-8 checkpoint.

Finds j0 high-motion windows across all training episodes and runs
focused_sign_eval (which now covers joint_0 through joint_4) to establish
a j0/j4 baseline before designing the next balanced recipe.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models/rynnvla-002/eval/data_analysis"))

REPO = Path("/home/caroline/quantycat-positronic")
CHECKPOINT = (
    REPO
    / "my_data/training_pipeline/fine_tuning/"
    / "testing_checkpoints/h1j123_j3restore_epoch8_robot_test_candidate"
)
CONFIG = REPO / "models/rynnvla-002/config.yaml"
EVAL_SCRIPT = REPO / "models/rynnvla-002/eval/model_eval/focused_sign_eval.py"
FIND_WINDOWS_SCRIPT = REPO / "models/rynnvla-002/eval/data_analysis/find_high_motion_windows.py"
TASK_DIR = REPO / "my_data/training_pipeline/training_data/Put_the_screwdriver_into_the_cup"
OUTPUT_DIR = (
    REPO
    / "eval_output/screwdriver_so101/model_eval/h1j123_j3restore_epoch8/"
    / "broad_j0_high_motion_top50"
)
WINDOWS_JSON = OUTPUT_DIR / "selected_windows.json"
LOG = REPO / "run_logs/j0j4_epoch8_baseline_eval.log"
PYTHON = "/home/caroline/miniconda3/envs/rynnvla002/bin/python"
RYNNVLA_REPO = "/home/caroline/RynnVLA-002/rynnvla-002"


def log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def find_j0_windows() -> list[dict]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON,
        str(FIND_WINDOWS_SCRIPT),
        "--task-dir", str(TASK_DIR),
        "--joint", "0",
        "--top-k", "50",
        "--window-size", "50",
        "--save-windows", str(WINDOWS_JSON),
        "--positronic-config", str(CONFIG),
    ]
    log(f"finding j0 high-motion windows from {TASK_DIR}")
    result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"find_windows failed:\n{result.stderr}")
        raise RuntimeError("find_high_motion_windows failed")
    log(f"saved windows to {WINDOWS_JSON}")
    return json.loads(WINDOWS_JSON.read_text(encoding="utf-8"))["windows"]


def main() -> int:
    report = OUTPUT_DIR / "focused_high_motion_joint_sign.json"
    if report.exists():
        log(f"report already exists: {report}")
        return 0
    if not (CHECKPOINT / "model.safetensors").exists():
        raise FileNotFoundError(CHECKPOINT / "model.safetensors")

    windows = find_j0_windows()
    log(f"using {len(windows)} j0 high-motion windows")

    cmd = [
        PYTHON,
        str(EVAL_SCRIPT),
        "--checkpoint", str(CHECKPOINT),
        "--positronic-config", str(CONFIG),
        "--rynnvla-repo", RYNNVLA_REPO,
        "--output-dir", str(OUTPUT_DIR),
        "--sign-eps", "0.01",
        "--gpu", "0",
        "--deterministic-crop",
        "--save-traces",
    ]
    for w in windows:
        cmd.extend(["--case", f"{w['episode']}:{w['start_step']}:{w['max_steps']}"])

    stdout_log = OUTPUT_DIR / "focused_eval_stdout.log"
    log(f"starting eval on checkpoint: {CHECKPOINT}")
    log(f"stdout/stderr -> {stdout_log}")
    with stdout_log.open("a", encoding="utf-8") as f:
        result = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        log(f"eval failed with exit code {result.returncode}; see {stdout_log}")
        return result.returncode
    log(f"finished: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
