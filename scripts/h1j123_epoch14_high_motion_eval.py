#!/usr/bin/env python3
"""Run focused high-motion eval for the h1/j1-j2-j3 balanced epoch-14 checkpoint."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CHECKPOINT = (
    REPO
    / "my_data/training_pipeline/fine_tuning/"
    / "screwdriver_so101_h1j123_balanced_scratch/epoch14"
)
CONFIG = REPO / "models/rynnvla-002/config.yaml"
EVAL_SCRIPT = REPO / "models/rynnvla-002/eval/model_eval/focused_sign_eval.py"
CASE_SOURCE = (
    REPO
    / "eval_output/screwdriver_so101/model_eval/fresh_h1j3_weighted_epoch14/"
    / "broad_j3_high_motion_top50/focused_high_motion_joint_sign.json"
)
OUTPUT_DIR = (
    REPO
    / "eval_output/screwdriver_so101/model_eval/h1j123_balanced_epoch14/"
    / "broad_j3_high_motion_top50"
)
LOG = REPO / "run_logs/h1j123_epoch14_high_motion_eval.log"
PYTHON = "/home/caroline/miniconda3/envs/rynnvla002/bin/python"
RYNNVLA_REPO = str(REPO / "vendor/rynnvla-002/rynnvla-002")


def log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> int:
    report = OUTPUT_DIR / "focused_high_motion_joint_sign.json"
    if report.exists():
        log(f"report already exists: {report}")
        return 0
    if not (CHECKPOINT / "model.safetensors").exists():
        raise FileNotFoundError(CHECKPOINT / "model.safetensors")

    cases = json.loads(CASE_SOURCE.read_text(encoding="utf-8"))["cases"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON,
        str(EVAL_SCRIPT),
        "--checkpoint",
        str(CHECKPOINT),
        "--positronic-config",
        str(CONFIG),
        "--rynnvla-repo",
        RYNNVLA_REPO,
        "--output-dir",
        str(OUTPUT_DIR),
        "--sign-eps",
        "0.01",
        "--gpu",
        "0",
        "--save-traces",
    ]
    for case in cases:
        cmd.extend(["--case", f"{case['episode']}:{case['start_step']}:{case['step_count']}"])

    stdout_log = OUTPUT_DIR / "focused_eval_stdout.log"
    log(f"loaded {len(cases)} high-motion cases from {CASE_SOURCE}")
    log(f"starting eval: {CHECKPOINT}")
    log(f"capturing stdout/stderr to {stdout_log}")
    with stdout_log.open("a", encoding="utf-8") as f:
        result = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        log(f"eval failed with exit code {result.returncode}; see {stdout_log}")
        return result.returncode
    log(f"finished eval: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
