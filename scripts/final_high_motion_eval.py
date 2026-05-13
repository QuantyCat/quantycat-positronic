#!/usr/bin/env python3
"""Run final focused high-motion evals for post-epoch-14 checkpoints."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO = Path("/home/caroline/quantycat-positronic")
OUT_DIR = REPO / "my_data/training_pipeline/fine_tuning/screwdriver_so101_fresh_h1j3_weighted_scratch"
CONFIG = REPO / "models/rynnvla-002/config.yaml"
EVAL_SCRIPT = REPO / "models/rynnvla-002/eval/model_eval/focused_sign_eval.py"
CASE_SOURCE = (
    REPO
    / "eval_output/screwdriver_so101/model_eval/fresh_h1j3_weighted_epoch14/"
    / "broad_j3_high_motion_top50/focused_high_motion_joint_sign.json"
)
MODEL_EVAL_DIR = REPO / "eval_output/screwdriver_so101/model_eval"
LOG = REPO / "run_logs/final_high_motion_eval.log"
PYTHON = "/home/caroline/miniconda3/envs/rynnvla002/bin/python"
RYNNVLA_REPO = "/home/caroline/RynnVLA-002/rynnvla-002"

CHECKPOINTS = [
    ("fresh_h1j3_weighted_epoch16_iter5999", OUT_DIR / "epoch16-iter5999"),
    ("fresh_h1j3_weighted_epoch16_iter7999", OUT_DIR / "epoch16-iter7999"),
    ("fresh_h1j3_weighted_epoch16", OUT_DIR / "epoch16"),
]


def log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_cases() -> list[dict[str, object]]:
    data = json.loads(CASE_SOURCE.read_text(encoding="utf-8"))
    return data["cases"]


def report_path(label: str) -> Path:
    return MODEL_EVAL_DIR / label / "broad_j3_high_motion_top50" / "focused_high_motion_joint_sign.json"


def run_one(label: str, checkpoint: Path, cases: list[dict[str, object]]) -> None:
    report = report_path(label)
    if report.exists():
        log(f"skipping {label}; report already exists: {report}")
        return

    if not (checkpoint / "model.safetensors").exists():
        raise FileNotFoundError(f"missing checkpoint model: {checkpoint / 'model.safetensors'}")

    output_dir = report.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON,
        str(EVAL_SCRIPT),
        "--checkpoint",
        str(checkpoint),
        "--positronic-config",
        str(CONFIG),
        "--rynnvla-repo",
        RYNNVLA_REPO,
        "--output-dir",
        str(output_dir),
        "--sign-eps",
        "0.01",
        "--gpu",
        "0",
        "--save-traces",
    ]
    for case in cases:
        cmd.extend(["--case", f"{case['episode']}:{case['start_step']}:{case['step_count']}"])

    stdout_log = output_dir / "focused_eval_stdout.log"
    log(f"starting {label}: {checkpoint}")
    log(f"capturing {label} stdout/stderr to {stdout_log}")
    with stdout_log.open("a", encoding="utf-8") as f:
        result = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        log(f"{label} failed with exit code {result.returncode}; see {stdout_log}")
        raise subprocess.CalledProcessError(result.returncode, cmd)
    log(f"finished {label}: {report}")


def main() -> None:
    cases = load_cases()
    log(f"loaded {len(cases)} high-motion cases from {CASE_SOURCE}")
    for label, checkpoint in CHECKPOINTS:
        run_one(label, checkpoint, cases)
    log("final high-motion eval round complete")


if __name__ == "__main__":
    main()
