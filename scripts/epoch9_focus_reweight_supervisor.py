#!/usr/bin/env python3
"""Pause after epoch 9 validation, run focused eval, optionally reweight, resume."""

from __future__ import annotations

import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SESSION = "fresh_h1j3_weighted"
EPOCH = 9
OUT_DIR = REPO / "my_data/training_pipeline/fine_tuning/screwdriver_so101_fresh_h1j3_weighted_scratch"
CONFIG = REPO / "models/rynnvla-002/config.yaml"
EVAL_SRC = REPO / "eval_output/screwdriver_so101/model_eval/scratch_epoch13_motion_amp/broad_j3_high_motion_top50/focused_high_motion_joint_sign.json"
EVAL_SCRIPT = REPO / "models/rynnvla-002/eval/model_eval/focused_sign_eval.py"
EVAL_MODULE_DIR = EVAL_SCRIPT.parent
REPORT_DIR = REPO / f"eval_output/screwdriver_so101/model_eval/fresh_h1j3_weighted_epoch{EPOCH}/broad_j3_high_motion_top50"
REPORT = REPORT_DIR / "focused_high_motion_joint_sign.json"
LOG = REPO / "run_logs" / f"focus_reweight_supervisor_e{EPOCH}.log"
PYTHON = "/home/caroline/miniconda3/envs/rynnvla002/bin/python"
RYNNVLA_REPO = str(REPO / "vendor/rynnvla-002/rynnvla-002")

THRESHOLDS = {
    "raw_sign_agreement": 0.94,
    "raw_same_corr": 0.85,
    "raw_fit_slope_min": 0.85,
    "raw_fit_slope_max": 1.15,
}

PHASE2_LINES = {
    "action_sign_joint_weights": "action_sign_joint_weights: [1, 6, 6, 8, 1, 0]",
    "action_motion_joint_weights": "action_motion_joint_weights: [0, 3, 3, 8, 0, 0]",
    "action_magnitude_joint_weights": "action_magnitude_joint_weights: [0, 1, 1, 1, 0, 0]",
}


def log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str], check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, check=check, text=True, **kwargs)


def session_exists() -> bool:
    return subprocess.run(["tmux", "has-session", "-t", SESSION], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def validation_done() -> bool:
    log_file = OUT_DIR / "log_eval_ood.txt"
    if not log_file.exists():
        return False
    for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]:
        try:
            if json.loads(line).get("epoch") == EPOCH:
                return True
        except json.JSONDecodeError:
            continue
    return False


def checkpoint_ready() -> bool:
    return (OUT_DIR / f"epoch{EPOCH}" / "model.safetensors").exists()


def wait_for_epoch_boundary() -> None:
    while True:
        if checkpoint_ready() and validation_done():
            log(f"epoch {EPOCH} checkpoint and OOD validation log are present")
            return
        time.sleep(60)


def stop_training() -> None:
    if not session_exists():
        log(f"tmux session {SESSION} is already absent")
        return
    run(["tmux", "send-keys", "-t", f"{SESSION}:0", "C-c"], check=False)
    deadline = time.time() + 300
    while time.time() < deadline:
        if not session_exists():
            log(f"tmux session {SESSION} exited after C-c")
            return
        time.sleep(5)
    run(["tmux", "send-keys", "-t", f"{SESSION}:0", "exit", "Enter"], check=False)
    time.sleep(5)
    if session_exists():
        raise RuntimeError(f"tmux session {SESSION} did not exit cleanly")


def run_focused_eval() -> None:
    if REPORT.exists():
        log(f"focused eval report already exists: {REPORT}")
        return
    cases = json.loads(EVAL_SRC.read_text(encoding="utf-8"))["cases"]
    sys.path.insert(0, str(EVAL_MODULE_DIR))
    argv = [
        str(EVAL_SCRIPT),
        "--checkpoint",
        str(OUT_DIR / f"epoch{EPOCH}"),
        "--positronic-config",
        str(CONFIG),
        "--rynnvla-repo",
        RYNNVLA_REPO,
        "--output-dir",
        str(REPORT_DIR),
        "--sign-eps",
        "0.01",
        "--gpu",
        "0",
        "--save-traces",
    ]
    for case in cases:
        argv.extend(["--case", f"{case['episode']}:{case['start_step']}:{case['step_count']}"])
    old_argv = sys.argv
    try:
        sys.argv = argv
        log(f"starting focused eval for epoch {EPOCH}")
        runpy.run_path(str(EVAL_SCRIPT), run_name="__main__")
    finally:
        sys.argv = old_argv
    log(f"focused eval complete: {REPORT}")


def load_j3_metrics() -> dict[str, float]:
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    return data["aggregate_focus_joints"]["joint_3"]


def passes(metrics: dict[str, float]) -> bool:
    slope = metrics["raw_fit_slope"]
    return (
        metrics["raw_sign_agreement"] >= THRESHOLDS["raw_sign_agreement"]
        and metrics["raw_same_corr"] >= THRESHOLDS["raw_same_corr"]
        and THRESHOLDS["raw_fit_slope_min"] <= slope <= THRESHOLDS["raw_fit_slope_max"]
    )


def update_config_for_phase2() -> None:
    text = CONFIG.read_text(encoding="utf-8")
    backup = CONFIG.with_suffix(f".yaml.epoch{EPOCH}_pre_phase2")
    if not backup.exists():
        shutil.copy2(CONFIG, backup)
        log(f"backed up config to {backup}")
    for key, replacement in PHASE2_LINES.items():
        text, count = re.subn(rf"^{key}: .*?$", replacement, text, count=1, flags=re.MULTILINE)
        if count != 1:
            raise RuntimeError(f"could not update {key} in {CONFIG}")
    CONFIG.write_text(text, encoding="utf-8")
    log("updated config to phase-2 weights")


def restart_training() -> None:
    if session_exists():
        raise RuntimeError(f"tmux session {SESSION} still exists before restart")
    command = (
        f"cd {REPO} && "
        "source models/rynnvla-002/run_scripts/setup.sh && "
        "bash models/rynnvla-002/run_scripts/finetune.sh"
    )
    run(["tmux", "new-session", "-d", "-s", SESSION, command])
    log(f"restarted training session {SESSION}")


def main() -> None:
    log(f"watching for epoch {EPOCH} checkpoint plus validation completion")
    wait_for_epoch_boundary()
    stop_training()
    run_focused_eval()
    j3 = load_j3_metrics()
    log(
        "epoch {epoch} j3 focused metrics: sign={sign:.4f} corr={corr:.4f} "
        "slope={slope:.4f} norm_mae={mae:.4f}".format(
            epoch=EPOCH,
            sign=j3["raw_sign_agreement"],
            corr=j3["raw_same_corr"],
            slope=j3["raw_fit_slope"],
            mae=j3["normalized_mae"],
        )
    )
    if passes(j3):
        log("j3 passed stability gate; applying phase-2 j1/j2 reweighting")
        update_config_for_phase2()
    else:
        log("j3 did not pass stability gate; resuming with current weights unchanged")
    restart_training()
    log("supervisor complete")


if __name__ == "__main__":
    os.chdir(REPO)
    main()
