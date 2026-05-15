#!/usr/bin/env python3
"""Fork one-epoch lm_head on/off ablations from the next scaled checkpoint."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


REPO = Path("/home/caroline/quantycat-positronic")
CONFIG = REPO / "models/rynnvla-002/config.yaml"
RUN_DIR = REPO / "my_data/training_pipeline/fine_tuning/screwdriver_so101_h1j01234_scaled_scratch"
PARENT_EPOCH = 3
PARENT_HUMAN_EPOCH = 4
BRANCH_EPOCH = 0
TRAIN_SESSION = "h1j01234_scaled_train"
SUPERVISOR_SESSION = "h1j01234_scaled_supervisor"
LOG = REPO / "run_logs/lm_head_ablation_supervisor.log"
PYTHON = "/home/caroline/miniconda3/envs/rynnvla002/bin/python"

BRANCHES = [
    {
        "name": "screwdriver_so101_h1j01234_scaled_e4_lmhead_on_plus1",
        "label": "h1j01234_scaled_e4_lmhead_on_plus1",
        "train_lm_head": True,
    },
    {
        "name": "screwdriver_so101_h1j01234_scaled_e4_lmhead_off_plus1",
        "label": "h1j01234_scaled_e4_lmhead_off_plus1",
        "train_lm_head": False,
    },
]


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{stamp()}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str], *, cwd: Path = REPO, env: dict[str, str] | None = None) -> None:
    log("$ " + " ".join(cmd))
    with LOG.open("a", encoding="utf-8") as f:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {' '.join(cmd)}")


def guard_config() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if cfg.get("fresh_start"):
        raise RuntimeError(f"Refusing to run while fresh_start is true in {CONFIG}")
    if cfg.get("training_run_name") != "screwdriver_so101_h1j01234_scaled_scratch":
        raise RuntimeError(f"Unexpected training_run_name in {CONFIG}: {cfg.get('training_run_name')}")
    log("config guard passed: fresh_start=false and current run name is scaled scratch")


def parent_checkpoint() -> Path:
    return RUN_DIR / f"epoch{PARENT_EPOCH}"


def validation_complete() -> bool:
    path = RUN_DIR / "log_eval_ood.txt"
    if not path.exists():
        return False
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        return row.get("epoch") == PARENT_EPOCH
    return False


def wait_for_parent() -> Path:
    ckpt = parent_checkpoint()
    model_file = ckpt / "model.safetensors"
    log(f"waiting for parent checkpoint {ckpt} and OOD validation epoch {PARENT_EPOCH}")
    while True:
        if model_file.exists() and validation_complete():
            log(f"parent checkpoint and validation are complete: {ckpt}")
            return ckpt
        time.sleep(60)


def stop_tmux_session(session: str) -> None:
    result = subprocess.run(["tmux", "has-session", "-t", session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode != 0:
        log(f"tmux session not present: {session}")
        return
    log(f"stopping tmux session with Ctrl-C: {session}")
    subprocess.run(["tmux", "send-keys", "-t", session, "C-c"], check=False)


def run_all_joint_eval(checkpoint: Path, label: str) -> None:
    run(
        [
            PYTHON,
            str(REPO / "scripts/alljoints_high_motion_eval.py"),
            "--checkpoint",
            str(checkpoint),
            "--label",
            label,
            "--force",
        ]
    )


def run_branch(parent: Path, branch: dict[str, object]) -> Path:
    output_dir = REPO / "my_data/training_pipeline/fine_tuning" / str(branch["name"])
    if output_dir.exists() and any(output_dir.glob("epoch*/model.safetensors")):
        raise RuntimeError(f"Refusing to overwrite existing branch checkpoints in {output_dir}")

    env = os.environ.copy()
    env.update(
        {
            "MODEL_ROOT": str(REPO / "models/rynnvla-002"),
            "PYTHON": PYTHON,
            "RYNNVLA_TRAINING_RUN_NAME": str(branch["name"]),
            "RYNNVLA_RESUME_FROM_CHECKPOINT": str(parent),
            "RYNNVLA_EPOCHS": "1",
            "RYNNVLA_FRESH_START": "false",
            "RYNNVLA_FT_FROM_CHECKPOINT": "true",
            "RYNNVLA_TRAIN_LM_HEAD": "true" if branch["train_lm_head"] else "false",
        }
    )
    log(
        f"starting branch {branch['name']} from parent human epoch {PARENT_HUMAN_EPOCH}; "
        f"train_lm_head={branch['train_lm_head']}"
    )
    run(["bash", "models/rynnvla-002/run_scripts/finetune.sh"], env=env)
    ckpt = output_dir / f"epoch{BRANCH_EPOCH}"
    if not (ckpt / "model.safetensors").exists():
        raise RuntimeError(f"expected branch checkpoint missing: {ckpt}")
    return ckpt


def main() -> int:
    guard_config()
    parent = wait_for_parent()
    stop_tmux_session(TRAIN_SESSION)
    stop_tmux_session(SUPERVISOR_SESSION)
    time.sleep(20)

    run_all_joint_eval(parent, f"h1j01234_scaled_parent_epoch{PARENT_HUMAN_EPOCH}")

    for branch in BRANCHES:
        ckpt = run_branch(parent, branch)
        run_all_joint_eval(ckpt, str(branch["label"]))

    log("lm_head ablation experiment complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
