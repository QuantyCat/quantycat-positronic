#!/usr/bin/env python3
"""Pause all-joint training at early checkpoints and run j0-j4 focused evals."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path("/home/caroline/quantycat-positronic")
SESSION = "h1j01234_alljoints_train"
RUN_NAME = "screwdriver_so101_h1j01234_alljoints_scratch"
OUT_DIR = REPO / "my_data/training_pipeline/fine_tuning" / RUN_NAME
EVAL_SCRIPT = REPO / "scripts/alljoints_high_motion_eval.py"
LOG = REPO / "run_logs/alljoints_early_stop_supervisor.log"
PYTHON = "/home/caroline/miniconda3/envs/rynnvla002/bin/python"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", default="4,6,8")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--resume-final", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=REPO, check=check, text=True)


def session_exists() -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", SESSION],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def _log_has_epoch(path: Path, epoch: int) -> bool:
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-50:]:
        try:
            if int(json.loads(line).get("epoch", -1)) == epoch:
                return True
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return False


def checkpoint_ready(epoch: int) -> bool:
    return (OUT_DIR / f"epoch{epoch}" / "model.safetensors").exists()


def validation_done(epoch: int) -> bool:
    return _log_has_epoch(OUT_DIR / "log_eval_ind.txt", epoch) and _log_has_epoch(OUT_DIR / "log_eval_ood.txt", epoch)


def wait_for_epoch(epoch: int, poll_seconds: int) -> None:
    last_status = ""
    while True:
        ckpt = checkpoint_ready(epoch)
        val = validation_done(epoch)
        status = f"epoch {epoch}: checkpoint={ckpt} validation={val}"
        if status != last_status:
            log(status)
            last_status = status
        if ckpt and val:
            return
        time.sleep(poll_seconds)


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


def start_training() -> None:
    if session_exists():
        raise RuntimeError(f"tmux session {SESSION} already exists")
    command = (
        f"cd {REPO} && "
        "source models/rynnvla-002/run_scripts/setup.sh && "
        "bash models/rynnvla-002/run_scripts/finetune.sh"
    )
    run(["tmux", "new-session", "-d", "-s", SESSION, command])
    log(f"started training session {SESSION}")


def run_eval(epoch: int) -> Path:
    label = f"h1j01234_alljoints_epoch{epoch}"
    checkpoint = OUT_DIR / f"epoch{epoch}"
    run([PYTHON, str(EVAL_SCRIPT), "--checkpoint", str(checkpoint), "--label", label])
    return REPO / "eval_output/screwdriver_so101/model_eval" / label / "all_joint_focused_summary.json"


def summarize(epoch: int, summary_path: Path) -> None:
    data: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
    for bucket, row in sorted(data["coverage"].items()):
        metrics = row.get("diagonal_metrics", {})
        log(
            "{bucket}: {joint} sign={sign:.4f} corr={corr:.4f} slope={slope:.4f} norm_mae={mae:.4f}".format(
                bucket=bucket,
                joint=row["diagonal_joint"],
                sign=float(metrics.get("raw_sign_agreement", float("nan"))),
                corr=float(metrics.get("raw_same_corr", float("nan"))),
                slope=float(metrics.get("raw_fit_slope", float("nan"))),
                mae=float(metrics.get("normalized_mae", float("nan"))),
            )
        )
    log(f"epoch {epoch} all-joint eval summary: {summary_path}")


def main() -> None:
    args = _parse_args()
    epochs = [int(item.strip()) for item in args.epochs.split(",") if item.strip()]
    if not epochs:
        raise ValueError("--epochs cannot be empty")
    final_epoch = epochs[-1]
    for epoch in epochs:
        log(f"watching for epoch {epoch} checkpoint plus validation")
        wait_for_epoch(epoch, args.poll_seconds)
        log(f"epoch {epoch} checkpoint and validation are ready; stopping training for eval")
        stop_training()
        summary = run_eval(epoch)
        summarize(epoch, summary)
        if epoch == final_epoch and not args.resume_final:
            log(f"leaving run stopped after epoch {epoch} for review")
            return
        start_training()
    log("supervisor complete")


if __name__ == "__main__":
    main()
