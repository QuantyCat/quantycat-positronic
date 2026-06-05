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


REPO = Path(__file__).resolve().parents[3]
EVAL_SCRIPT = REPO / "eval/rynnvla_002/alljoints_high_motion_eval.py"
PYTHON = "/home/caroline/miniconda3/envs/rynnvla002/bin/python"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", default="4,6,8")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--resume-final", action="store_true")
    parser.add_argument("--session", default="h1j01234_alljoints_train")
    parser.add_argument("--run-name", default="screwdriver_so101_h1j01234_alljoints_scratch")
    parser.add_argument("--label-prefix", default="h1j01234_alljoints")
    parser.add_argument("--log", default=str(REPO / "run_logs/alljoints_early_stop_supervisor.log"))
    return parser.parse_args()


def log(message: str, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str], log_path: Path, *, check: bool = True) -> subprocess.CompletedProcess:
    log("$ " + " ".join(cmd), log_path)
    return subprocess.run(cmd, cwd=REPO, check=check, text=True)


def session_exists(session: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", session],
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


def checkpoint_ready(out_dir: Path, epoch: int) -> bool:
    return (out_dir / f"epoch{epoch}" / "model.safetensors").exists()


def validation_done(out_dir: Path, epoch: int) -> bool:
    return _log_has_epoch(out_dir / "log_eval_ind.txt", epoch) and _log_has_epoch(out_dir / "log_eval_ood.txt", epoch)


def wait_for_epoch(out_dir: Path, epoch: int, poll_seconds: int, log_path: Path) -> None:
    last_status = ""
    while True:
        ckpt = checkpoint_ready(out_dir, epoch)
        val = validation_done(out_dir, epoch)
        status = f"epoch {epoch}: checkpoint={ckpt} validation={val}"
        if status != last_status:
            log(status, log_path)
            last_status = status
        if ckpt and val:
            return
        time.sleep(poll_seconds)


def stop_training(session: str, log_path: Path) -> None:
    if not session_exists(session):
        log(f"tmux session {session} is already absent", log_path)
        return
    run(["tmux", "send-keys", "-t", f"{session}:0", "C-c"], log_path, check=False)
    deadline = time.time() + 300
    while time.time() < deadline:
        if not session_exists(session):
            log(f"tmux session {session} exited after C-c", log_path)
            return
        time.sleep(5)
    run(["tmux", "send-keys", "-t", f"{session}:0", "exit", "Enter"], log_path, check=False)
    time.sleep(5)
    if session_exists(session):
        raise RuntimeError(f"tmux session {session} did not exit cleanly")


def start_training(session: str, log_path: Path) -> None:
    if session_exists(session):
        raise RuntimeError(f"tmux session {session} already exists")
    command = (
        f"cd {REPO} && "
        "source models/rynnvla-002/run_scripts/setup.sh && "
        "bash models/rynnvla-002/run_scripts/finetune.sh"
    )
    run(["tmux", "new-session", "-d", "-s", session, command], log_path)
    log(f"started training session {session}", log_path)


def run_eval(out_dir: Path, label_prefix: str, epoch: int, log_path: Path) -> Path:
    label = f"{label_prefix}_epoch{epoch}"
    checkpoint = out_dir / f"epoch{epoch}"
    run([PYTHON, str(EVAL_SCRIPT), "--checkpoint", str(checkpoint), "--label", label], log_path)
    return REPO / "eval_output/screwdriver_so101/model_eval" / label / "all_joint_focused_summary.json"


def summarize(epoch: int, summary_path: Path, log_path: Path) -> None:
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
            ,
            log_path,
        )
    log(f"epoch {epoch} all-joint eval summary: {summary_path}", log_path)


def main() -> None:
    args = _parse_args()
    log_path = Path(args.log).expanduser().resolve()
    out_dir = REPO / "my_data/training_pipeline/fine_tuning" / args.run_name
    epochs = [int(item.strip()) for item in args.epochs.split(",") if item.strip()]
    if not epochs:
        raise ValueError("--epochs cannot be empty")
    final_epoch = epochs[-1]
    for epoch in epochs:
        log(f"watching for epoch {epoch} checkpoint plus validation", log_path)
        wait_for_epoch(out_dir, epoch, args.poll_seconds, log_path)
        log(f"epoch {epoch} checkpoint and validation are ready; stopping training for eval", log_path)
        stop_training(args.session, log_path)
        summary = run_eval(out_dir, args.label_prefix, epoch, log_path)
        summarize(epoch, summary, log_path)
        if epoch == final_epoch and not args.resume_final:
            log(f"leaving run stopped after epoch {epoch} for review", log_path)
            return
        start_training(args.session, log_path)
    log("supervisor complete", log_path)


if __name__ == "__main__":
    main()
