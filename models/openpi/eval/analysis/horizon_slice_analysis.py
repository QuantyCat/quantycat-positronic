#!/usr/bin/env python3
"""
Analyse how eval metrics vary by action-horizon position.

Uses saved trace .npz files from openpi_lerobot_high_motion_eval.py
(--save-traces flag) to avoid re-running inference.

Produces three outputs in the eval label directory:
  horizon_slice_summary.json   -- full vs early-only diagonal metrics per joint
  horizon_degradation.json     -- per-position (k=0..19) metrics per joint
  horizon_degradation_report.md -- human-readable markdown table

Background / motivation
-----------------------
The standard eval collapses all 20 horizon positions into one flat pool when
computing slope/correlation/MAE.  At ~9 Hz inference against a 30 Hz dataset,
the robot executes only positions 0-2 before the next inference replaces the
chunk (~3.3 actions per chunk).  This script isolates those early positions so
the reported numbers reflect what the robot actually sees at runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[3]
EVAL_DIR = REPO / "models/openpi/eval/libs"
DEFAULT_EVAL_ROOT = REPO / "eval_output/screwdriver_so101/model_eval"
JOINT_LABELS = ("joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "gripper")
FOCUS_JOINTS = ("joint_0", "joint_1", "joint_2", "joint_3", "joint_4")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-label",
        required=True,
        help="Subdirectory name under --eval-root containing the trace npz files.",
    )
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument(
        "--early-n",
        type=int,
        default=3,
        help="Number of horizon positions to include in the 'early-only' slice "
             "(default 3 — ~111 ms at 30 Hz / 9 Hz inference).",
    )
    parser.add_argument(
        "--sign-eps",
        type=float,
        default=0.01,
        help="Dead-zone threshold for sign-agreement (matches main eval default).",
    )
    return parser.parse_args()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_batch_eval():
    sys.path.insert(0, str(EVAL_DIR))
    import episode_batch_eval as be
    return be


def _load_bounds(label_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    bounds_file = label_dir / "achieved_delta_action_bounds.json"
    if not bounds_file.exists():
        raise FileNotFoundError(f"Action bounds not found: {bounds_file}")
    data = json.loads(bounds_file.read_text())
    return np.asarray(data["min"], dtype=np.float32), np.asarray(data["max"], dtype=np.float32)


def _load_traces(label_dir: Path, joint: int) -> tuple[np.ndarray, np.ndarray]:
    npz_path = label_dir / f"broad_j{joint}_high_motion_top10" / "focused_high_motion_traces.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Trace file not found: {npz_path}\nRe-run the main eval with --save-traces.")
    d = np.load(npz_path)
    # shape: (cases, steps, horizon, dims)
    return d["pred"].astype(np.float32), d["gt"].astype(np.float32)


def _diagonal_metrics(summary: dict[str, Any], joint_name: str) -> dict[str, float | None]:
    conv = summary["convention_checks"][joint_name]
    norm_conv = summary["normalized_convention_checks"][joint_name]
    return {
        "sign_agreement": summary["per_joint"]["sign_agreement"][joint_name],
        "corr": conv["same_joint_corr"],
        "slope": conv["fit_same_joint"]["slope"],
        "norm_mae": summary["normalized_per_joint"]["mean_abs_error"][joint_name],
        "norm_slope": norm_conv["fit_same_joint"]["slope"],
    }


def _run_slice(
    be,
    pred: np.ndarray,
    gt: np.ndarray,
    horizon_slice: slice,
    sign_eps: float,
    min_vals: np.ndarray,
    max_vals: np.ndarray,
) -> dict[str, Any]:
    # pred/gt: (cases, steps, horizon, dims) → flatten cases+steps, then slice horizon
    p = pred[:, :, horizon_slice, :].reshape(-1, pred[:, :, horizon_slice, :].shape[2], pred.shape[3])
    g = gt[:, :, horizon_slice, :].reshape(-1, gt[:, :, horizon_slice, :].shape[2], gt.shape[3])
    return be._summarize(p, g, sign_eps, min_vals, max_vals)


def _fmt(v: float | None, precision: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{precision}f}"


def main() -> int:
    args = _parse_args()
    label_dir = (args.eval_root / args.eval_label).resolve()
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Eval label directory not found: {label_dir}")

    be = _load_batch_eval()
    min_vals, max_vals = _load_bounds(label_dir)

    full_results: dict[str, dict[str, Any]] = {}
    early_results: dict[str, dict[str, Any]] = {}
    degradation: dict[str, list[dict[str, Any]]] = {}

    n_horizon = None

    for joint in range(5):
        jname = f"joint_{joint}"
        pred, gt = _load_traces(label_dir, joint)
        # pred: (cases, steps, horizon, dims)
        if n_horizon is None:
            n_horizon = pred.shape[2]

        # full horizon
        full_sum = _run_slice(be, pred, gt, slice(None), args.sign_eps, min_vals, max_vals)
        full_results[jname] = _diagonal_metrics(full_sum, jname)

        # early-only slice
        early_sum = _run_slice(be, pred, gt, slice(0, args.early_n), args.sign_eps, min_vals, max_vals)
        early_results[jname] = _diagonal_metrics(early_sum, jname)

        # per-position breakdown
        pos_rows: list[dict[str, Any]] = []
        for k in range(n_horizon):
            pos_sum = _run_slice(be, pred, gt, slice(k, k + 1), args.sign_eps, min_vals, max_vals)
            pos_rows.append({"horizon_pos": k, **_diagonal_metrics(pos_sum, jname)})
        degradation[jname] = pos_rows

        print(f"j{joint} done  full_slope={_fmt(full_results[jname]['slope'])}  "
              f"early_slope={_fmt(early_results[jname]['slope'])}", flush=True)

    # ── JSON outputs ──────────────────────────────────────────────────────────

    summary_payload = {
        "created_at": _stamp(),
        "eval_label": args.eval_label,
        "early_n": args.early_n,
        "early_n_rationale": (
            f"dataset_fps=30, inference_hz=9 → ~3.3 actions/chunk; "
            f"early_n={args.early_n} covers positions 0..{args.early_n - 1}"
        ),
        "sign_eps": args.sign_eps,
        "n_horizon": n_horizon,
        "full_horizon": full_results,
        "early_only": early_results,
    }
    summary_path = label_dir / "horizon_slice_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    degradation_path = label_dir / "horizon_degradation.json"
    degradation_path.write_text(json.dumps({
        "created_at": _stamp(),
        "eval_label": args.eval_label,
        "sign_eps": args.sign_eps,
        "n_horizon": n_horizon,
        "per_joint": degradation,
    }, indent=2))

    # ── Markdown report ───────────────────────────────────────────────────────

    lines: list[str] = [
        "# Horizon Slice Analysis",
        "",
        f"Eval: `{args.eval_label}`  ",
        f"Early-only: positions 0–{args.early_n - 1} "
        f"(dataset 30 Hz / ~9 Hz inference ≈ {args.early_n} actions/chunk)  ",
        f"Created: {_stamp()}",
        "",
        "## Full vs Early-Only Diagonal Metrics",
        "",
        "| Joint | Full slope | Early slope | Δslope | Full sign | Early sign | Full norm_MAE | Early norm_MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for jname in FOCUS_JOINTS:
        f = full_results[jname]
        e = early_results[jname]
        delta = None if (f["slope"] is None or e["slope"] is None) else e["slope"] - f["slope"]
        lines.append(
            f"| {jname} "
            f"| {_fmt(f['slope'])} "
            f"| {_fmt(e['slope'])} "
            f"| {_fmt(delta, 3)} "
            f"| {_fmt(f['sign_agreement'])} "
            f"| {_fmt(e['sign_agreement'])} "
            f"| {_fmt(f['norm_mae'])} "
            f"| {_fmt(e['norm_mae'])} |"
        )

    lines += [
        "",
        "## Per-Horizon-Position Degradation",
        "",
        *(
            f"### {jname}"
            + "\n\n"
            + "| pos | slope | corr | sign_agree | norm_MAE |\n"
            + "|---:|---:|---:|---:|---:|\n"
            + "\n".join(
                f"| {row['horizon_pos']} "
                f"| {_fmt(row['slope'])} "
                f"| {_fmt(row['corr'])} "
                f"| {_fmt(row['sign_agreement'])} "
                f"| {_fmt(row['norm_mae'])} |"
                for row in degradation[jname]
            )
            for jname in FOCUS_JOINTS
        ),
        "",
    ]

    md_path = label_dir / "horizon_degradation_report.md"
    md_path.write_text("\n".join(lines) + "\n")

    print(f"\nWrote: {summary_path}")
    print(f"Wrote: {degradation_path}")
    print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
