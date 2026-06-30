#!/usr/bin/env python3
"""Score fixed action-horizon execution indices from saved eval traces.

The lag sweep asks whether pred[h] aligns better with gt[h + lag]. This script
asks the live-control question instead: if the controller consumes predicted
action[k] immediately, how does that command compare to the current target
gt[0]?

This is an offline approximation because it does not roll the robot state
forward after a different command. It is still useful for checking whether a
later horizon index helps a delayed wrist joint while damaging proximal joints.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


TRACE_RE = re.compile(r"broad_j(?P<joint>\d+)_high_motion_top(?P<top_k>\d+)")
REPORT_INDICES = (0, 1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 49)


def _parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label_dir", type=Path)
    parser.add_argument("--joints", default="0,1,2,3,4")
    parser.add_argument("--gt-index", type=int, default=0)
    parser.add_argument("--max-index", type=int, default=None)
    parser.add_argument("--sign-eps", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_float(value: float | np.floating[Any]) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _metrics(pred: np.ndarray, gt: np.ndarray, sign_eps: float) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    gt = np.asarray(gt, dtype=np.float64).reshape(-1)
    finite = np.isfinite(pred) & np.isfinite(gt)
    pred = pred[finite]
    gt = gt[finite]
    count = int(pred.size)
    if count == 0:
        return {
            "count": 0,
            "mean_abs_error": None,
            "sign_agreement": None,
            "sign_count": 0,
            "corr": None,
            "fit_slope": None,
            "gt_abs_mean": None,
            "pred_abs_mean": None,
            "pred_gt_abs_ratio": None,
        }

    centered_pred = pred - np.mean(pred)
    centered_gt = gt - np.mean(gt)
    denom = float(np.sqrt(np.sum(centered_pred**2) * np.sum(centered_gt**2)))
    corr = float(np.sum(centered_pred * centered_gt) / denom) if denom > 0 else math.nan
    slope_denom = float(np.sum(centered_gt**2))
    slope = float(np.sum(centered_gt * centered_pred) / slope_denom) if slope_denom > 0 else math.nan

    sign_mask = np.abs(gt) > sign_eps
    sign_count = int(np.sum(sign_mask))
    if sign_count:
        sign_agreement = float(np.mean(np.sign(pred[sign_mask]) == np.sign(gt[sign_mask])))
    else:
        sign_agreement = math.nan

    gt_abs = float(np.mean(np.abs(gt)))
    pred_abs = float(np.mean(np.abs(pred)))
    ratio = pred_abs / gt_abs if gt_abs > 0 else math.nan

    return {
        "count": count,
        "mean_abs_error": _safe_float(np.mean(np.abs(pred - gt))),
        "sign_agreement": _safe_float(sign_agreement),
        "sign_count": sign_count,
        "corr": _safe_float(corr),
        "fit_slope": _safe_float(slope),
        "gt_abs_mean": _safe_float(gt_abs),
        "pred_abs_mean": _safe_float(pred_abs),
        "pred_gt_abs_ratio": _safe_float(ratio),
    }


def _best_row(rows: list[dict[str, Any]], metric: str, minimize: bool = False) -> dict[str, Any] | None:
    valid = [row for row in rows if row.get(metric) is not None]
    if not valid:
        return None
    return min(valid, key=lambda row: row[metric]) if minimize else max(valid, key=lambda row: row[metric])


def _best_slope_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [row for row in rows if row.get("fit_slope") is not None]
    if not valid:
        return None
    return min(valid, key=lambda row: abs(row["fit_slope"] - 1.0))


def _trace_joint_from_name(trace_path: Path) -> int | None:
    match = TRACE_RE.fullmatch(trace_path.parent.name)
    return int(match.group("joint")) if match else None


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return None
    return _safe_float(np.mean(values))


def _analyze_trace(
    trace_path: Path,
    joints: list[int],
    gt_index: int,
    max_index: int | None,
    sign_eps: float,
) -> dict[str, Any]:
    data = np.load(trace_path)
    pred = np.asarray(data["pred"], dtype=np.float32)
    gt = np.asarray(data["gt"], dtype=np.float32)
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch in {trace_path}: {pred.shape} vs {gt.shape}")
    if pred.ndim != 4:
        raise ValueError(f"expected case x step x horizon x dim in {trace_path}, got {pred.shape}")

    horizon = pred.shape[2]
    if gt_index < 0 or gt_index >= horizon:
        raise ValueError(f"gt-index {gt_index} is outside horizon 0..{horizon - 1}")
    max_exec_index = horizon - 1 if max_index is None else min(max_index, horizon - 1)
    if max_exec_index < 0:
        raise ValueError(f"max-index {max_index} leaves no valid execution index")

    gt_current = gt[:, :, gt_index, :].reshape(-1, gt.shape[-1])
    per_joint: dict[str, Any] = {}
    per_index: list[dict[str, Any]] = []

    for joint in joints:
        joint_name = f"joint_{joint}"
        rows: list[dict[str, Any]] = []
        for exec_index in range(max_exec_index + 1):
            pred_exec = pred[:, :, exec_index, :].reshape(-1, pred.shape[-1])
            metrics = _metrics(pred_exec[:, joint], gt_current[:, joint], sign_eps)
            rows.append({"execution_index": exec_index, **metrics})

        zero = rows[0]
        best_mae = _best_row(rows, "mean_abs_error", minimize=True)
        best_corr = _best_row(rows, "corr")
        best_slope = _best_slope_row(rows)
        per_joint[joint_name] = {
            "indices": rows,
            "best": {
                "index_0": zero,
                "best_mae": best_mae,
                "best_corr": best_corr,
                "best_slope_closest_to_1": best_slope,
                "mae_gain_vs_index_0": (
                    None
                    if best_mae is None or zero["mean_abs_error"] is None
                    else _safe_float(zero["mean_abs_error"] - best_mae["mean_abs_error"])
                ),
                "corr_gain_vs_index_0": (
                    None
                    if best_corr is None or zero["corr"] is None
                    else _safe_float(best_corr["corr"] - zero["corr"])
                ),
                "slope_gain_at_best_corr_vs_index_0": (
                    None
                    if best_corr is None or best_corr["fit_slope"] is None or zero["fit_slope"] is None
                    else _safe_float(best_corr["fit_slope"] - zero["fit_slope"])
                ),
            },
        }

    for exec_index in range(max_exec_index + 1):
        joint_rows = [
            per_joint[f"joint_{joint}"]["indices"][exec_index]
            for joint in joints
        ]
        proximal_rows = [
            per_joint[f"joint_{joint}"]["indices"][exec_index]
            for joint in joints
            if joint <= 2
        ]
        distal_rows = [
            per_joint[f"joint_{joint}"]["indices"][exec_index]
            for joint in joints
            if joint >= 3
        ]
        per_index.append({
            "execution_index": exec_index,
            "all_joints_mean_abs_error": _mean_metric(joint_rows, "mean_abs_error"),
            "all_joints_mean_corr": _mean_metric(joint_rows, "corr"),
            "all_joints_mean_fit_slope": _mean_metric(joint_rows, "fit_slope"),
            "proximal_joints_mean_abs_error": _mean_metric(proximal_rows, "mean_abs_error"),
            "proximal_joints_mean_corr": _mean_metric(proximal_rows, "corr"),
            "proximal_joints_mean_fit_slope": _mean_metric(proximal_rows, "fit_slope"),
            "distal_joints_mean_abs_error": _mean_metric(distal_rows, "mean_abs_error"),
            "distal_joints_mean_corr": _mean_metric(distal_rows, "corr"),
            "distal_joints_mean_fit_slope": _mean_metric(distal_rows, "fit_slope"),
        })

    return {
        "trace": str(trace_path),
        "trace_name": trace_path.parent.name,
        "ranked_joint": _trace_joint_from_name(trace_path),
        "shape": list(pred.shape),
        "gt_index": gt_index,
        "max_execution_index": max_exec_index,
        "comparison": (
            "pred[:, :, execution_index, joint] is compared to "
            "gt[:, :, gt_index, joint]. Default gt_index=0 approximates commanding "
            "that horizon index immediately in live control."
        ),
        "per_joint": per_joint,
        "per_index_summary": per_index,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _selected_indices(max_index: int) -> list[int]:
    indices = [idx for idx in REPORT_INDICES if idx <= max_index]
    if max_index not in indices:
        indices.append(max_index)
    return sorted(set(indices))


def _write_report(payload: dict[str, Any], report: Path) -> None:
    lines = [
        "# Execution Index Sweep",
        "",
        f"Eval: `{payload['label_dir']}`  ",
        f"Ground-truth comparison index: `{payload['gt_index']}`  ",
        f"Created: `{payload['created_at']}`",
        "",
        "This compares `predicted action[k]` against the current ground-truth target "
        "`gt[0]`. It approximates what happens if live control consumes horizon "
        "index `k` immediately.",
        "",
        "## Best Diagonal Index",
        "",
        "| Trace | Joint | k0 slope | k0 corr | k0 MAE | best-MAE k | best MAE | slope there | corr there | best-corr k | best corr | slope there |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for trace_name, analysis in sorted(payload["analyses"].items()):
        ranked_joint = analysis["ranked_joint"]
        if ranked_joint is None:
            continue
        joint_name = f"joint_{ranked_joint}"
        if joint_name not in analysis["per_joint"]:
            continue
        best = analysis["per_joint"][joint_name]["best"]
        zero = best["index_0"]
        best_mae = best["best_mae"] or {}
        best_corr = best["best_corr"] or {}
        lines.append(
            f"| `{trace_name}` "
            f"| j{ranked_joint} "
            f"| {_fmt(zero['fit_slope'])} "
            f"| {_fmt(zero['corr'])} "
            f"| {_fmt(zero['mean_abs_error'])} "
            f"| {best_mae.get('execution_index', 'n/a')} "
            f"| {_fmt(best_mae.get('mean_abs_error'))} "
            f"| {_fmt(best_mae.get('fit_slope'))} "
            f"| {_fmt(best_mae.get('corr'))} "
            f"| {best_corr.get('execution_index', 'n/a')} "
            f"| {_fmt(best_corr.get('corr'))} "
            f"| {_fmt(best_corr.get('fit_slope'))} |"
        )

    lines += [
        "",
        "## Selected Diagonal Indices",
        "",
        "| Trace | Joint | k | slope | corr | MAE | pred/gt abs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for trace_name, analysis in sorted(payload["analyses"].items()):
        ranked_joint = analysis["ranked_joint"]
        if ranked_joint is None:
            continue
        joint_name = f"joint_{ranked_joint}"
        if joint_name not in analysis["per_joint"]:
            continue
        rows = analysis["per_joint"][joint_name]["indices"]
        for exec_index in _selected_indices(analysis["max_execution_index"]):
            row = rows[exec_index]
            lines.append(
                f"| `{trace_name}` "
                f"| j{ranked_joint} "
                f"| {exec_index} "
                f"| {_fmt(row['fit_slope'])} "
                f"| {_fmt(row['corr'])} "
                f"| {_fmt(row['mean_abs_error'])} "
                f"| {_fmt(row['pred_gt_abs_ratio'])} |"
            )

    j4_analysis = payload["analyses"].get("broad_j4_high_motion_top5")
    if j4_analysis is not None:
        lines += [
            "",
            "## j4 High-Motion Cross-Joint Cost",
            "",
            "| k | prox MAE | prox corr | prox slope | j3 slope | j3 corr | j4 slope | j4 corr | j4 MAE | j4 pred/gt abs |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        summaries = j4_analysis["per_index_summary"]
        j3_rows = j4_analysis["per_joint"].get("joint_3", {}).get("indices", [])
        j4_rows = j4_analysis["per_joint"].get("joint_4", {}).get("indices", [])
        for exec_index in _selected_indices(j4_analysis["max_execution_index"]):
            summary = summaries[exec_index]
            j3 = j3_rows[exec_index] if exec_index < len(j3_rows) else {}
            j4 = j4_rows[exec_index] if exec_index < len(j4_rows) else {}
            lines.append(
                f"| {exec_index} "
                f"| {_fmt(summary['proximal_joints_mean_abs_error'])} "
                f"| {_fmt(summary['proximal_joints_mean_corr'])} "
                f"| {_fmt(summary['proximal_joints_mean_fit_slope'])} "
                f"| {_fmt(j3.get('fit_slope'))} "
                f"| {_fmt(j3.get('corr'))} "
                f"| {_fmt(j4.get('fit_slope'))} "
                f"| {_fmt(j4.get('corr'))} "
                f"| {_fmt(j4.get('mean_abs_error'))} "
                f"| {_fmt(j4.get('pred_gt_abs_ratio'))} |"
            )

    lines += [
        "",
        "## Interpretation",
        "",
        "- If a later `k` improves j4 while proximal metrics degrade, whole-arm `actions[k]` is risky.",
        "- If a later `k` improves j4 and proximal metrics remain stable, an execution offset may be worth a live diagnostic.",
        "- If j4 best-MAE remains at `k=0`, the lag-sweep signal is probably not enough to justify consuming a later index directly.",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    label_dir = args.label_dir.expanduser().resolve()
    joints = _parse_ints(args.joints)
    output = args.output.expanduser().resolve() if args.output else label_dir / "execution_index_sweep.json"
    report = args.report.expanduser().resolve() if args.report else label_dir / "execution_index_sweep_report.md"

    traces = sorted(label_dir.glob("broad_j*_high_motion_top*/focused_high_motion_traces.npz"))
    if not traces:
        raise FileNotFoundError(f"No focused_high_motion_traces.npz files found under {label_dir}")

    analyses: dict[str, Any] = {}
    for trace in traces:
        analyses[trace.parent.name] = _analyze_trace(
            trace,
            joints=joints,
            gt_index=args.gt_index,
            max_index=args.max_index,
            sign_eps=args.sign_eps,
        )

    payload = {
        "created_at": _stamp(),
        "label_dir": str(label_dir),
        "joints": joints,
        "gt_index": args.gt_index,
        "requested_max_index": args.max_index,
        "sign_eps": args.sign_eps,
        "interpretation": (
            "Fixed execution-index sweep. Each predicted horizon index is scored "
            "as though it were commanded immediately against gt_index, default 0."
        ),
        "analyses": analyses,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(payload, report)
    print(f"Wrote execution index sweep: {output}")
    print(f"Wrote execution index report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
