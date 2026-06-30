#!/usr/bin/env python3
"""Sweep horizon lags in saved Dacha high-motion eval traces.

The high-motion eval writes trace arrays:
    pred, gt: case x eval_step x action_horizon x action_dim

Both arrays contain target-current deltas from the same eval origin step. This
diagnostic shifts ground truth along the action-horizon axis and recomputes
joint metrics. If slope/correlation improve at a nonzero lag, the policy may be
predicting the right motion too early or too late relative to the labels.
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


def _parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label_dir", type=Path)
    parser.add_argument("--joints", default="0,1,2,3,4")
    parser.add_argument("--lag-min", type=int, default=-25)
    parser.add_argument("--lag-max", type=int, default=25)
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


def _lagged_views(pred: np.ndarray, gt: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    horizon = pred.shape[2]
    if abs(lag) >= horizon:
        raise ValueError(f"lag {lag} leaves no overlapping horizon positions for horizon {horizon}")

    if lag > 0:
        return pred[:, :, : horizon - lag, :], gt[:, :, lag:, :]
    if lag < 0:
        shift = -lag
        return pred[:, :, shift:, :], gt[:, :, : horizon - shift, :]
    return pred, gt


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


def _analyze_trace(
    trace_path: Path,
    joints: list[int],
    lag_min: int,
    lag_max: int,
    sign_eps: float,
) -> dict[str, Any]:
    data = np.load(trace_path)
    pred = np.asarray(data["pred"], dtype=np.float32)
    gt = np.asarray(data["gt"], dtype=np.float32)
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch in {trace_path}: {pred.shape} vs {gt.shape}")
    if pred.ndim != 4:
        raise ValueError(f"expected case x step x horizon x dim in {trace_path}, got {pred.shape}")

    lag_min = max(lag_min, -(pred.shape[2] - 1))
    lag_max = min(lag_max, pred.shape[2] - 1)
    lag_rows: dict[str, list[dict[str, Any]]] = {f"joint_{joint}": [] for joint in joints}

    for lag in range(lag_min, lag_max + 1):
        lagged_pred, lagged_gt = _lagged_views(pred, gt, lag)
        flat_pred = lagged_pred.reshape(-1, lagged_pred.shape[-1])
        flat_gt = lagged_gt.reshape(-1, lagged_gt.shape[-1])
        overlap = int(lagged_pred.shape[2])
        for joint in joints:
            metrics = _metrics(flat_pred[:, joint], flat_gt[:, joint], sign_eps)
            lag_rows[f"joint_{joint}"].append({"lag": lag, "overlap_horizon": overlap, **metrics})

    best: dict[str, Any] = {}
    for joint_name, rows in lag_rows.items():
        zero = next(row for row in rows if row["lag"] == 0)
        best_corr = _best_row(rows, "corr")
        best_mae = _best_row(rows, "mean_abs_error", minimize=True)
        best_slope = _best_slope_row(rows)
        best[joint_name] = {
            "zero_lag": zero,
            "best_corr": best_corr,
            "best_mae": best_mae,
            "best_slope_closest_to_1": best_slope,
            "corr_gain_vs_zero": (
                None
                if best_corr is None or zero["corr"] is None
                else _safe_float(best_corr["corr"] - zero["corr"])
            ),
            "slope_gain_at_best_corr_vs_zero": (
                None
                if best_corr is None or best_corr["fit_slope"] is None or zero["fit_slope"] is None
                else _safe_float(best_corr["fit_slope"] - zero["fit_slope"])
            ),
        }

    return {
        "trace": str(trace_path),
        "trace_name": trace_path.parent.name,
        "ranked_joint": _trace_joint_from_name(trace_path),
        "shape": list(pred.shape),
        "lag_definition": (
            "lag > 0 compares pred[h] to later gt[h+lag]; lag < 0 compares pred[h] "
            "to earlier gt[h+lag]. Nonzero best lags suggest temporal offset within the action chunk."
        ),
        "per_joint": {
            joint_name: {
                "lags": rows,
                "best": best[joint_name],
            }
            for joint_name, rows in lag_rows.items()
        },
    }


def _fmt(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _write_report(payload: dict[str, Any], report: Path) -> None:
    lines = [
        "# Lag Sweep Diagnostic",
        "",
        f"Eval: `{payload['label_dir']}`  ",
        f"Lags: `{payload['lag_min']}..{payload['lag_max']}`  ",
        f"Created: `{payload['created_at']}`",
        "",
        "Positive lag compares a prediction at horizon `h` to later ground truth `h+lag`; "
        "negative lag compares it to earlier ground truth.",
        "",
        "## Diagonal High-Motion Joints",
        "",
        "| Trace | Joint | zero slope | zero corr | best-corr lag | best corr | slope at best corr | corr gain | best-slope lag | best slope |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for trace_name, analysis in sorted(payload["analyses"].items()):
        ranked_joint = analysis["ranked_joint"]
        if ranked_joint is None:
            continue
        joint_name = f"joint_{ranked_joint}"
        if joint_name not in analysis["per_joint"]:
            continue
        best = analysis["per_joint"][joint_name]["best"]
        zero = best["zero_lag"]
        best_corr = best["best_corr"] or {}
        best_slope = best["best_slope_closest_to_1"] or {}
        lines.append(
            f"| `{trace_name}` "
            f"| j{ranked_joint} "
            f"| {_fmt(zero['fit_slope'])} "
            f"| {_fmt(zero['corr'])} "
            f"| {best_corr.get('lag', 'n/a')} "
            f"| {_fmt(best_corr.get('corr'))} "
            f"| {_fmt(best_corr.get('fit_slope'))} "
            f"| {_fmt(best['corr_gain_vs_zero'])} "
            f"| {best_slope.get('lag', 'n/a')} "
            f"| {_fmt(best_slope.get('fit_slope'))} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- If best-corr lag is consistently nonzero and materially improves corr/slope, timing is a plausible contributor.",
        "- If lag 0 remains best or gains are tiny, the chronic slope issue is probably not explained by simple horizon timing.",
        "- If best-slope improves at a lag but corr does not, the amplitude may be phase-sensitive but the shape is still not aligned.",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    label_dir = args.label_dir.expanduser().resolve()
    joints = _parse_ints(args.joints)
    output = args.output.expanduser().resolve() if args.output else label_dir / "lag_sweep_diagnostic.json"
    report = args.report.expanduser().resolve() if args.report else label_dir / "lag_sweep_report.md"

    traces = sorted(label_dir.glob("broad_j*_high_motion_top*/focused_high_motion_traces.npz"))
    if not traces:
        raise FileNotFoundError(f"No focused_high_motion_traces.npz files found under {label_dir}")

    analyses: dict[str, Any] = {}
    actual_lag_min: int | None = None
    actual_lag_max: int | None = None
    for trace in traces:
        analysis = _analyze_trace(trace, joints, args.lag_min, args.lag_max, args.sign_eps)
        analyses[trace.parent.name] = analysis
        lags = [row["lag"] for row in next(iter(analysis["per_joint"].values()))["lags"]]
        actual_lag_min = min(lags) if actual_lag_min is None else min(actual_lag_min, min(lags))
        actual_lag_max = max(lags) if actual_lag_max is None else max(actual_lag_max, max(lags))

    payload = {
        "created_at": _stamp(),
        "label_dir": str(label_dir),
        "joints": joints,
        "requested_lag_min": args.lag_min,
        "requested_lag_max": args.lag_max,
        "lag_min": actual_lag_min,
        "lag_max": actual_lag_max,
        "sign_eps": args.sign_eps,
        "lag_definition": (
            "lag > 0 means pred[h] is scored against later gt[h+lag]; "
            "lag < 0 means pred[h] is scored against earlier gt[h+lag]."
        ),
        "analyses": analyses,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(payload, report)
    print(f"Wrote lag sweep diagnostic: {output}")
    print(f"Wrote lag sweep report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
