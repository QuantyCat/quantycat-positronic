#!/usr/bin/env python3
"""Offline gain calibration for saved Dacha high-motion eval traces.

This evaluates per-joint gains applied to predicted target-current deltas:

    calibrated_delta[j] = pred_delta[j] * gain[j]

It uses saved high-motion trace .npz files, so it does not rerun policy
inference. The goal is to determine whether j2/j3/j4 are mostly amplitude
compressed, and whether gain can improve holdout metrics without breaking
correlation/sign agreement.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np


TARGET_JOINTS = (2, 3, 4)
FOCUS_JOINTS = (0, 1, 2, 3, 4)


def _parse_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _parse_float_list(value: str | None) -> list[float] | None:
    return None if value is None else _parse_floats(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--j2-gains", default="1.00,1.10,1.20,1.30,1.40,1.50,1.60,1.70,1.80")
    parser.add_argument("--j3-gains", default="1.00,1.10,1.20,1.30,1.40,1.50,1.60,1.70,1.80,1.90,2.00")
    parser.add_argument("--j4-gains", default="1.00,1.10,1.20,1.30,1.40,1.50,1.60,1.70,1.80,1.90,2.00")
    parser.add_argument("--sign-eps", type=float, default=0.01)
    parser.add_argument("--action-bounds-json", type=Path, default=None)
    parser.add_argument(
        "--clip-delta",
        default=None,
        help="Optional comma-separated per-dim delta clip, e.g. 3,3,4,3,3,1. Usually leave off for chunk-level scoring.",
    )
    parser.add_argument("--min-corr-drop", type=float, default=-0.03)
    parser.add_argument("--max-mae-increase", type=float, default=0.02)
    return parser.parse_args()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_float(value: float | np.floating[Any]) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _load_bounds(label_dir: Path, explicit: Path | None) -> tuple[np.ndarray, np.ndarray]:
    path = explicit if explicit is not None else label_dir / "commanded_delta_action_bounds.json"
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    return np.asarray(payload["min"], dtype=np.float64), np.asarray(payload["max"], dtype=np.float64)


def _load_traces(label_dir: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    traces: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for joint in FOCUS_JOINTS:
        matches = sorted(label_dir.glob(f"broad_j{joint}_high_motion_top*/focused_high_motion_traces.npz"))
        if not matches:
            raise FileNotFoundError(f"No trace found for broad_j{joint} under {label_dir}")
        data = np.load(matches[0])
        pred = np.asarray(data["pred"], dtype=np.float64)
        gt = np.asarray(data["gt"], dtype=np.float64)
        if pred.shape != gt.shape:
            raise ValueError(f"pred/gt shape mismatch in {matches[0]}: {pred.shape} vs {gt.shape}")
        traces[joint] = (pred, gt)
    return traces


def _normalize(values: np.ndarray, min_values: np.ndarray, max_values: np.ndarray) -> np.ndarray:
    return np.clip(2.0 * (values - min_values) / (max_values - min_values + 1e-8) - 1.0, -1.0, 1.0)


def _metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    sign_eps: float,
    min_values: np.ndarray,
    max_values: np.ndarray,
) -> dict[str, Any]:
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
            "normalized_mae": None,
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
    sign_agreement = float(np.mean(np.sign(pred[sign_mask]) == np.sign(gt[sign_mask]))) if sign_count else math.nan

    pred_norm = _normalize(pred, min_values, max_values)
    gt_norm = _normalize(gt, min_values, max_values)
    gt_abs = float(np.mean(np.abs(gt)))
    pred_abs = float(np.mean(np.abs(pred)))
    return {
        "count": count,
        "mean_abs_error": _safe_float(np.mean(np.abs(pred - gt))),
        "normalized_mae": _safe_float(np.mean(np.abs(pred_norm - gt_norm))),
        "sign_agreement": _safe_float(sign_agreement),
        "sign_count": sign_count,
        "corr": _safe_float(corr),
        "fit_slope": _safe_float(slope),
        "gt_abs_mean": _safe_float(gt_abs),
        "pred_abs_mean": _safe_float(pred_abs),
        "pred_gt_abs_ratio": _safe_float(pred_abs / gt_abs if gt_abs > 0 else math.nan),
    }


def _apply_gains(pred: np.ndarray, gains: dict[int, float], clip_delta: np.ndarray | None) -> np.ndarray:
    calibrated = pred.copy()
    for joint, gain in gains.items():
        calibrated[..., joint] *= gain
    if clip_delta is not None:
        calibrated = np.clip(calibrated, -clip_delta.reshape((1,) * (calibrated.ndim - 1) + (-1,)), clip_delta)
    return calibrated


def _score_candidate(
    traces: dict[int, tuple[np.ndarray, np.ndarray]],
    gains: dict[int, float],
    sign_eps: float,
    min_values: np.ndarray,
    max_values: np.ndarray,
    clip_delta: np.ndarray | None,
) -> dict[str, Any]:
    coverage: dict[str, Any] = {}
    diagonal: dict[str, Any] = {}
    all_focus: dict[str, dict[str, Any]] = {}

    for ranked_joint, (pred, gt) in traces.items():
        calibrated = _apply_gains(pred, gains, clip_delta)
        flat_pred = calibrated.reshape(-1, calibrated.shape[-1])
        flat_gt = gt.reshape(-1, gt.shape[-1])
        focus = {
            str(joint): _metrics(flat_pred[:, joint], flat_gt[:, joint], sign_eps, min_values[joint], max_values[joint])
            for joint in FOCUS_JOINTS
        }
        coverage[str(ranked_joint)] = focus
        diagonal[str(ranked_joint)] = focus[str(ranked_joint)]
        all_focus[str(ranked_joint)] = focus

    target = [diagonal[str(joint)] for joint in TARGET_JOINTS]
    mean_mae = float(np.mean([row["normalized_mae"] for row in target if row["normalized_mae"] is not None]))
    mean_slope_error = float(np.mean([abs(row["fit_slope"] - 1.0) for row in target if row["fit_slope"] is not None]))
    mean_corr = float(np.mean([row["corr"] for row in target if row["corr"] is not None]))
    mean_sign = float(np.mean([row["sign_agreement"] for row in target if row["sign_agreement"] is not None]))
    mean_gain = float(np.mean([gains[joint] for joint in TARGET_JOINTS]))
    score = mean_mae + 0.04 * mean_slope_error - 0.01 * mean_corr + 0.001 * (mean_gain - 1.0)

    return {
        "gains": {str(joint): gains[joint] for joint in TARGET_JOINTS},
        "score": score,
        "mean_target_norm_mae": mean_mae,
        "mean_target_slope_abs_error": mean_slope_error,
        "mean_target_corr": mean_corr,
        "mean_target_sign_agreement": mean_sign,
        "diagonal": diagonal,
        "coverage": coverage,
    }


def _gain_sum(row: dict[str, Any]) -> float:
    return sum(float(row["gains"][str(joint)]) - 1.0 for joint in TARGET_JOINTS)


def _within_constraints(row: dict[str, Any], baseline: dict[str, Any], min_corr_drop: float, max_mae_increase: float) -> bool:
    for joint in TARGET_JOINTS:
        key = str(joint)
        base = baseline["diagonal"][key]
        cand = row["diagonal"][key]
        if cand["corr"] is None or base["corr"] is None or cand["corr"] - base["corr"] < min_corr_drop:
            return False
        if (
            cand["normalized_mae"] is None
            or base["normalized_mae"] is None
            or cand["normalized_mae"] - base["normalized_mae"] > max_mae_increase
        ):
            return False
    return True


def _best_rows(rows: list[dict[str, Any]], baseline: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    by_score = sorted(rows, key=lambda row: row["score"])
    by_mae = sorted(rows, key=lambda row: (row["mean_target_norm_mae"], _gain_sum(row)))
    by_slope = sorted(rows, key=lambda row: (row["mean_target_slope_abs_error"], row["mean_target_norm_mae"], _gain_sum(row)))
    constrained = [
        row
        for row in rows
        if _within_constraints(row, baseline, args.min_corr_drop, args.max_mae_increase)
        and any(float(row["gains"][str(joint)]) > 1.0 for joint in TARGET_JOINTS)
    ]
    constrained_by_slope = sorted(
        constrained,
        key=lambda row: (row["mean_target_slope_abs_error"], row["mean_target_norm_mae"], _gain_sum(row)),
    )
    conservative = min(
        constrained_by_slope[:25] or constrained or [baseline],
        key=lambda row: (_gain_sum(row), row["mean_target_slope_abs_error"], row["mean_target_norm_mae"]),
    )
    return {
        "score_selected": by_score[0],
        "mae_selected": by_mae[0],
        "slope_selected": by_slope[0],
        "constrained_slope_selected": constrained_by_slope[0] if constrained_by_slope else None,
        "conservative_selected": conservative,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _row_line(rank: int, row: dict[str, Any]) -> str:
    d = row["diagonal"]
    return (
        f"| {rank} | {row['gains']['2']:.2f} | {row['gains']['3']:.2f} | {row['gains']['4']:.2f} "
        f"| {_fmt(d['2']['fit_slope'])} | {_fmt(d['3']['fit_slope'])} | {_fmt(d['4']['fit_slope'])} "
        f"| {_fmt(d['2']['normalized_mae'])} | {_fmt(d['3']['normalized_mae'])} | {_fmt(d['4']['normalized_mae'])} "
        f"| {_fmt(row['mean_target_norm_mae'])} | {_fmt(row['mean_target_corr'])} | {_fmt(row['score'], 4)} |"
    )


def _write_report(payload: dict[str, Any], output: Path) -> None:
    lines = [
        "# Dacha Gain Calibration",
        "",
        f"Eval: `{payload['label_dir']}`  ",
        f"Created: `{payload['created_at']}`",
        "",
        "Applied to saved predicted deltas:",
        "",
        "```text",
        "calibrated_delta[j] = predicted_delta[j] * gain[j]",
        "```",
        "",
        "## Selected Candidates",
        "",
        "| Selection | j2 | j3 | j4 | mean MAE | mean slope error | mean corr |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in payload["selected"].items():
        if row is None:
            continue
        lines.append(
            f"| {name} | {row['gains']['2']:.2f} | {row['gains']['3']:.2f} | {row['gains']['4']:.2f} "
            f"| {_fmt(row['mean_target_norm_mae'])} | {_fmt(row['mean_target_slope_abs_error'])} "
            f"| {_fmt(row['mean_target_corr'])} |"
        )

    lines += [
        "",
        "## Top Score Rows",
        "",
        "| Rank | j2 | j3 | j4 | j2 slope | j3 slope | j4 slope | j2 MAE | j3 MAE | j4 MAE | mean MAE | mean corr | score |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(payload["ranked_by_score"][:16], start=1):
        lines.append(_row_line(rank, row))

    lines += [
        "",
        "## Baseline Vs Conservative",
        "",
        "| Joint | baseline slope | conservative slope | baseline corr | conservative corr | baseline MAE | conservative MAE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    baseline = payload["baseline"]["diagonal"]
    conservative = payload["selected"]["conservative_selected"]["diagonal"]
    for joint in FOCUS_JOINTS:
        key = str(joint)
        lines.append(
            f"| j{joint} | {_fmt(baseline[key]['fit_slope'])} | {_fmt(conservative[key]['fit_slope'])} "
            f"| {_fmt(baseline[key]['corr'])} | {_fmt(conservative[key]['corr'])} "
            f"| {_fmt(baseline[key]['normalized_mae'])} | {_fmt(conservative[key]['normalized_mae'])} |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    label_dir = args.label_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else label_dir / "gain_calibration"
    output_dir.mkdir(parents=True, exist_ok=True)

    min_values, max_values = _load_bounds(label_dir, args.action_bounds_json)
    traces = _load_traces(label_dir)
    clip_delta = _parse_float_list(args.clip_delta)
    clip_arr = np.asarray(clip_delta, dtype=np.float64) if clip_delta is not None else None

    rows: list[dict[str, Any]] = []
    for j2, j3, j4 in product(_parse_floats(args.j2_gains), _parse_floats(args.j3_gains), _parse_floats(args.j4_gains)):
        rows.append(
            _score_candidate(
                traces,
                {2: j2, 3: j3, 4: j4},
                args.sign_eps,
                min_values,
                max_values,
                clip_arr,
            )
        )

    baseline = next(row for row in rows if row["gains"] == {"2": 1.0, "3": 1.0, "4": 1.0})
    selected = _best_rows(rows, baseline, args)
    ranked_by_score = sorted(rows, key=lambda row: row["score"])
    payload = {
        "created_at": _stamp(),
        "label_dir": str(label_dir),
        "output_dir": str(output_dir),
        "clip_delta": clip_delta,
        "sign_eps": args.sign_eps,
        "constraints": {
            "min_corr_drop": args.min_corr_drop,
            "max_mae_increase": args.max_mae_increase,
        },
        "swept_gains": {
            "2": _parse_floats(args.j2_gains),
            "3": _parse_floats(args.j3_gains),
            "4": _parse_floats(args.j4_gains),
        },
        "baseline": baseline,
        "selected": selected,
        "ranked_by_score": ranked_by_score,
        "ranked_by_slope": sorted(rows, key=lambda row: (row["mean_target_slope_abs_error"], row["mean_target_norm_mae"]))[:32],
        "ranked_by_mae": sorted(rows, key=lambda row: (row["mean_target_norm_mae"], _gain_sum(row)))[:32],
    }
    json_path = output_dir / "gain_calibration_summary.json"
    report_path = output_dir / "gain_calibration_summary.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(payload, report_path)

    print(f"Wrote gain calibration summary: {json_path}")
    print(f"Wrote gain calibration report: {report_path}")
    for name, row in selected.items():
        if row is None:
            continue
        print(
            f"{name}: "
            f"j2={row['gains']['2']:.2f} j3={row['gains']['3']:.2f} j4={row['gains']['4']:.2f} "
            f"mean_mae={row['mean_target_norm_mae']:.4f} "
            f"mean_slope_err={row['mean_target_slope_abs_error']:.4f} "
            f"mean_corr={row['mean_target_corr']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
