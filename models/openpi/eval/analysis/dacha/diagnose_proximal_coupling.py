#!/usr/bin/env python3
"""Analyze whether distal joint errors track proximal prediction error.

The Dacha high-motion eval writes trace arrays for each broad high-motion joint:
    pred, gt: case x eval_step x action_horizon x action_dim

This diagnostic buckets individual predicted targets by proximal error
(`j0/j1` by default) and recomputes distal `j2/j3/j4` metrics within each
bucket. If distal slopes/correlations collapse in the high-proximal-error
bucket, wrist/elbow failures are likely downstream of proximal plan mismatch.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label_dir", type=Path)
    parser.add_argument("--proximal-joints", default="0,1")
    parser.add_argument("--distal-joints", default="2,3,4")
    parser.add_argument("--sign-eps", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=None)
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

    return {
        "count": count,
        "mean_abs_error": _safe_float(np.mean(np.abs(pred - gt))),
        "sign_agreement": _safe_float(sign_agreement),
        "sign_count": sign_count,
        "corr": _safe_float(corr),
        "fit_slope": _safe_float(slope),
        "gt_abs_mean": _safe_float(np.mean(np.abs(gt))),
        "pred_abs_mean": _safe_float(np.mean(np.abs(pred))),
    }


def _bucket_masks(proximal_error: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    flat = np.asarray(proximal_error, dtype=np.float64).reshape(-1)
    finite = np.isfinite(flat)
    values = flat[finite]
    if values.size == 0:
        empty = np.zeros_like(flat, dtype=bool)
        return {"all": empty, "low": empty, "mid": empty, "high": empty}, {}

    q33, q67 = np.quantile(values, [1 / 3, 2 / 3])
    masks = {
        "all": finite,
        "low": finite & (flat <= q33),
        "mid": finite & (flat > q33) & (flat <= q67),
        "high": finite & (flat > q67),
    }
    metadata = {
        "count": int(values.size),
        "mean": _safe_float(np.mean(values)),
        "q33": _safe_float(q33),
        "q67": _safe_float(q67),
        "q90": _safe_float(np.quantile(values, 0.90)),
        "q99": _safe_float(np.quantile(values, 0.99)),
    }
    return masks, metadata


def _analyze_trace(trace_path: Path, proximal_joints: list[int], distal_joints: list[int], sign_eps: float) -> dict[str, Any]:
    data = np.load(trace_path)
    pred = np.asarray(data["pred"], dtype=np.float32)
    gt = np.asarray(data["gt"], dtype=np.float32)
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch in {trace_path}: {pred.shape} vs {gt.shape}")
    if pred.ndim != 4:
        raise ValueError(f"expected trace shape case x step x horizon x dim in {trace_path}, got {pred.shape}")

    error = pred - gt
    proximal_error = np.mean(np.abs(error[..., proximal_joints]), axis=-1).reshape(-1)
    flat_pred = pred.reshape(-1, pred.shape[-1])
    flat_gt = gt.reshape(-1, gt.shape[-1])
    masks, proximal_summary = _bucket_masks(proximal_error)

    buckets: dict[str, Any] = {}
    for bucket_name, mask in masks.items():
        bucket: dict[str, Any] = {
            "sample_count": int(np.sum(mask)),
            "proximal_error_mean": _safe_float(np.mean(proximal_error[mask])) if np.any(mask) else None,
            "proximal_joint_metrics": {},
            "distal_joint_metrics": {},
        }
        for joint in proximal_joints:
            bucket["proximal_joint_metrics"][f"joint_{joint}"] = _metrics(flat_pred[mask, joint], flat_gt[mask, joint], sign_eps)
        for joint in distal_joints:
            bucket["distal_joint_metrics"][f"joint_{joint}"] = _metrics(flat_pred[mask, joint], flat_gt[mask, joint], sign_eps)
        buckets[bucket_name] = bucket

    return {
        "trace": str(trace_path),
        "shape": list(pred.shape),
        "proximal_error": proximal_summary,
        "buckets": buckets,
    }


def main() -> int:
    args = _parse_args()
    label_dir = args.label_dir.expanduser().resolve()
    proximal_joints = _parse_ints(args.proximal_joints)
    distal_joints = _parse_ints(args.distal_joints)
    output = args.output.expanduser().resolve() if args.output else label_dir / "proximal_coupling_diagnostic.json"

    traces = sorted(label_dir.glob("broad_j*_high_motion_top*/focused_high_motion_traces.npz"))
    if not traces:
        raise FileNotFoundError(f"No focused_high_motion_traces.npz files found under {label_dir}")

    analyses: dict[str, Any] = {}
    for trace in traces:
        analyses[trace.parent.name] = _analyze_trace(trace, proximal_joints, distal_joints, args.sign_eps)

    payload = {
        "created_at": _stamp(),
        "label_dir": str(label_dir),
        "proximal_joints": proximal_joints,
        "distal_joints": distal_joints,
        "bucket_definition": "low/mid/high thirds by mean absolute j0/j1 prediction error per predicted target",
        "interpretation_hint": (
            "If distal corr/slope/sign degrade sharply from low to high proximal-error buckets, "
            "distal failure is likely coupled to proximal plan mismatch."
        ),
        "analyses": analyses,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote proximal coupling diagnostic: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
