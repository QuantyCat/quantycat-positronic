#!/usr/bin/env python3
"""Compare sign and amplitude behavior across focused trace NPZ files."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def fit_slope_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.std(x) == 0:
        return float("nan"), float("nan")
    dx = x - x.mean()
    dy = y - y.mean()
    slope = float(np.dot(dx, dy) / np.dot(dx, dx))
    corr = float(np.corrcoef(x, y)[0, 1]) if np.std(y) > 0 else float("nan")
    return slope, corr


def fmt(value: float, precision: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:.{precision}f}"


def active_horizon_values(
    pred: np.ndarray,
    gt: np.ndarray,
    horizon: int,
    joint: int,
    eps: float,
    center: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    pred_values = pred[:, :, horizon, joint].reshape(-1) - center
    gt_values = gt[:, :, horizon, joint].reshape(-1) - center
    active = np.abs(gt_values) > eps
    return pred_values[active], gt_values[active]


def metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    horizon: int,
    joint: int,
    eps: float,
    center: float = 0.0,
) -> dict[str, float]:
    pred_values, gt_values = active_horizon_values(pred, gt, horizon, joint, eps, center)
    slope, corr = fit_slope_corr(gt_values, pred_values)
    same_sign = np.sign(pred_values) == np.sign(gt_values)
    wrong_sign = ~same_sign
    same_mag = float(
        np.mean(np.abs(pred_values[same_sign]))
        / (np.mean(np.abs(gt_values[same_sign])) + 1e-12)
    ) if np.any(same_sign) else float("nan")
    wrong_mag = float(
        np.mean(np.abs(pred_values[wrong_sign]))
        / (np.mean(np.abs(gt_values[wrong_sign])) + 1e-12)
    ) if np.any(wrong_sign) else float("nan")
    return {
        "n": int(gt_values.size),
        "slope": slope,
        "corr": corr,
        "sign": float(np.mean(same_sign)),
        "mag_ratio": float(np.mean(np.abs(pred_values)) / (np.mean(np.abs(gt_values)) + 1e-12)),
        "pred_mean": float(np.mean(pred_values)),
        "gt_mean": float(np.mean(gt_values)),
        "pred_abs": float(np.mean(np.abs(pred_values))),
        "gt_abs": float(np.mean(np.abs(gt_values))),
        "same_mag": same_mag,
        "wrong_mag": wrong_mag,
        "wrong_frac": float(np.mean(wrong_sign)),
        "pred_pos_frac": float(np.mean(pred_values > 0)),
        "gt_pos_frac": float(np.mean(gt_values > 0)),
    }


def confusion_by_gt_direction(
    pred: np.ndarray,
    gt: np.ndarray,
    horizon: int,
    joint: int,
    eps: float,
    center: float = 0.0,
) -> dict[str, dict[str, float]]:
    pred_values, gt_values = active_horizon_values(pred, gt, horizon, joint, eps, center)
    pred_positive = pred_values > 0
    buckets = {
        "gt_pos": gt_values > 0,
        "gt_neg": gt_values < 0,
    }
    result: dict[str, dict[str, float]] = {}
    for name, mask in buckets.items():
        if not np.any(mask):
            continue
        slope, corr = fit_slope_corr(gt_values[mask], pred_values[mask])
        result[name] = {
            "n": int(np.sum(mask)),
            "pred_pos": float(np.mean(pred_positive[mask])),
            "pred_neg": float(np.mean(~pred_positive[mask])),
            "slope": slope,
            "corr": corr,
            "pred_abs": float(np.mean(np.abs(pred_values[mask]))),
            "gt_abs": float(np.mean(np.abs(gt_values[mask]))),
        }
    return result


def temporal_offsets(
    pred: np.ndarray,
    gt: np.ndarray,
    joint: int,
    eps: float,
    max_offset: int,
    center: float = 0.0,
) -> list[tuple[int, float, float, float]]:
    pred_h1 = pred[:, :, 0, joint] - center
    gt_h1 = gt[:, :, 0, joint] - center
    rows: list[tuple[int, float, float, float]] = []
    for offset in range(-max_offset, max_offset + 1):
        pred_chunks = []
        gt_chunks = []
        for case_index in range(pred_h1.shape[0]):
            if offset >= 0:
                pred_case = pred_h1[case_index, : pred_h1.shape[1] - offset]
                gt_case = gt_h1[case_index, offset:]
            else:
                pred_case = pred_h1[case_index, -offset:]
                gt_case = gt_h1[case_index, : gt_h1.shape[1] + offset]
            pred_chunks.append(pred_case)
            gt_chunks.append(gt_case)
        pred_values = np.concatenate(pred_chunks)
        gt_values = np.concatenate(gt_chunks)
        active = np.abs(gt_values) > eps
        pred_values = pred_values[active]
        gt_values = gt_values[active]
        slope, corr = fit_slope_corr(gt_values, pred_values)
        sign = float(np.mean(np.sign(pred_values) == np.sign(gt_values)))
        rows.append((offset, slope, corr, sign))
    return rows


def load_trace(path: Path) -> tuple[np.ndarray, np.ndarray]:
    trace = np.load(path, allow_pickle=True)
    return trace["pred"], trace["gt"]


def load_action_zero_norm(path: Path, action_dim: int) -> np.ndarray:
    mins: list[float] = []
    maxs: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("Dim "):
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 3:
            continue
        mins.append(float(parts[1]))
        maxs.append(float(parts[2]))
        if len(mins) == action_dim:
            break
    if len(mins) != action_dim:
        raise ValueError(f"expected {action_dim} action stat rows in {path}, got {len(mins)}")
    return np.asarray(
        [2.0 * (0.0 - min_value) / max(max_value - min_value, 1e-8) - 1.0 for min_value, max_value in zip(mins, maxs)],
        dtype=np.float32,
    )


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("runs must use name=/path/to/traces.npz")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("run name cannot be empty")
    return name, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--joint", type=int, default=3)
    parser.add_argument("--sign-eps", type=float, default=0.01)
    parser.add_argument("--max-offset", type=int, default=5)
    parser.add_argument(
        "--action-stats",
        type=Path,
        default=None,
        help="Optional min_max_action.txt used to report raw-motor-zero-centered metrics.",
    )
    args = parser.parse_args()

    loaded = {name: load_trace(path) for name, path in args.run}
    first_pred = next(iter(loaded.values()))[0]
    zero_norm = None
    if args.action_stats is not None:
        zero_norm = load_action_zero_norm(args.action_stats, first_pred.shape[-1])
    lines: list[str] = []
    lines.append("# Scratch Reproducibility Failure Analysis")
    lines.append("")
    lines.append(
        "All metrics use the same focused trace windows. H1 means the deployment action "
        "at action index 0."
    )
    lines.append("")
    lines.append("## H1 Joint Metrics")
    lines.append("")
    lines.append(
        "| run | slope | corr | sign | mag_ratio | pred_abs | gt_abs | pred_pos | "
        "gt_pos | wrong_sign | same_sign_mag | wrong_sign_mag |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, (pred, gt) in loaded.items():
        row = metrics(pred, gt, horizon=0, joint=args.joint, eps=args.sign_eps)
        lines.append(
            f"| {name} | {fmt(row['slope'])} | {fmt(row['corr'])} | {fmt(row['sign'])} | "
            f"{fmt(row['mag_ratio'])} | {fmt(row['pred_abs'], 4)} | {fmt(row['gt_abs'], 4)} | "
            f"{fmt(row['pred_pos_frac'])} | {fmt(row['gt_pos_frac'])} | {fmt(row['wrong_frac'])} | "
            f"{fmt(row['same_mag'])} | {fmt(row['wrong_mag'])} |"
        )

    if zero_norm is not None:
        center = float(zero_norm[args.joint])
        lines.append("")
        lines.append("## H1 Joint Metrics Around Raw Motor Zero")
        lines.append("")
        lines.append(f"Normalized center for joint {args.joint}: `{fmt(center, 6)}`")
        lines.append("")
        lines.append(
            "| run | slope | corr | sign | mag_ratio | pred_abs | gt_abs | pred_pos | "
            "gt_pos | wrong_sign | same_sign_mag | wrong_sign_mag |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for name, (pred, gt) in loaded.items():
            row = metrics(pred, gt, horizon=0, joint=args.joint, eps=args.sign_eps, center=center)
            lines.append(
                f"| {name} | {fmt(row['slope'])} | {fmt(row['corr'])} | {fmt(row['sign'])} | "
                f"{fmt(row['mag_ratio'])} | {fmt(row['pred_abs'], 4)} | {fmt(row['gt_abs'], 4)} | "
                f"{fmt(row['pred_pos_frac'])} | {fmt(row['gt_pos_frac'])} | {fmt(row['wrong_frac'])} | "
                f"{fmt(row['same_mag'])} | {fmt(row['wrong_mag'])} |"
            )

    lines.append("")
    lines.append("## Sign Confusion By GT Direction")
    for name, (pred, gt) in loaded.items():
        lines.append("")
        lines.append(f"### {name}")
        lines.append("| bucket | n | pred_pos | pred_neg | slope | corr | pred_abs | gt_abs |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for bucket, row in confusion_by_gt_direction(
            pred, gt, horizon=0, joint=args.joint, eps=args.sign_eps
        ).items():
            lines.append(
                f"| {bucket} | {int(row['n'])} | {fmt(row['pred_pos'])} | "
                f"{fmt(row['pred_neg'])} | {fmt(row['slope'])} | {fmt(row['corr'])} | "
                f"{fmt(row['pred_abs'], 4)} | {fmt(row['gt_abs'], 4)} |"
            )

    lines.append("")
    lines.append("## H1 Temporal Offset Check")
    lines.append("")
    lines.append(
        "Positive offset compares prediction at step t to GT later in the window; "
        "negative offset compares prediction at t to earlier GT."
    )
    for name, (pred, gt) in loaded.items():
        rows = temporal_offsets(pred, gt, joint=args.joint, eps=args.sign_eps, max_offset=args.max_offset)
        best = max(rows, key=lambda item: -999.0 if math.isnan(item[2]) else item[2])
        lines.append("")
        lines.append(
            f"### {name} best_corr_offset={best[0]} corr={fmt(best[2])} "
            f"slope={fmt(best[1])} sign={fmt(best[3])}"
        )
        lines.append("| offset | slope | corr | sign |")
        lines.append("| ---: | ---: | ---: | ---: |")
        for offset, slope, corr, sign in rows:
            lines.append(f"| {offset} | {fmt(slope)} | {fmt(corr)} | {fmt(sign)} |")

    lines.append("")
    lines.append("## Per-Horizon Joint Metrics")
    for name, (pred, gt) in loaded.items():
        lines.append("")
        lines.append(f"### {name}")
        lines.append("| horizon | slope | corr | sign | mag_ratio |")
        lines.append("| ---: | ---: | ---: | ---: | ---: |")
        for horizon in range(pred.shape[2]):
            row = metrics(pred, gt, horizon=horizon, joint=args.joint, eps=args.sign_eps)
            lines.append(
                f"| {horizon + 1} | {fmt(row['slope'])} | {fmt(row['corr'])} | "
                f"{fmt(row['sign'])} | {fmt(row['mag_ratio'])} |"
            )

    if zero_norm is not None:
        center = float(zero_norm[args.joint])
        lines.append("")
        lines.append("## Per-Horizon Joint Metrics Around Raw Motor Zero")
        for name, (pred, gt) in loaded.items():
            lines.append("")
            lines.append(f"### {name}")
            lines.append("| horizon | slope | corr | sign | mag_ratio | pred_abs | gt_abs |")
            lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for horizon in range(pred.shape[2]):
                row = metrics(pred, gt, horizon=horizon, joint=args.joint, eps=args.sign_eps, center=center)
                lines.append(
                    f"| {horizon + 1} | {fmt(row['slope'])} | {fmt(row['corr'])} | "
                    f"{fmt(row['sign'])} | {fmt(row['mag_ratio'])} | "
                    f"{fmt(row['pred_abs'], 4)} | {fmt(row['gt_abs'], 4)} |"
                )

    lines.append("")
    lines.append("## Prediction Agreement Between Runs")
    lines.append("")
    names = list(loaded.keys())
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            left_pred, left_gt = loaded[left_name]
            right_pred, _ = loaded[right_name]
            gt_values = left_gt[:, :, 0, args.joint].reshape(-1)
            active = np.abs(gt_values) > args.sign_eps
            left_values = left_pred[:, :, 0, args.joint].reshape(-1)[active]
            right_values = right_pred[:, :, 0, args.joint].reshape(-1)[active]
            slope, corr = fit_slope_corr(left_values, right_values)
            sign = float(np.mean(np.sign(left_values) == np.sign(right_values)))
            lines.append(
                f"- {left_name} vs {right_name}: pred_corr={fmt(corr)}, "
                f"pred_slope={fmt(slope)}, pred_sign_agree={fmt(sign)}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
