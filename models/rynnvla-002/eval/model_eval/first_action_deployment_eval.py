#!/usr/bin/env python3
"""Evaluate deployment-mode first-action-only behavior from saved trace arrays."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


JOINT_NAMES = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "gripper"]
DELTA_JOINTS = range(5)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def _fit_slope(pred: np.ndarray, gt: np.ndarray) -> tuple[float | None, float | None]:
    x = gt.reshape(-1).astype(np.float64)
    y = pred.reshape(-1).astype(np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2 or float(np.var(x)) < 1e-12:
        return None, None
    slope, _ = np.polyfit(x, y, 1)
    corr = float(np.corrcoef(x, y)[0, 1]) if np.std(y) > 1e-12 else None
    return float(slope), corr


def _action_metrics(pred: np.ndarray, gt: np.ndarray, joint: int) -> dict[str, Any]:
    p = pred[..., joint]
    g = gt[..., joint]
    slope, corr = _fit_slope(p, g)
    valid_sign = np.abs(g.reshape(-1)) > 1e-6
    sign = None
    if np.any(valid_sign):
        sign = float(np.mean(np.sign(p.reshape(-1)[valid_sign]) == np.sign(g.reshape(-1)[valid_sign])))
    pred_abs = float(np.mean(np.abs(p)))
    gt_abs = float(np.mean(np.abs(g)))
    return {
        "joint": JOINT_NAMES[joint],
        "slope": slope,
        "corr": corr,
        "sign_agreement": sign,
        "mae": float(np.mean(np.abs(p - g))),
        "gt_mean_abs": gt_abs,
        "pred_mean_abs": pred_abs,
        "magnitude_ratio": pred_abs / (gt_abs + 1e-8),
        "pred_mean": float(np.mean(p)),
        "gt_mean": float(np.mean(g)),
        "pred_std": float(np.std(p)),
        "gt_std": float(np.std(g)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_aggregate(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    labels = [r["joint"] for r in rows]
    slopes = [r["slope"] if r["slope"] is not None else np.nan for r in rows]
    mags = [r["magnitude_ratio"] for r in rows]
    signs = [r["sign_agreement"] if r["sign_agreement"] is not None else np.nan for r in rows]

    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width, slopes, width, label="slope")
    ax.bar(x, mags, width, label="magnitude ratio")
    ax.bar(x + width, signs, width, label="sign agreement")
    ax.axhline(1.0, color="black", linewidth=1, alpha=0.35)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, max(1.2, float(np.nanmax([slopes, mags, signs])) + 0.1))
    ax.set_title("First-action-only aggregate metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "first_action_aggregate_metrics.png", dpi=150)
    plt.close(fig)


def _plot_case_trajectories(output_dir: Path, pred_h1: np.ndarray, gt_h1: np.ndarray, case_rows: list[dict[str, Any]], top_k: int) -> None:
    valid_rows = [r for r in case_rows if r["joint_3_slope"] != ""]
    ranked = sorted(valid_rows, key=lambda r: float(r["joint_3_slope"]))
    selected = ranked[:top_k] + ranked[-top_k:]
    fig, axes = plt.subplots(len(selected), 2, figsize=(12, max(2.2 * len(selected), 8)), sharex=False)
    if len(selected) == 1:
        axes = np.asarray([axes])
    for ax_row, row in zip(axes, selected):
        idx = int(row["case_index"])
        x = np.arange(pred_h1.shape[1])
        ax_row[0].plot(x, gt_h1[idx, :, 3], label="gt action0 j3", color="#111111")
        ax_row[0].plot(x, pred_h1[idx, :, 3], label="pred action0 j3", color="#d62728")
        ax_row[0].set_title(f"{row['case_id']}:{row['start_step']} s={float(row['joint_3_slope']):.3f}")
        ax_row[0].set_ylabel("raw delta")
        ax_row[1].plot(x, np.cumsum(gt_h1[idx, :, 3]), label="gt cumulative", color="#111111")
        ax_row[1].plot(x, np.cumsum(pred_h1[idx, :, 3]), label="pred cumulative", color="#d62728")
        ax_row[1].set_ylabel("cumulative delta")
    axes[0, 0].legend(loc="best", fontsize=8)
    axes[0, 1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "first_action_j3_worst_best_sequences.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    traces = np.load(Path(args.trace_npz).expanduser().resolve(), allow_pickle=False)
    pred_h1 = traces["pred"][:, :, 0, :]
    gt_h1 = traces["gt"][:, :, 0, :]
    case_ids = [str(x) for x in traces["case_id"]]
    starts = [int(x) for x in traces["case_start_step"]]

    aggregate_rows = [_action_metrics(pred_h1, gt_h1, joint) for joint in range(len(JOINT_NAMES))]
    _write_csv(output_dir / "first_action_aggregate_metrics.csv", aggregate_rows)
    _plot_aggregate(output_dir, aggregate_rows)

    case_rows: list[dict[str, Any]] = []
    for i, (case_id, start) in enumerate(zip(case_ids, starts)):
        row = {
            "case_index": i,
            "case_id": case_id,
            "start_step": start,
        }
        for joint in range(len(JOINT_NAMES)):
            metrics = _action_metrics(pred_h1[i : i + 1], gt_h1[i : i + 1], joint)
            row[f"{JOINT_NAMES[joint]}_slope"] = metrics["slope"] if metrics["slope"] is not None else ""
            row[f"{JOINT_NAMES[joint]}_corr"] = metrics["corr"] if metrics["corr"] is not None else ""
            row[f"{JOINT_NAMES[joint]}_mae"] = metrics["mae"]
            row[f"{JOINT_NAMES[joint]}_mag_ratio"] = metrics["magnitude_ratio"]
            row[f"{JOINT_NAMES[joint]}_sign"] = metrics["sign_agreement"] if metrics["sign_agreement"] is not None else ""
        case_rows.append(row)
    _write_csv(output_dir / "first_action_case_metrics.csv", case_rows)
    _plot_case_trajectories(output_dir, pred_h1, gt_h1, case_rows, args.top_k)

    trajectory_rows: list[dict[str, Any]] = []
    for joint in DELTA_JOINTS:
        pred_cum = np.cumsum(pred_h1[:, :, joint], axis=1)
        gt_cum = np.cumsum(gt_h1[:, :, joint], axis=1)
        final_pred = pred_cum[:, -1]
        final_gt = gt_cum[:, -1]
        slope, corr = _fit_slope(final_pred, final_gt)
        trajectory_rows.append(
            {
                "joint": JOINT_NAMES[joint],
                "cumulative_final_slope": slope,
                "cumulative_final_corr": corr,
                "cumulative_rmse": float(np.sqrt(np.mean((pred_cum - gt_cum) ** 2))),
                "final_mae": float(np.mean(np.abs(final_pred - final_gt))),
                "final_gt_mean_abs": float(np.mean(np.abs(final_gt))),
                "final_pred_mean_abs": float(np.mean(np.abs(final_pred))),
                "final_magnitude_ratio": float(np.mean(np.abs(final_pred)) / (np.mean(np.abs(final_gt)) + 1e-8)),
            }
        )
    _write_csv(output_dir / "first_action_cumulative_delta_metrics.csv", trajectory_rows)

    j3 = aggregate_rows[3]
    j3_traj = trajectory_rows[3]
    report = output_dir / "first_action_deployment_eval.md"
    report.write_text(
        "\n".join(
            [
                "# First-Action Deployment Eval",
                "",
                f"Trace source: `{Path(args.trace_npz).expanduser().resolve()}`",
                "",
                "This evaluates the deployment pattern where the policy predicts a 5-action chunk but only action 0 is executed before replanning.",
                "It is still an offline teacher-forced proxy: each prefix comes from the dataset, not from a real closed-loop environment rollout.",
                "",
                "## Aggregate First-Action Metrics",
                "",
                "| joint | slope | corr | sign | magnitude_ratio | mae |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
                *[
                    f"| {r['joint']} | {r['slope'] if r['slope'] is not None else float('nan'):.3f} | {r['corr'] if r['corr'] is not None else float('nan'):.3f} | {r['sign_agreement'] if r['sign_agreement'] is not None else float('nan'):.3f} | {r['magnitude_ratio']:.3f} | {r['mae']:.3f} |"
                    for r in aggregate_rows
                ],
                "",
                "## Joint 3 Readout",
                "",
                f"- immediate h1 slope: {j3['slope']:.3f}",
                f"- immediate h1 correlation: {j3['corr']:.3f}",
                f"- immediate h1 sign agreement: {j3['sign_agreement']:.3f}",
                f"- immediate h1 magnitude ratio: {j3['magnitude_ratio']:.3f}",
                f"- cumulative final displacement slope: {j3_traj['cumulative_final_slope']:.3f}",
                f"- cumulative final displacement magnitude ratio: {j3_traj['final_magnitude_ratio']:.3f}",
                f"- cumulative trajectory RMSE: {j3_traj['cumulative_rmse']:.3f}",
                "",
                "## Artifacts",
                "",
                "- `first_action_aggregate_metrics.csv`",
                "- `first_action_case_metrics.csv`",
                "- `first_action_cumulative_delta_metrics.csv`",
                "- `first_action_aggregate_metrics.png`",
                "- `first_action_j3_worst_best_sequences.png`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
