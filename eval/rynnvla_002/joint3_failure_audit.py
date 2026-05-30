#!/usr/bin/env python3
"""Audit joint-3 high-motion focused eval failures."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


JOINT = "joint_3"
JOINT_INDEX = 3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focused-json", required=True)
    parser.add_argument("--trace-npz", default=None)
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
    slope, intercept = np.polyfit(x, y, 1)
    corr = float(np.corrcoef(x, y)[0, 1]) if np.std(y) > 1e-12 else None
    return float(slope), corr


def _bucket(slope: float | None) -> str:
    if slope is None:
        return "undefined"
    if slope < 0:
        return "negative"
    if slope < 0.2:
        return "weak"
    return "good"


def _episode_len(episode: Path) -> int:
    return len(list((episode / "abs_action").iterdir()))


def _phase(start: int, length: int) -> str:
    frac = start / max(length - 1, 1)
    if frac < 0.45:
        return "early"
    if frac < 0.75:
        return "middle"
    return "late"


def _image_path(episode: Path, step: int, camera: str = "front_image") -> Path:
    return episode / camera / f"image_{step}.png"


def _state_path(episode: Path, step: int) -> Path:
    return episode / "state" / f"state_{step}.npy"


def _load_prefix_feature(episode: Path, step: int) -> np.ndarray:
    state = np.load(_state_path(episode, step)).astype(np.float32)
    chunks = [state / (np.linalg.norm(state) + 1e-6)]
    for camera in ("front_image", "wrist_image"):
        img = Image.open(_image_path(episode, step, camera)).convert("L").resize((32, 32))
        arr = np.asarray(img, dtype=np.float32).reshape(-1) / 255.0
        arr = arr - float(arr.mean())
        chunks.append(arr / (float(np.linalg.norm(arr)) + 1e-6))
    return np.concatenate(chunks)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_contact_sheet(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    thumbs = []
    for row in rows:
        image = Image.open(_image_path(Path(row["episode"]), int(row["start_step"]))).convert("RGB")
        image.thumbnail((220, 150))
        canvas = Image.new("RGB", (240, 190), "white")
        canvas.paste(image, ((240 - image.width) // 2, 8))
        draw = ImageDraw.Draw(canvas)
        label = f"{Path(row['episode']).name}:{row['start_step']} {row['bucket']}"
        metrics = f"s={row['j3_slope']:.3f} sign={row['j3_sign']:.3f}"
        draw.text((8, 158), label, fill="black")
        draw.text((8, 174), metrics, fill="black")
        thumbs.append(canvas)
    if not thumbs:
        return
    cols = min(4, len(thumbs))
    rows_n = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 240, rows_n * 190 + 28), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 6), title, fill="black")
    for i, thumb in enumerate(thumbs):
        x = (i % cols) * 240
        y = 28 + (i // cols) * 190
        sheet.paste(thumb, (x, y))
    sheet.save(path)


def _plot_hist(path: Path, rows: list[dict[str, Any]]) -> None:
    slopes = [r["j3_slope"] for r in rows if r["j3_slope"] is not None]
    plt.figure(figsize=(8, 4))
    plt.hist(slopes, bins=18, color="#4c78a8", edgecolor="white")
    plt.axvline(0, color="black", linewidth=1)
    plt.axvline(0.2, color="#e45756", linewidth=1, linestyle="--")
    plt.title("Joint 3 raw fit slope distribution")
    plt.xlabel("raw fit slope")
    plt.ylabel("case count")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _trace_plots(path: Path, rows: list[dict[str, Any]], traces: dict[str, np.ndarray], top_k: int) -> None:
    pred = traces["pred"]
    gt = traces["gt"]
    case_ids = [str(x) for x in traces["case_id"]]
    starts = [int(x) for x in traces["case_start_step"]]
    index = {(case_ids[i], starts[i]): i for i in range(len(case_ids))}
    ranked = sorted([r for r in rows if r["j3_slope"] is not None], key=lambda r: r["j3_slope"])
    selected = ranked[:top_k] + ranked[-top_k:]
    fig, axes = plt.subplots(len(selected), 2, figsize=(12, max(2.2 * len(selected), 8)), sharex=False)
    if len(selected) == 1:
        axes = np.asarray([axes])
    for ax_row, row in zip(axes, selected):
        idx = index[(Path(row["episode"]).name, int(row["start_step"]))]
        x = np.arange(pred.shape[1])
        for horizon in range(pred.shape[2]):
            ax_row[0].plot(x, gt[idx, :, horizon, JOINT_INDEX], alpha=0.35, color="#111111")
            ax_row[0].plot(x, pred[idx, :, horizon, JOINT_INDEX], alpha=0.35, color="#d62728")
        ax_row[0].set_title(f"{Path(row['episode']).name}:{row['start_step']} {row['bucket']} s={row['j3_slope']:.3f}")
        ax_row[0].set_ylabel("raw j3")
        ax_row[1].plot(x, gt[idx, :, 0, JOINT_INDEX], label="gt h1", color="#111111")
        ax_row[1].plot(x, pred[idx, :, 0, JOINT_INDEX], label="pred h1", color="#d62728")
        ax_row[1].plot(x, gt[idx, :, -1, JOINT_INDEX], label="gt h5", color="#777777", linestyle="--")
        ax_row[1].plot(x, pred[idx, :, -1, JOINT_INDEX], label="pred h5", color="#ff9896", linestyle="--")
    axes[0, 1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _per_horizon(output_dir: Path, traces: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    pred = traces["pred"]
    gt = traces["gt"]
    rows = []
    for h in range(pred.shape[2]):
        p = pred[:, :, h, JOINT_INDEX]
        g = gt[:, :, h, JOINT_INDEX]
        slope, corr = _fit_slope(p, g)
        sign = float(np.mean(np.sign(p.reshape(-1)) == np.sign(g.reshape(-1))))
        rows.append(
            {
                "horizon": h + 1,
                "j3_slope": slope,
                "j3_corr": corr,
                "j3_sign_agreement": sign,
                "j3_mae": float(np.mean(np.abs(p - g))),
                "gt_mean_abs": float(np.mean(np.abs(g))),
                "pred_mean_abs": float(np.mean(np.abs(p))),
                "magnitude_ratio": float(np.mean(np.abs(p)) / (np.mean(np.abs(g)) + 1e-8)),
            }
        )
    _write_csv(output_dir / "joint3_per_horizon_metrics.csv", rows)
    plt.figure(figsize=(8, 4))
    plt.plot([r["horizon"] for r in rows], [r["j3_slope"] for r in rows], marker="o", label="slope")
    plt.plot([r["horizon"] for r in rows], [r["magnitude_ratio"] for r in rows], marker="o", label="magnitude ratio")
    plt.axhline(1.0, color="black", linewidth=1, alpha=0.4)
    plt.xlabel("action horizon")
    plt.title("Joint 3 per-horizon behavior")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "joint3_per_horizon_metrics.png", dpi=150)
    plt.close()
    return rows


def main() -> None:
    args = _parse_args()
    focused_json = Path(args.focused_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(focused_json.read_text())
    rows = []
    for case in data["cases"]:
        episode = Path(case["episode"])
        start = int(case["start_step"])
        length = _episode_len(episode)
        focus = case["focus_joints"][JOINT]
        summary = case["summary"]
        slope = focus["raw_fit_slope"]
        gt_abs = summary["raw_distribution"]["gt"]["mean_abs"] if "mean_abs" in summary["raw_distribution"]["gt"] else None
        rows.append(
            {
                "episode": str(episode),
                "start_step": start,
                "episode_len": length,
                "phase": _phase(start, length),
                "start_frac": start / max(length - 1, 1),
                "bucket": _bucket(slope),
                "j3_slope": slope,
                "j3_corr": focus["raw_same_corr"],
                "j3_sign": focus["raw_sign_agreement"],
                "j3_mae_norm": focus["normalized_mae"],
                "j3_gt_mean_raw": summary["raw_distribution"]["gt"]["mean"][JOINT],
                "j3_pred_mean_raw": summary["raw_distribution"]["pred"]["mean"][JOINT],
                "j3_gt_std_raw": summary["raw_distribution"]["gt"]["std"][JOINT],
                "j3_pred_std_raw": summary["raw_distribution"]["pred"]["std"][JOINT],
                "j3_mag_ratio_raw": summary["per_joint"]["magnitude_ratio"][JOINT],
            }
        )
    rows.sort(key=lambda r: (float("inf") if r["j3_slope"] is None else r["j3_slope"]))
    _write_csv(output_dir / "joint3_case_table.csv", rows)
    _plot_hist(output_dir / "joint3_slope_hist.png", rows)

    phase_rows = []
    for phase in ("early", "middle", "late"):
        for bucket in ("negative", "weak", "good", "undefined"):
            subset = [r for r in rows if r["phase"] == phase and r["bucket"] == bucket]
            phase_rows.append({"phase": phase, "bucket": bucket, "count": len(subset)})
    _write_csv(output_dir / "joint3_phase_buckets.csv", phase_rows)

    worst = rows[: args.top_k]
    best = sorted([r for r in rows if r["j3_slope"] is not None], key=lambda r: r["j3_slope"], reverse=True)[: args.top_k]
    _make_contact_sheet(output_dir / "joint3_worst_contact_sheet.png", worst, "Worst joint-3 slope cases")
    _make_contact_sheet(output_dir / "joint3_best_contact_sheet.png", best, "Best joint-3 slope cases")

    features = []
    for row in rows:
        feat = _load_prefix_feature(Path(row["episode"]), int(row["start_step"]))
        features.append(feat)
    feature_arr = np.asarray(features, dtype=np.float32)
    feature_arr /= np.linalg.norm(feature_arr, axis=1, keepdims=True) + 1e-6
    pair_rows = []
    for i, row in enumerate(rows):
        sims = feature_arr @ feature_arr[i]
        order = np.argsort(-sims)
        for j in order[1:6]:
            other = rows[int(j)]
            opposite = np.sign(row["j3_gt_mean_raw"]) != np.sign(other["j3_gt_mean_raw"])
            if opposite or row["bucket"] != other["bucket"]:
                pair_rows.append(
                    {
                        "case_a": f"{Path(row['episode']).name}:{row['start_step']}",
                        "case_b": f"{Path(other['episode']).name}:{other['start_step']}",
                        "prefix_cosine": float(sims[j]),
                        "bucket_a": row["bucket"],
                        "bucket_b": other["bucket"],
                        "gt_mean_j3_a": row["j3_gt_mean_raw"],
                        "gt_mean_j3_b": other["j3_gt_mean_raw"],
                        "slope_a": row["j3_slope"],
                        "slope_b": other["j3_slope"],
                    }
                )
    pair_rows.sort(key=lambda r: -r["prefix_cosine"])
    _write_csv(output_dir / "prefix_ambiguity_pairs.csv", pair_rows[:80])

    horizon_rows: list[dict[str, Any]] = []
    trace_npz = Path(args.trace_npz).expanduser().resolve() if args.trace_npz else None
    if trace_npz and trace_npz.exists():
        traces = dict(np.load(trace_npz, allow_pickle=False))
        _trace_plots(output_dir / "joint3_worst_best_traces.png", rows, traces, args.top_k)
        horizon_rows = _per_horizon(output_dir, traces)

    report = output_dir / "joint3_epoch15_failure_audit.md"
    counts = {bucket: sum(1 for r in rows if r["bucket"] == bucket) for bucket in ("negative", "weak", "good", "undefined")}
    report.write_text(
        "\n".join(
            [
                "# Joint 3 Epoch 15 Failure Audit",
                "",
                f"Source: `{focused_json}`",
                "",
                "## Buckets",
                "",
                f"- negative slope: {counts['negative']}",
                f"- weak positive slope 0-0.2: {counts['weak']}",
                f"- good slope >=0.2: {counts['good']}",
                f"- undefined: {counts['undefined']}",
                "",
                "## Main Artifacts",
                "",
                "- `joint3_case_table.csv`: all cases sorted by joint-3 slope.",
                "- `joint3_phase_buckets.csv`: early/middle/late counts by bucket.",
                "- `prefix_ambiguity_pairs.csv`: nearest prefix pairs with conflicting bucket or GT direction.",
                "- `joint3_worst_contact_sheet.png` and `joint3_best_contact_sheet.png`: start-frame visual check.",
                "- `joint3_slope_hist.png`: slope distribution.",
                "- `joint3_worst_best_traces.png`: GT/pred traces for worst and best cases, if trace arrays were provided.",
                "- `joint3_per_horizon_metrics.csv` and `.png`: per-horizon behavior, if trace arrays were provided.",
                "",
                "## Per-Horizon Summary",
                "",
                *(
                    [
                        "| horizon | slope | corr | sign | magnitude_ratio | mae |",
                        "| --- | ---: | ---: | ---: | ---: | ---: |",
                    ]
                    + [
                        f"| {r['horizon']} | {r['j3_slope']:.3f} | {r['j3_corr']:.3f} | {r['j3_sign_agreement']:.3f} | {r['magnitude_ratio']:.3f} | {r['j3_mae']:.3f} |"
                        for r in horizon_rows
                    ]
                    if horizon_rows
                    else ["Trace arrays were not provided yet, so per-horizon metrics were skipped."]
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote audit to {report}")


if __name__ == "__main__":
    main()
