#!/usr/bin/env python3
"""Offline gain calibration for saved OpenPI high-motion trace arrays."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[3]
EVAL_DIR = REPO / "models/openpi/eval/libs"
NORM_STATS_ROOT = Path.home() / "quantycat-data/norm_stats/openpi"
DEFAULT_LABEL_DIR = (
    REPO
    / "eval_output/screwdriver_so101/model_eval/openpi_pi05_h20_lora_step9999"
)
DEFAULT_OUTPUT_DIR = (
    REPO
    / "eval_output/screwdriver_so101/model_eval/openpi_pi05_h20_lora_step9999_gain_calibration"
)


def _parse_gains(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--j2-gains", default="1.00,1.10,1.20,1.30,1.40,1.50,1.60")
    parser.add_argument("--j3-gains", default="1.00,1.10,1.20,1.30,1.40,1.50")
    parser.add_argument("--j4-gains", default="1.00,1.10,1.20,1.30,1.40")
    parser.add_argument("--sign-eps", type=float, default=0.01)
    parser.add_argument(
        "--action-bounds-json",
        type=Path,
        default=None,
        help="Optional JSON with {'min': [...], 'max': [...]} bounds for normalized metrics.",
    )
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=NORM_STATS_ROOT / "pi05_quantycat_lora_achieved_delta" / "norm_stats.json",
        help="Path to openpi norm_stats.json used for action bound normalization.",
    )
    parser.add_argument(
        "--conservative-score-tolerance",
        type=float,
        default=0.00025,
        help="Choose the smallest gain combo within this score delta of the best combo.",
    )
    return parser.parse_args()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _configure_eval_imports() -> None:
    sys.path.insert(0, str(EVAL_DIR))


def _joint_focus(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    focus_joints = ("joint_0", "joint_1", "joint_2", "joint_3", "joint_4")
    result: dict[str, dict[str, Any]] = {}
    for joint in focus_joints:
        convention = summary["convention_checks"][joint]
        norm_convention = summary["normalized_convention_checks"][joint]
        result[joint] = {
            "raw_sign_agreement": summary["per_joint"]["sign_agreement"][joint],
            "raw_same_corr": convention["same_joint_corr"],
            "raw_fit_slope": convention["fit_same_joint"]["slope"],
            "normalized_mae": summary["normalized_per_joint"]["mean_abs_error"][joint],
            "normalized_sign_agreement": summary["normalized_per_joint"]["sign_agreement"][joint],
            "normalized_same_corr": norm_convention["same_joint_corr"],
            "normalized_fit_slope": norm_convention["fit_same_joint"]["slope"],
        }
    return result


def _load_trace(label_dir: Path, joint: int) -> tuple[np.ndarray, np.ndarray]:
    path = label_dir / f"broad_j{joint}_high_motion_top50" / "focused_high_motion_traces.npz"
    if not path.is_file():
        matches = sorted(label_dir.glob(f"broad_j{joint}_high_motion_top*/focused_high_motion_traces.npz"))
        if matches:
            path = matches[0]
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.load(path)
    return np.asarray(data["pred"], dtype=np.float32), np.asarray(data["gt"], dtype=np.float32)


def _load_action_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(payload["min"], dtype=np.float32), np.asarray(payload["max"], dtype=np.float32)


def _score_trace(
    pred: np.ndarray,
    gt: np.ndarray,
    gains: dict[int, float],
    sign_eps: float,
    min_values: np.ndarray,
    max_values: np.ndarray,
) -> dict[str, dict[str, Any]]:
    import episode_batch_eval as batch_eval

    calibrated = pred.copy()
    for joint, gain in gains.items():
        calibrated[..., joint] *= gain
    summary = batch_eval._summarize(calibrated, gt, sign_eps, min_values, max_values)
    return _joint_focus(summary)


def _diag_row(focus: dict[str, dict[str, Any]], joint: int) -> dict[str, float | None]:
    row = focus[f"joint_{joint}"]
    return {
        "sign": row["raw_sign_agreement"],
        "corr": row["raw_same_corr"],
        "slope": row["raw_fit_slope"],
        "norm_mae": row["normalized_mae"],
        "norm_corr": row["normalized_same_corr"],
        "norm_slope": row["normalized_fit_slope"],
    }


def _mean(values: list[float | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return None
    return float(np.mean(cleaned))


def _format_value(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# OpenPI pi0.5 LoRA Gain Calibration",
        "",
        f"Created: {payload['created_at']}",
        "",
        "Calibration is applied to saved predicted deltas before metric scoring:",
        "",
        "```text",
        "calibrated_delta[j] = predicted_delta[j] * gain[j]",
        "```",
        "",
        "## Ranked Gain Grid",
        "",
        "| Rank | j2 gain | j3 gain | j4 gain | j2 slope | j3 slope | j4 slope | mean target MAE | score |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(payload["ranked"][:16], start=1):
        lines.append(
            "| "
            f"{idx} | {row['gains']['2']:.2f} | {row['gains']['3']:.2f} | {row['gains']['4']:.2f} | "
            f"{_format_value(row['diagonal']['2']['slope'])} | "
            f"{_format_value(row['diagonal']['3']['slope'])} | "
            f"{_format_value(row['diagonal']['4']['slope'])} | "
            f"{_format_value(row['mean_target_norm_mae'])} | "
            f"{row['score']:.4f} |"
        )

    selected = payload["selected"]
    conservative = payload["conservative_selected"]
    lines.extend(
        [
            "",
            "## Selected",
            "",
            (
                "Score-selected gains: "
                f"j2={selected['gains']['2']:.3f}, "
                f"j3={selected['gains']['3']:.3f}, "
                f"j4={selected['gains']['4']:.3f}"
            ),
            "",
            (
                "Conservative deployment gains: "
                f"j2={conservative['gains']['2']:.3f}, "
                f"j3={conservative['gains']['3']:.3f}, "
                f"j4={conservative['gains']['4']:.3f}"
            ),
            "",
            (
                "Reason: the conservative choice is the smallest gain combo within the configured "
                "near-best score tolerance; positive gains preserve raw sign and raw correlation."
            ),
            "",
            "## Baseline Vs Conservative",
            "",
            "| Joint | Baseline slope | Conservative slope | Baseline MAE | Conservative MAE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    baseline = payload["baseline"]["diagonal"]
    chosen = conservative["diagonal"]
    for joint in ("0", "1", "2", "3", "4"):
        lines.append(
            "| "
            f"j{joint} | {_format_value(baseline[joint]['slope'])} | "
            f"{_format_value(chosen[joint]['slope'])} | "
            f"{_format_value(baseline[joint]['norm_mae'])} | "
            f"{_format_value(chosen[joint]['norm_mae'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    label_dir = args.label_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _configure_eval_imports()
    import episode_batch_eval as batch_eval

    if args.action_bounds_json is not None:
        min_values, max_values = _load_action_bounds(args.action_bounds_json.expanduser().resolve())
    else:
        min_values, max_values = batch_eval._action_bounds(args.norm_stats.expanduser().resolve())
    traces = {joint: _load_trace(label_dir, joint) for joint in range(5)}
    j2_gains = _parse_gains(args.j2_gains)
    j3_gains = _parse_gains(args.j3_gains)
    j4_gains = _parse_gains(args.j4_gains)

    rows: list[dict[str, Any]] = []
    for j2_gain, j3_gain, j4_gain in product(j2_gains, j3_gains, j4_gains):
        gains = {2: j2_gain, 3: j3_gain, 4: j4_gain}
        coverage: dict[str, Any] = {}
        diagonal: dict[str, Any] = {}
        for ranked_joint, (pred, gt) in traces.items():
            focus = _score_trace(pred, gt, gains, args.sign_eps, min_values, max_values)
            diag = _diag_row(focus, ranked_joint)
            coverage[str(ranked_joint)] = {
                "diagonal_joint": f"joint_{ranked_joint}",
                "diagonal_metrics": diag,
                "all_focus_joints": focus,
            }
            diagonal[str(ranked_joint)] = diag

        target_mae = _mean(
            [
                diagonal["2"]["norm_mae"],
                diagonal["3"]["norm_mae"],
                diagonal["4"]["norm_mae"],
            ]
        )
        target_slope_error = _mean(
            [
                abs(float(diagonal["2"]["slope"]) - 1.0),
                abs(float(diagonal["3"]["slope"]) - 1.0),
                abs(float(diagonal["4"]["slope"]) - 1.0),
            ]
        )
        # Keep the scoring simple: prioritize normalized MAE, then closeness to
        # unit raw slope, then smaller gains if metrics tie closely.
        score = float(target_mae or 0.0) + 0.05 * float(target_slope_error or 0.0) + 0.002 * (
            (j2_gain - 1.0) + (j3_gain - 1.0) + (j4_gain - 1.0)
        )
        rows.append(
            {
                "gains": {"2": j2_gain, "3": j3_gain, "4": j4_gain},
                "score": score,
                "mean_target_norm_mae": target_mae,
                "mean_target_slope_abs_error": target_slope_error,
                "diagonal": diagonal,
                "coverage": coverage,
            }
        )

    ranked = sorted(rows, key=lambda row: row["score"])
    best_score = ranked[0]["score"]
    near_best = [
        row
        for row in ranked
        if row["score"] <= best_score + args.conservative_score_tolerance
        and (row["gains"]["2"] > 1.0 or row["gains"]["3"] > 1.0 or row["gains"]["4"] > 1.0)
    ]
    conservative_selected = min(
        near_best or [ranked[0]],
        key=lambda row: (
            (row["gains"]["2"] - 1.0) + (row["gains"]["3"] - 1.0) + (row["gains"]["4"] - 1.0),
            row["gains"]["2"],
            row["gains"]["3"],
            row["gains"]["4"],
        ),
    )
    baseline = next(row for row in rows if row["gains"] == {"2": 1.0, "3": 1.0, "4": 1.0})
    payload = {
        "created_at": _stamp(),
        "label_dir": str(label_dir),
        "output_dir": str(output_dir),
        "sign_eps": args.sign_eps,
        "conservative_score_tolerance": args.conservative_score_tolerance,
        "swept_gains": {"2": j2_gains, "3": j3_gains, "4": j4_gains},
        "baseline": baseline,
        "selected": ranked[0],
        "conservative_selected": conservative_selected,
        "ranked": ranked,
    }

    json_path = output_dir / "gain_calibration_summary.json"
    md_path = output_dir / "gain_calibration_summary.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(md_path, payload)

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    selected = ranked[0]
    print(
        "score_selected "
        f"j2={selected['gains']['2']:.2f} "
        f"j3={selected['gains']['3']:.2f} "
        f"j4={selected['gains']['4']:.2f} "
        f"score={selected['score']:.4f}"
    )
    print(
        "conservative_selected "
        f"j2={conservative_selected['gains']['2']:.3f} "
        f"j3={conservative_selected['gains']['3']:.3f} "
        f"j4={conservative_selected['gains']['4']:.3f} "
        f"score={conservative_selected['score']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
