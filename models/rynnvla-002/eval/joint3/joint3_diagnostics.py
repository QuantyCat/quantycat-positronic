#!/usr/bin/env python3
"""Joint-3 focused diagnostics for high-motion model eval windows."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "libs"))
import episode_batch_eval as batch_eval
import episode_step_eval as step_eval


JOINT_INDEX = 3
JOINT_NAME = "joint_3"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="Broad focused eval JSON.")
    parser.add_argument("--selected-windows", required=True, help="Selected window JSON.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--positronic-config", default="models/rynnvla-002/config.yaml")
    parser.add_argument("--rynnvla-repo", default=os.environ.get("RYNNVLA_REPO", step_eval._default_rynnvla_repo()))
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--worst-k", type=int, default=10)
    parser.add_argument("--best-k", type=int, default=5)
    parser.add_argument("--sign-eps", type=float, default=1e-6)
    parser.add_argument("--deterministic-crop", action="store_true")
    parser.add_argument("--skip-model", action="store_true", help="Only run data/image audits; skip prediction plots.")
    return parser.parse_args()


def _norm_action(action: np.ndarray, min_values: np.ndarray, max_values: np.ndarray) -> np.ndarray:
    return np.clip(2.0 * (action - min_values) / (max_values - min_values + 1e-8) - 1.0, -1.0, 1.0)


def _load_action_chunk(action_dir: Path) -> np.ndarray:
    files = sorted(action_dir.glob("*.npy"), key=lambda p: int(p.stem))
    return np.asarray([np.load(f).astype(np.float32) for f in files], dtype=np.float32)


def _episode_action_dirs(episode_dir: Path) -> list[Path]:
    return step_eval._sorted_action_dirs(episode_dir / "abs_action")


def _window_chunks(episode_dir: Path, start: int, end: int) -> np.ndarray:
    action_dirs = _episode_action_dirs(episode_dir)
    return np.asarray([_load_action_chunk(action_dirs[step]) for step in range(start, end + 1)], dtype=np.float32)


def _all_task_chunks(task_dir: Path) -> np.ndarray:
    arrays = []
    for episode_dir in sorted(task_dir.glob("episode_*")):
        for action_dir in _episode_action_dirs(episode_dir):
            arrays.append(_load_action_chunk(action_dir))
    return np.asarray(arrays, dtype=np.float32)


def _sign(values: np.ndarray, eps: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.where(values > eps, 1, np.where(values < -eps, -1, 0)).astype(np.int8)


def _sign_boundary_stats(values: np.ndarray, min_values: np.ndarray, max_values: np.ndarray, eps: float) -> dict[str, Any]:
    raw = np.asarray(values, dtype=np.float32).reshape(-1)
    norm = _norm_action(raw[:, None], min_values[[JOINT_INDEX]], max_values[[JOINT_INDEX]]).reshape(-1)
    raw_sign = _sign(raw, eps)
    norm_sign = _sign(norm, eps)
    active = raw_sign != 0
    disagree = active & (raw_sign != norm_sign)
    raw_midpoint = float((min_values[JOINT_INDEX] + max_values[JOINT_INDEX]) / 2.0)
    return {
        "count": int(raw.size),
        "active_raw_count": int(active.sum()),
        "raw_min": float(raw.min()),
        "raw_max": float(raw.max()),
        "raw_mean": float(raw.mean()),
        "raw_median": float(np.median(raw)),
        "raw_midpoint_where_normalized_zero": raw_midpoint,
        "normalized_value_of_raw_zero": float(_norm_action(np.asarray([[0.0]], dtype=np.float32), min_values[[JOINT_INDEX]], max_values[[JOINT_INDEX]])[0, 0]),
        "raw_positive_fraction": float((raw_sign == 1).mean()),
        "raw_negative_fraction": float((raw_sign == -1).mean()),
        "normalized_positive_fraction": float((norm_sign == 1).mean()),
        "normalized_negative_fraction": float((norm_sign == -1).mean()),
        "raw_vs_normalized_sign_disagreement_fraction_all": float(disagree.mean()),
        "raw_vs_normalized_sign_disagreement_fraction_active_raw": float(disagree.sum() / max(1, active.sum())),
        "small_positive_raw_relabelled_negative_by_norm_fraction": float(((raw > eps) & (raw < raw_midpoint)).mean()),
    }


def _case_rows(report: dict[str, Any], selected: dict[str, Any]) -> list[dict[str, Any]]:
    selected_by_episode = {Path(row["episode"]).name: row for row in selected["windows"]}
    rows = []
    for case in report["cases"]:
        episode_name = Path(case["episode"]).name
        focus = case["focus_joints"][JOINT_NAME]
        selected_row = selected_by_episode.get(episode_name, {})
        rows.append(
            {
                "episode": case["episode"],
                "episode_name": episode_name,
                "start_step": int(case["start_step"]),
                "end_step": int(case["end_step"]),
                "joint_3_window_score": selected_row.get("joint_3_window_score"),
                "raw_sign": focus["raw_sign_agreement"],
                "normalized_sign": focus["normalized_sign_agreement"],
                "corr": focus["normalized_same_corr"],
                "slope": focus["normalized_fit_slope"],
                "normalized_mae": focus["normalized_mae"],
            }
        )
    return rows


def _slope_sort_key(row: dict[str, Any]) -> float:
    slope = row["slope"]
    return -999.0 if slope is None else float(slope)


def _select_diagnostic_cases(rows: list[dict[str, Any]], worst_k: int, best_k: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    worst = sorted(rows, key=_slope_sort_key)[:worst_k]
    best = sorted(rows, key=_slope_sort_key, reverse=True)[:best_k]
    return worst, best


def _window_semantics(row: dict[str, Any], min_values: np.ndarray, max_values: np.ndarray, eps: float) -> dict[str, Any]:
    chunks = _window_chunks(Path(row["episode"]), row["start_step"], row["end_step"])
    j3 = chunks[:, :, JOINT_INDEX]
    j3_norm = _norm_action(j3[..., None], min_values[[JOINT_INDEX]], max_values[[JOINT_INDEX]])[..., 0]
    raw_sign = _sign(j3, eps)
    norm_sign = _sign(j3_norm, eps)
    raw_pos = float((raw_sign == 1).mean())
    raw_neg = float((raw_sign == -1).mean())
    norm_pos = float((norm_sign == 1).mean())
    norm_neg = float((norm_sign == -1).mean())

    def label(pos: float, neg: float) -> str:
        if pos >= 0.8:
            return "mostly_positive"
        if neg >= 0.8:
            return "mostly_negative"
        return "mixed"

    state_dir = Path(row["episode"]) / "state"
    states = []
    for step in range(row["start_step"], row["end_step"] + 1):
        states.append(np.load(state_dir / f"state_{step}.npy").astype(np.float32))
    states_arr = np.asarray(states, dtype=np.float32)
    return {
        "episode_name": row["episode_name"],
        "start_step": row["start_step"],
        "end_step": row["end_step"],
        "raw_gt_mean": float(j3.mean()),
        "raw_gt_abs_mean": float(np.abs(j3).mean()),
        "raw_gt_min": float(j3.min()),
        "raw_gt_max": float(j3.max()),
        "raw_positive_fraction": raw_pos,
        "raw_negative_fraction": raw_neg,
        "raw_sign_category": label(raw_pos, raw_neg),
        "normalized_positive_fraction": norm_pos,
        "normalized_negative_fraction": norm_neg,
        "normalized_sign_category": label(norm_pos, norm_neg),
        "raw_norm_sign_disagreement_fraction": float((_sign(j3, eps) != _sign(j3_norm, eps)).mean()),
        "state_j3_start": float(states_arr[0, JOINT_INDEX]),
        "state_j3_end": float(states_arr[-1, JOINT_INDEX]),
        "state_j3_delta": float(states_arr[-1, JOINT_INDEX] - states_arr[0, JOINT_INDEX]),
    }


def _load_frame(episode_dir: Path, camera: str, step: int, size: tuple[int, int]) -> Image.Image:
    return Image.open(episode_dir / camera / f"image_{step}.png").convert("RGB").resize(size)


def _make_contact_sheet(row: dict[str, Any], out_dir: Path) -> str:
    episode_dir = Path(row["episode"])
    start = row["start_step"]
    end = row["end_step"]
    steps = sorted(set([start, start + (end - start) // 4, start + (end - start) // 2, start + 3 * (end - start) // 4, end]))
    thumb = (192, 144)
    pad = 8
    label_h = 24
    width = pad + len(steps) * (thumb[0] + pad)
    height = pad + 2 * (thumb[1] + label_h + pad)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for col, step in enumerate(steps):
        x = pad + col * (thumb[0] + pad)
        for row_idx, camera in enumerate(("front_image", "wrist_image")):
            y = pad + row_idx * (thumb[1] + label_h + pad)
            img = _load_frame(episode_dir, camera, step, thumb)
            sheet.paste(img, (x, y + label_h))
            draw.text((x, y), f"{camera.replace('_image', '')} step {step}", fill=(0, 0, 0))
    out_path = out_dir / f"{row['episode_name']}_{start:06d}_{end:06d}_contact_sheet.jpg"
    sheet.save(out_path, quality=92)
    return str(out_path)


def _plot_state_and_gt(row: dict[str, Any], min_values: np.ndarray, max_values: np.ndarray, out_dir: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episode_dir = Path(row["episode"])
    start = row["start_step"]
    end = row["end_step"]
    steps = np.arange(start, end + 1)
    chunks = _window_chunks(episode_dir, start, end)
    j3 = chunks[:, :, JOINT_INDEX]
    j3_norm = _norm_action(j3[..., None], min_values[[JOINT_INDEX]], max_values[[JOINT_INDEX]])[..., 0]
    states = np.asarray([np.load(episode_dir / "state" / f"state_{step}.npy").astype(np.float32) for step in steps], dtype=np.float32)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(steps, j3[:, 0], label="gt raw chunk[0]", color="#1f77b4")
    axes[0].plot(steps, j3.mean(axis=1), label="gt raw chunk mean", color="#ff7f0e")
    axes[0].axhline(0.0, color="#444444", linewidth=0.8)
    axes[0].axhline(float((min_values[JOINT_INDEX] + max_values[JOINT_INDEX]) / 2.0), color="#d62728", linestyle="--", linewidth=0.9, label="raw value where norm=0")
    axes[0].set_ylabel("raw joint_3")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(steps, j3_norm[:, 0], label="gt norm chunk[0]", color="#1f77b4")
    axes[1].plot(steps, j3_norm.mean(axis=1), label="gt norm chunk mean", color="#ff7f0e")
    axes[1].axhline(0.0, color="#d62728", linestyle="--", linewidth=0.9)
    axes[1].set_ylabel("normalized joint_3")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(steps, states[:, JOINT_INDEX], color="#2ca02c", label="state joint_3")
    axes[2].set_ylabel("state joint_3")
    axes[2].set_xlabel("episode step")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.25)

    fig.suptitle(f"{row['episode_name']} steps {start}-{end}: joint_3 target/state")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = out_dir / f"{row['episode_name']}_{start:06d}_{end:06d}_state_gt.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def _run_model_window(row: dict[str, Any], solver: Any, cfg: dict[str, Any], min_values: np.ndarray, max_values: np.ndarray, out_dir: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episode_dir = Path(row["episode"])
    prompt = step_eval._extract_task_from_episode(episode_dir)
    steps = list(range(row["start_step"], row["end_step"] + 1))
    pred_steps = []
    gt_steps = []
    his = int(cfg["his"])
    for step in steps:
        sample = step_eval._build_sample(episode_dir, step=step, his=his)
        step_eval._reset_solver_history(solver, sample)
        pred = solver.get_action_wrist_action_head_state(
            front_image=sample["front_current"],
            wrist_image=sample["wrist_current"],
            state=sample["state"],
            prompt=prompt,
        )
        pred_steps.append(np.asarray(pred, dtype=np.float32))
        gt_steps.append(np.asarray(sample["gt_action"], dtype=np.float32))
    pred_arr = np.asarray(pred_steps, dtype=np.float32)
    gt_arr = np.asarray(gt_steps, dtype=np.float32)
    pred_norm = _norm_action(pred_arr, min_values, max_values)
    gt_norm = _norm_action(gt_arr, min_values, max_values)
    stem = f"{row['episode_name']}_{row['start_step']:06d}_{row['end_step']:06d}"
    npz_path = out_dir / f"{stem}_pred_gt_arrays.npz"
    np.savez_compressed(npz_path, steps=np.asarray(steps), pred=pred_arr, gt=gt_arr, pred_norm=pred_norm, gt_norm=gt_norm)

    x = np.asarray(steps)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    axes[0].plot(x, gt_arr[:, 0, JOINT_INDEX], label="gt raw chunk[0]", color="#1f77b4")
    axes[0].plot(x, pred_arr[:, 0, JOINT_INDEX], label="pred raw chunk[0]", color="#d62728")
    axes[0].plot(x, gt_arr[:, :, JOINT_INDEX].mean(axis=1), label="gt raw chunk mean", color="#1f77b4", alpha=0.45, linestyle="--")
    axes[0].plot(x, pred_arr[:, :, JOINT_INDEX].mean(axis=1), label="pred raw chunk mean", color="#d62728", alpha=0.45, linestyle="--")
    axes[0].axhline(0.0, color="#444444", linewidth=0.8)
    axes[0].axhline(float((min_values[JOINT_INDEX] + max_values[JOINT_INDEX]) / 2.0), color="#9467bd", linestyle=":", linewidth=1.0, label="raw norm-zero")
    axes[0].set_ylabel("raw joint_3")
    axes[0].legend(loc="best", ncol=2)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(x, gt_norm[:, 0, JOINT_INDEX], label="gt norm chunk[0]", color="#1f77b4")
    axes[1].plot(x, pred_norm[:, 0, JOINT_INDEX], label="pred norm chunk[0]", color="#d62728")
    axes[1].plot(x, gt_norm[:, :, JOINT_INDEX].mean(axis=1), label="gt norm chunk mean", color="#1f77b4", alpha=0.45, linestyle="--")
    axes[1].plot(x, pred_norm[:, :, JOINT_INDEX].mean(axis=1), label="pred norm chunk mean", color="#d62728", alpha=0.45, linestyle="--")
    axes[1].axhline(0.0, color="#9467bd", linestyle=":", linewidth=1.0)
    axes[1].set_ylabel("normalized joint_3")
    axes[1].legend(loc="best", ncol=2)
    axes[1].grid(True, alpha=0.25)

    gt_flat = gt_norm[:, :, JOINT_INDEX].reshape(-1)
    pred_flat = pred_norm[:, :, JOINT_INDEX].reshape(-1)
    axes[2].scatter(gt_flat, pred_flat, s=14, alpha=0.45, color="#2ca02c")
    low = float(min(gt_flat.min(), pred_flat.min()))
    high = float(max(gt_flat.max(), pred_flat.max()))
    axes[2].plot([low, high], [low, high], color="#444444", linestyle="--", linewidth=0.9)
    axes[2].axhline(0.0, color="#999999", linewidth=0.7)
    axes[2].axvline(0.0, color="#999999", linewidth=0.7)
    axes[2].set_xlabel("gt normalized joint_3")
    axes[2].set_ylabel("pred normalized joint_3")
    axes[2].grid(True, alpha=0.25)
    fit = batch_eval._linear_fit(gt_flat, pred_flat)
    corr = batch_eval._corrcoef_1d(pred_flat, gt_flat)
    axes[2].set_title(f"all chunks corr={corr:.3f} slope={fit['slope'] if fit['slope'] is not None else float('nan'):.3f}")

    fig.suptitle(f"{row['episode_name']} steps {row['start_step']}-{row['end_step']}: joint_3 pred vs gt")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plot_path = out_dir / f"{stem}_pred_gt_joint3.png"
    fig.savefig(plot_path, dpi=170)
    plt.close(fig)
    return {"arrays": str(npz_path), "plot": str(plot_path)}


def main() -> None:
    args = _parse_args()
    root = step_eval._repo_root()
    os.chdir(root)
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    images_dir = out_dir / "image_state_inspection"
    arrays_dir = out_dir / "arrays"
    for d in (plots_dir, images_dir, arrays_dir):
        d.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(args.positronic_config)
    if not cfg_path.is_absolute():
        cfg_path = (root / cfg_path).resolve()
    cfg = step_eval._load_positronic_config(cfg_path)
    batch_eval._configure_env(root, cfg)
    step_eval._ensure_rynnvla_on_path(args.rynnvla_repo.strip() or None)
    min_values, max_values = batch_eval._action_bounds()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    selected = json.loads(Path(args.selected_windows).read_text(encoding="utf-8"))
    rows = _case_rows(report, selected)
    worst, best = _select_diagnostic_cases(rows, args.worst_k, args.best_k)
    diagnostic_rows = worst + [row for row in best if row["episode_name"] not in {w["episode_name"] for w in worst}]

    task_dir = Path(rows[0]["episode"]).parent
    train_chunks = _all_task_chunks(task_dir)
    selected_chunks = np.concatenate(
        [_window_chunks(Path(row["episode"]), row["start_step"], row["end_step"]) for row in rows],
        axis=0,
    )
    worst_chunks = np.concatenate(
        [_window_chunks(Path(row["episode"]), row["start_step"], row["end_step"]) for row in worst],
        axis=0,
    )
    best_chunks = np.concatenate(
        [_window_chunks(Path(row["episode"]), row["start_step"], row["end_step"]) for row in best],
        axis=0,
    )

    boundary_audit = {
        "joint": JOINT_NAME,
        "raw_min": float(min_values[JOINT_INDEX]),
        "raw_max": float(max_values[JOINT_INDEX]),
        "raw_midpoint_where_normalized_zero": float((min_values[JOINT_INDEX] + max_values[JOINT_INDEX]) / 2.0),
        "normalized_value_of_raw_zero": float(_norm_action(np.asarray([[0.0]], dtype=np.float32), min_values[[JOINT_INDEX]], max_values[[JOINT_INDEX]])[0, 0]),
        "train_all": _sign_boundary_stats(train_chunks[:, :, JOINT_INDEX], min_values, max_values, args.sign_eps),
        "broad_top50": _sign_boundary_stats(selected_chunks[:, :, JOINT_INDEX], min_values, max_values, args.sign_eps),
        "worst_by_slope": _sign_boundary_stats(worst_chunks[:, :, JOINT_INDEX], min_values, max_values, args.sign_eps),
        "best_by_slope": _sign_boundary_stats(best_chunks[:, :, JOINT_INDEX], min_values, max_values, args.sign_eps),
    }

    semantics = [_window_semantics(row, min_values, max_values, args.sign_eps) for row in rows]
    semantic_counts: dict[str, int] = {}
    for item in semantics:
        key = f"raw_{item['raw_sign_category']}__norm_{item['normalized_sign_category']}"
        semantic_counts[key] = semantic_counts.get(key, 0) + 1

    inspection_paths = []
    for row in diagnostic_rows:
        inspection_paths.append(
            {
                "episode_name": row["episode_name"],
                "start_step": row["start_step"],
                "end_step": row["end_step"],
                "contact_sheet": _make_contact_sheet(row, images_dir),
                "state_gt_plot": _plot_state_and_gt(row, min_values, max_values, images_dir),
            }
        )

    model_outputs = []
    if not args.skip_model:
        ckpt_path = Path(args.checkpoint).expanduser().resolve()
        solver_args = batch_eval._make_solver_args(args, cfg, ckpt_path, out_dir)
        solver = batch_eval._load_solver(args.rynnvla_repo, solver_args)
        for index, row in enumerate(diagnostic_rows, start=1):
            print(f"[{index}/{len(diagnostic_rows)}] model plots for {row['episode_name']} {row['start_step']}-{row['end_step']}", flush=True)
            output = _run_model_window(row, solver, cfg, min_values, max_values, arrays_dir)
            model_outputs.append({"episode_name": row["episode_name"], "start_step": row["start_step"], "end_step": row["end_step"], **output})

    summary = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "source_report": str(Path(args.report).expanduser().resolve()),
        "selected_windows": str(Path(args.selected_windows).expanduser().resolve()),
        "worst_by_slope": worst,
        "best_by_slope": best,
        "sign_boundary_audit": boundary_audit,
        "semantic_mix_counts": semantic_counts,
        "window_semantics": semantics,
        "image_state_inspection": inspection_paths,
        "model_pred_gt_outputs": model_outputs,
    }
    out_path = out_dir / "joint3_diagnostics_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved {out_path}")
    print(json.dumps({"semantic_mix_counts": semantic_counts, "sign_boundary_audit": boundary_audit}, indent=2))


if __name__ == "__main__":
    main()
