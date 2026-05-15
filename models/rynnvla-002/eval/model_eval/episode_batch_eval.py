#!/usr/bin/env python3
"""
Batch-evaluate many timesteps from one episode against saved ground-truth action chunks.

This is a higher-signal follow-up to episode_step_eval.py when a single step is not
enough to diagnose train/eval mismatch. It evaluates multiple timesteps from one real
training episode and summarizes:
  - overall MAE / max error / L2
  - per-joint signed bias
  - per-joint sign agreement
  - per-joint magnitude ratio
  - the same metrics after action normalization to [-1, 1]

Example:

  conda run -n rynnvla002 python models/rynnvla-002/inference/eval/episode_batch_eval.py \
    --episode my_data/training_pipeline/training_data/Put_the_screwdriver_into_the_cup/episode_000025 \
    --checkpoint /home/caroline/old_checkpoints/04152025_epoch4 \
    --rynnvla-repo /home/caroline/RynnVLA-002/rynnvla-002 \
    --start-step 100 \
    --max-steps 50 \
    --save-json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

import episode_step_eval as step_eval

_DEFAULT_CONFIG = "models/rynnvla-002/config.yaml"
_JOINT_LABELS = ("joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "gripper")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate many episode timesteps against saved ground-truth actions.")
    parser.add_argument("--episode", type=str, required=True, help="Path to one episode directory.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--positronic-config", type=str, default=_DEFAULT_CONFIG)
    parser.add_argument("--prompt", type=str, default=None, help="Override task prompt; default uses episode task name.")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--deterministic-crop", action="store_true")
    parser.add_argument(
        "--rynnvla-repo",
        type=str,
        default=os.environ.get("RYNNVLA_REPO", ""),
        help="If set, prepend this directory to sys.path (e.g. ~/RynnVLA-002/rynnvla-002)",
    )
    parser.add_argument("--start-step", type=int, default=0, help="First timestep index to evaluate.")
    parser.add_argument("--max-steps", type=int, default=50, help="Maximum number of timesteps to evaluate.")
    parser.add_argument("--stride", type=int, default=1, help="Step stride between evaluated timesteps.")
    parser.add_argument(
        "--sign-eps",
        type=float,
        default=1e-6,
        help="Treat absolute values <= this threshold as zero for sign-agreement reporting.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory for JSON/PNG/log outputs. Default: "
            "<training_output>/<task_label>_<robot>/model_eval_reports/<checkpoint_name>/"
        ),
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Write a JSON report under the resolved output directory.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Write PNG plots under the resolved output directory.",
    )
    return parser.parse_args()


def _configure_env(root: Path, cfg: dict[str, Any]) -> Path:
    work_dir = Path(cfg["work_dir"])
    if not work_dir.is_absolute():
        work_dir = (root / work_dir).resolve()
    os.environ["RYNNVLA_ACTION_STATS_FILE"] = str((work_dir / "min_max_action.txt").resolve())
    os.environ["RYNNVLA_STATE_STATS_FILE"] = str((work_dir / "min_max_state.txt").resolve())
    if cfg.get("action_norm_joint_scales"):
        os.environ["RYNNVLA_ACTION_NORM_SCALES"] = str(cfg["action_norm_joint_scales"])
    return work_dir


def _resolve_output_dir(root: Path, args: argparse.Namespace, cfg: dict[str, Any], ckpt_path: Path) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()

    training_output = Path(cfg.get("training_output", "training_output"))
    if not training_output.is_absolute():
        training_output = (root / training_output).resolve()
    task_dir = f"{cfg['task_label']}_{cfg['robot']}"
    episode_name = Path(args.episode).expanduser().resolve().name
    return training_output / task_dir / "model_eval" / ckpt_path.name / episode_name


def _make_solver_args(args: argparse.Namespace, cfg: dict[str, Any], ckpt_path: Path, output_dir: Path) -> argparse.Namespace:
    import argparse as _argparse

    return _argparse.Namespace(
        resume_path=str(ckpt_path),
        output_dir=str(output_dir.resolve()),
        device=args.gpu if args.gpu is not None else cfg["gpu"],
        action_dim=cfg["action_dim"],
        time_horizon=cfg["chunk_size"],
        max_seq_len=cfg["max_seq_len"],
        mask_image_logits=cfg["mask_image_logits"],
        dropout=cfg["dropout"],
        z_loss_weight=cfg["inference_z_loss_weight"],
        his=cfg["his_mode"],
        action_steps=cfg["action_steps"],
        deterministic_crop=bool(args.deterministic_crop or cfg.get("deterministic_crop", False)),
    )


def _load_solver(extra_repo: str | None, solver_args: argparse.Namespace):
    step_eval._ensure_rynnvla_on_path(extra_repo.strip() or None if extra_repo else None)
    try:
        from eval_solver_lerobot_action_head_state import Solver
    except ImportError as e:
        raise SystemExit(
            "Import failed: eval_solver_lerobot_action_head_state.Solver\n"
            "Add the QuantyCat RynnVLA-002 python root to PYTHONPATH, e.g.:\n"
            "  export PYTHONPATH=\"$HOME/RynnVLA-002/rynnvla-002:$PYTHONPATH\"\n"
            "Or pass --rynnvla-repo pointing at that directory.\n"
            f"Original error: {e}"
        ) from e
    return Solver(solver_args)


def _action_bounds() -> tuple[np.ndarray, np.ndarray]:
    from data_lerobot.norm_stats import get_action_stats

    min_values, max_values = get_action_stats()
    return np.asarray(min_values, dtype=np.float32), np.asarray(max_values, dtype=np.float32)


def _norm_action(action: np.ndarray, min_values: np.ndarray, max_values: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    return np.clip(2.0 * (action - min_values) / (max_values - min_values + 1e-8) - 1.0, -1.0, 1.0)


def _zero_norm(min_values: np.ndarray, max_values: np.ndarray) -> np.ndarray:
    min_values = np.asarray(min_values, dtype=np.float32)
    max_values = np.asarray(max_values, dtype=np.float32)
    return 2.0 * (0.0 - min_values) / (max_values - min_values + 1e-8) - 1.0


def _sign_agreement(pred: np.ndarray, gt: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    active = (np.abs(pred) > eps) | (np.abs(gt) > eps)
    pred_sign = np.sign(pred)
    gt_sign = np.sign(gt)
    matches = (pred_sign == gt_sign) & active
    counts = np.sum(active, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        agreement = np.divide(np.sum(matches, axis=0), counts, out=np.full_like(counts, np.nan, dtype=np.float32), where=counts > 0)
    return agreement.astype(np.float32), counts.astype(np.int32)


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    num = np.asarray(num, dtype=np.float32)
    den = np.asarray(den, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(num, den, out=np.full_like(num, np.nan, dtype=np.float32), where=den > 0)


def _joint_table(values: np.ndarray) -> dict[str, float | None]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    result: dict[str, float | None] = {}
    for idx, name in enumerate(_JOINT_LABELS[: len(flat)]):
        value = float(flat[idx])
        result[name] = None if np.isnan(value) else value
    return result


def _distribution_table(values: np.ndarray) -> dict[str, dict[str, float]]:
    values = np.asarray(values, dtype=np.float32)
    return {
        "mean": _joint_table(np.mean(values, axis=0)),
        "std": _joint_table(np.std(values, axis=0)),
        "min": _joint_table(np.min(values, axis=0)),
        "max": _joint_table(np.max(values, axis=0)),
    }


def _corrcoef_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return float("nan")
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= 1e-12 or y_std <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return {"slope": None, "intercept": None, "r2": None}
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_centered = x - x_mean
    denom = float(np.dot(x_centered, x_centered))
    if denom <= 1e-12:
        return {"slope": None, "intercept": None, "r2": None}
    slope = float(np.dot(x_centered, y - y_mean) / denom)
    intercept = float(y_mean - slope * x_mean)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = None if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)
    return {"slope": slope, "intercept": intercept, "r2": r2}


def _best_joint_match(pred_flat: np.ndarray, gt_flat: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for pred_idx, pred_name in enumerate(_JOINT_LABELS):
        pred_col = pred_flat[:, pred_idx]
        same_corr = _corrcoef_1d(pred_col, gt_flat[:, pred_idx])
        neg_corr = _corrcoef_1d(pred_col, -gt_flat[:, pred_idx])

        best_abs_corr = float("-inf")
        best_match: dict[str, Any] | None = None
        for gt_idx, gt_name in enumerate(_JOINT_LABELS):
            gt_col = gt_flat[:, gt_idx]
            corr = _corrcoef_1d(pred_col, gt_col)
            negated_corr = _corrcoef_1d(pred_col, -gt_col)
            candidates = (
                {"gt_joint": gt_name, "relation": "same", "corr": corr},
                {"gt_joint": gt_name, "relation": "negated", "corr": negated_corr},
            )
            for candidate in candidates:
                corr_value = candidate["corr"]
                if np.isnan(corr_value):
                    continue
                if abs(corr_value) > best_abs_corr:
                    best_abs_corr = abs(corr_value)
                    best_match = {
                        "gt_joint": candidate["gt_joint"],
                        "relation": candidate["relation"],
                        "corr": float(corr_value),
                    }

        fit_same = _linear_fit(gt_flat[:, pred_idx], pred_col)
        fit_negated = _linear_fit(-gt_flat[:, pred_idx], pred_col)

        result[pred_name] = {
            "same_joint_corr": None if np.isnan(same_corr) else same_corr,
            "negated_same_joint_corr": None if np.isnan(neg_corr) else neg_corr,
            "best_match": best_match,
            "fit_same_joint": fit_same,
            "fit_negated_same_joint": fit_negated,
        }
    return result


def _summarize(pred_steps: np.ndarray, gt_steps: np.ndarray, sign_eps: float, min_values: np.ndarray, max_values: np.ndarray) -> dict[str, Any]:
    # `gt_steps` are loaded directly from saved `abs_action` files. Those files
    # already contain target deltas for joints 0-4; subtracting state here would
    # double-apply the preprocessing transform.
    pred_flat = pred_steps.reshape(-1, pred_steps.shape[-1]).astype(np.float32)
    gt_flat = gt_steps.reshape(-1, gt_steps.shape[-1]).astype(np.float32)
    diff_flat = pred_flat - gt_flat

    pred_norm = _norm_action(pred_flat, min_values, max_values)
    gt_norm = _norm_action(gt_flat, min_values, max_values)
    diff_norm = pred_norm - gt_norm

    sign_agreement, sign_counts = _sign_agreement(pred_flat, gt_flat, sign_eps)
    sign_agreement_norm, sign_counts_norm = _sign_agreement(pred_norm, gt_norm, sign_eps)
    zero_norm = _zero_norm(min_values, max_values)
    sign_agreement_norm_centered, sign_counts_norm_centered = _sign_agreement(
        pred_norm - zero_norm.reshape(1, -1),
        gt_norm - zero_norm.reshape(1, -1),
        sign_eps,
    )

    per_joint_mae = np.mean(np.abs(diff_flat), axis=0)
    per_joint_bias = np.mean(diff_flat, axis=0)
    per_joint_gt_abs = np.mean(np.abs(gt_flat), axis=0)
    per_joint_pred_abs = np.mean(np.abs(pred_flat), axis=0)
    per_joint_mag_ratio = _safe_divide(per_joint_pred_abs, per_joint_gt_abs)

    per_joint_mae_norm = np.mean(np.abs(diff_norm), axis=0)
    per_joint_bias_norm = np.mean(diff_norm, axis=0)
    per_joint_gt_abs_norm = np.mean(np.abs(gt_norm), axis=0)
    per_joint_pred_abs_norm = np.mean(np.abs(pred_norm), axis=0)
    per_joint_mag_ratio_norm = _safe_divide(per_joint_pred_abs_norm, per_joint_gt_abs_norm)
    convention_checks = _best_joint_match(pred_flat, gt_flat)
    normalized_convention_checks = _best_joint_match(pred_norm, gt_norm)

    return {
        "action_convention": "saved target deltas for joints 0-4; gripper absolute; no eval-time state subtraction",
        "overall": {
            "mean_abs": float(np.mean(np.abs(diff_flat))),
            "max_abs": float(np.max(np.abs(diff_flat))),
            "l2": float(np.linalg.norm(diff_flat)),
            "magnitude_ratio": float(_safe_divide(np.array([np.mean(np.abs(pred_flat))]), np.array([np.mean(np.abs(gt_flat))]))[0]),
        },
        "normalized_overall": {
            "mean_abs": float(np.mean(np.abs(diff_norm))),
            "max_abs": float(np.max(np.abs(diff_norm))),
            "l2": float(np.linalg.norm(diff_norm)),
            "magnitude_ratio": float(_safe_divide(np.array([np.mean(np.abs(pred_norm))]), np.array([np.mean(np.abs(gt_norm))]))[0]),
        },
        "per_joint": {
            "mean_signed_bias": _joint_table(per_joint_bias),
            "mean_abs_error": _joint_table(per_joint_mae),
            "sign_agreement": _joint_table(sign_agreement),
            "sign_count": {k: int(v) for k, v in _joint_table(sign_counts.astype(np.float32)).items() if v is not None},
            "mean_abs_gt": _joint_table(per_joint_gt_abs),
            "mean_abs_pred": _joint_table(per_joint_pred_abs),
            "magnitude_ratio": _joint_table(per_joint_mag_ratio),
        },
        "normalized_per_joint": {
            "mean_signed_bias": _joint_table(per_joint_bias_norm),
            "mean_abs_error": _joint_table(per_joint_mae_norm),
            "sign_agreement": _joint_table(sign_agreement_norm),
            "sign_count": {k: int(v) for k, v in _joint_table(sign_counts_norm.astype(np.float32)).items() if v is not None},
            "mean_abs_gt": _joint_table(per_joint_gt_abs_norm),
            "mean_abs_pred": _joint_table(per_joint_pred_abs_norm),
            "magnitude_ratio": _joint_table(per_joint_mag_ratio_norm),
        },
        "normalized_centered_per_joint": {
            "sign_center": _joint_table(zero_norm),
            "sign_agreement": _joint_table(sign_agreement_norm_centered),
            "sign_count": {k: int(v) for k, v in _joint_table(sign_counts_norm_centered.astype(np.float32)).items() if v is not None},
        },
        "normalized_distribution": {
            "pred": _distribution_table(pred_norm),
            "gt": _distribution_table(gt_norm),
        },
        "raw_distribution": {
            "pred": _distribution_table(pred_flat),
            "gt": _distribution_table(gt_flat),
        },
        "convention_checks": convention_checks,
        "normalized_convention_checks": normalized_convention_checks,
    }


def _print_joint_metric(title: str, values: dict[str, float | None], precision: int = 6) -> None:
    print(f"\n{title}")
    for name in _JOINT_LABELS:
        if name not in values:
            continue
        value = values[name]
        if value is None:
            print(f"  {name}: n/a")
        else:
            print(f"  {name}: {value:.{precision}f}")


def _print_summary(summary: dict[str, Any]) -> None:
    print("\nOverall")
    for key, value in summary["overall"].items():
        print(f"  {key}: {value:.6f}")

    _print_joint_metric("Per-joint mean signed bias", summary["per_joint"]["mean_signed_bias"])
    _print_joint_metric("Per-joint MAE", summary["per_joint"]["mean_abs_error"])
    _print_joint_metric("Per-joint sign agreement", summary["per_joint"]["sign_agreement"])
    _print_joint_metric("Per-joint magnitude ratio", summary["per_joint"]["magnitude_ratio"])

    print("\nNormalized Overall")
    for key, value in summary["normalized_overall"].items():
        print(f"  {key}: {value:.6f}")

    _print_joint_metric("Normalized per-joint mean signed bias", summary["normalized_per_joint"]["mean_signed_bias"])
    _print_joint_metric("Normalized per-joint MAE", summary["normalized_per_joint"]["mean_abs_error"])
    _print_joint_metric("Normalized per-joint sign agreement", summary["normalized_per_joint"]["sign_agreement"])
    _print_joint_metric("Normalized per-joint magnitude ratio", summary["normalized_per_joint"]["magnitude_ratio"])

    print("\nConvention checks")
    for joint_name in _JOINT_LABELS:
        info = summary["convention_checks"][joint_name]
        best_match = info["best_match"]
        best_match_text = "n/a"
        if best_match is not None:
            best_match_text = f"{best_match['relation']} {best_match['gt_joint']} (corr={best_match['corr']:.6f})"
        print(
            f"  {joint_name}: same_corr={info['same_joint_corr'] if info['same_joint_corr'] is not None else 'n/a'} "
            f"neg_same_corr={info['negated_same_joint_corr'] if info['negated_same_joint_corr'] is not None else 'n/a'} "
            f"best_match={best_match_text}"
        )

    print("\nLinear fit checks")
    for joint_name in _JOINT_LABELS:
        info = summary["convention_checks"][joint_name]
        fit_same = info["fit_same_joint"]
        fit_neg = info["fit_negated_same_joint"]
        print(
            f"  {joint_name} vs same: slope={fit_same['slope'] if fit_same['slope'] is not None else 'n/a'} "
            f"intercept={fit_same['intercept'] if fit_same['intercept'] is not None else 'n/a'} "
            f"r2={fit_same['r2'] if fit_same['r2'] is not None else 'n/a'}"
        )
        print(
            f"  {joint_name} vs negated: slope={fit_neg['slope'] if fit_neg['slope'] is not None else 'n/a'} "
            f"intercept={fit_neg['intercept'] if fit_neg['intercept'] is not None else 'n/a'} "
            f"r2={fit_neg['r2'] if fit_neg['r2'] is not None else 'n/a'}"
        )


def _save_plots(
    out_dir: Path,
    stem: str,
    steps: list[int],
    pred_steps: np.ndarray,
    gt_steps: np.ndarray,
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "Plotting requested with --save-plots, but matplotlib is not available in this environment.\n"
            "Install matplotlib in the active env or rerun without --save-plots.\n"
            f"Original error: {e}"
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)
    step_axis = np.asarray(steps, dtype=np.int32)
    chunk_axis = np.arange(pred_steps.shape[1], dtype=np.int32)
    abs_err = np.abs(pred_steps - gt_steps).astype(np.float32)
    saved_paths: list[str] = []

    n_joints = len(_JOINT_LABELS)
    n_cols = min(3, n_joints)
    n_rows = int(math.ceil(n_joints / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 4.0 * n_rows))
    axes_flat = np.atleast_1d(axes).ravel()
    for idx, joint in enumerate(_JOINT_LABELS):
        ax = axes_flat[idx]
        gt_vals = gt_steps[:, :, idx].reshape(-1)
        pred_vals = pred_steps[:, :, idx].reshape(-1)
        ax.scatter(gt_vals, pred_vals, s=14, alpha=0.4, color="#2ca02c")
        low = float(min(np.min(gt_vals), np.min(pred_vals)))
        high = float(max(np.max(gt_vals), np.max(pred_vals)))
        if abs(high - low) <= 1e-12:
            low -= 1e-3
            high += 1e-3
        ax.plot([low, high], [low, high], linestyle="--", linewidth=1.0, color="#444444")
        corr = _corrcoef_1d(pred_vals, gt_vals)
        mae = float(np.mean(np.abs(pred_vals - gt_vals)))
        ax.set_title(f"{joint}\nMAE={mae:.4f} corr={corr:.3f}")
        ax.set_xlabel("gt")
        ax.set_ylabel("pred")
        ax.grid(True, alpha=0.25)
    for ax in axes_flat[n_joints:]:
        ax.set_visible(False)
    fig.suptitle("All chunk positions: GT vs Pred scatter", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    scatter_path = out_dir / f"{stem}_scatter_all_chunks.png"
    fig.savefig(scatter_path, dpi=180)
    plt.close(fig)
    saved_paths.append(str(scatter_path))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 2.9 * n_rows), sharex=False, sharey=False)
    axes_flat = axes.flatten()
    for idx, joint in enumerate(_JOINT_LABELS):
        ax = axes_flat[idx]
        im = ax.imshow(
            abs_err[:, :, idx].T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=(step_axis[0] - 0.5, step_axis[-1] + 0.5, chunk_axis[0] - 0.5, chunk_axis[-1] + 0.5),
            cmap="magma",
        )
        mae = float(np.mean(abs_err[:, :, idx]))
        max_err = float(np.max(abs_err[:, :, idx]))
        ax.set_title(f"{joint}\nMAE={mae:.4f} max={max_err:.4f}")
        ax.set_xlabel("episode step")
        ax.set_ylabel("chunk pos")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes_flat[n_joints:]:
        ax.set_visible(False)
    fig.suptitle("Absolute error heatmap by joint", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    heatmap_path = out_dir / f"{stem}_error_heatmap.png"
    fig.savefig(heatmap_path, dpi=180)
    plt.close(fig)
    saved_paths.append(str(heatmap_path))

    joint_mae = np.mean(abs_err, axis=(0, 1))
    joint_bias = np.mean(pred_steps - gt_steps, axis=(0, 1))
    joint_corr = np.asarray([_corrcoef_1d(pred_steps[:, :, idx].reshape(-1), gt_steps[:, :, idx].reshape(-1)) for idx in range(n_joints)], dtype=np.float32)
    joint_sign_agreement = np.asarray(
        [_sign_agreement(pred_steps[:, :, idx].reshape(-1), gt_steps[:, :, idx].reshape(-1), 1e-6)[0] for idx in range(n_joints)],
        dtype=np.float32,
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metric_specs = [
        ("MAE", joint_mae, "#d62728"),
        ("Signed bias", joint_bias, "#9467bd"),
        ("Correlation", joint_corr, "#2ca02c"),
        ("Sign agreement", joint_sign_agreement, "#1f77b4"),
    ]
    for ax, (title, values, color) in zip(axes.ravel(), metric_specs):
        ax.bar(_JOINT_LABELS, values, color=color, alpha=0.9)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=25)
        if title in {"Correlation", "Sign agreement"}:
            ax.set_ylim(-1.0 if title == "Correlation" else 0.0, 1.0)
        ax.axhline(0.0, color="#777777", linewidth=0.8)
    fig.suptitle("Per-joint summary across all chunk positions", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    summary_path = out_dir / f"{stem}_summary.png"
    fig.savefig(summary_path, dpi=180)
    plt.close(fig)
    saved_paths.append(str(summary_path))

    return saved_paths


def main() -> None:
    args = _parse_args()
    root = step_eval._repo_root()
    os.chdir(root)

    if args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")
    if args.stride <= 0:
        raise ValueError("--stride must be > 0")

    cfg_path = Path(args.positronic_config)
    if not cfg_path.is_absolute():
        cfg_path = (root / cfg_path).resolve()
    cfg = step_eval._load_positronic_config(cfg_path)

    _configure_env(root, cfg)
    ckpt_path = step_eval._resolve_checkpoint(args, cfg)
    step_eval._ensure_rynnvla_on_path(args.rynnvla_repo.strip() or None)
    output_dir = _resolve_output_dir(root, args, cfg, ckpt_path)
    solver_args = _make_solver_args(args, cfg, ckpt_path, output_dir)

    episode_dir = Path(args.episode).expanduser().resolve()
    his = int(cfg["his"])
    prompt = args.prompt if args.prompt is not None else step_eval._extract_task_from_episode(episode_dir)

    Path(solver_args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Loading Solver from {ckpt_path} ...")
    solver = _load_solver(args.rynnvla_repo, solver_args)
    min_values, max_values = _action_bounds()

    action_dirs = step_eval._sorted_action_dirs(episode_dir / "abs_action")
    last_step = len(action_dirs) - 1
    if args.start_step < 0 or args.start_step > last_step:
        raise ValueError(f"--start-step {args.start_step} out of range for {episode_dir.name}; valid range is 0..{last_step}")

    steps = list(range(args.start_step, last_step + 1, args.stride))[: args.max_steps]
    if not steps:
        raise ValueError("No evaluation steps selected")

    pred_steps = []
    gt_steps = []
    records = []

    print(f"episode={episode_dir}")
    print(f"prompt={prompt}")
    print(f"steps={steps}")

    for index, step in enumerate(steps, start=1):
        sample = step_eval._build_sample(episode_dir, step=step, his=his)
        step_eval._reset_solver_history(solver, sample)
        pred = solver.get_action_wrist_action_head_state(
            front_image=sample["front_current"],
            wrist_image=sample["wrist_current"],
            state=sample["state"],
            prompt=prompt,
        )
        pred = np.asarray(pred, dtype=np.float32)
        gt = np.asarray(sample["gt_action"], dtype=np.float32)
        if pred.shape != gt.shape:
            raise ValueError(f"step {step}: pred shape {pred.shape} != gt shape {gt.shape}")

        pred_steps.append(pred)
        gt_steps.append(gt)
        step_metrics = step_eval._metrics(pred, gt)
        records.append(
            {
                "step": step,
                "metrics": step_metrics,
                "first_action_gt": np.round(gt[0], 6).tolist(),
                "first_action_pred": np.round(pred[0], 6).tolist(),
                "first_action_diff": np.round(pred[0] - gt[0], 6).tolist(),
            }
        )
        print(f"[{index}/{len(steps)}] step={step} mean_abs={step_metrics['mean_abs']:.6f} first_step_mean_abs={step_metrics['first_step_mean_abs']:.6f}")

    pred_arr = np.asarray(pred_steps, dtype=np.float32)
    gt_arr = np.asarray(gt_steps, dtype=np.float32)
    summary = _summarize(pred_arr, gt_arr, args.sign_eps, min_values, max_values)
    report_stem = f"{episode_dir.name}_steps_{steps[0]:06d}_to_{steps[-1]:06d}"

    print(f"\nEvaluated {len(steps)} steps")
    _print_summary(summary)

    if args.save_plots:
        saved_plot_paths = _save_plots(Path(solver_args.output_dir), report_stem, steps, pred_arr, gt_arr)
        print("\nSaved plots")
        for plot_path in saved_plot_paths:
            print(f"  {plot_path}")

    if args.save_json:
        out_path = Path(solver_args.output_dir) / f"{report_stem}.json"
        payload = {
            "episode": str(episode_dir),
            "prompt": prompt,
            "checkpoint": str(ckpt_path),
            "steps": steps,
            "step_count": len(steps),
            "chunk_size": int(pred_arr.shape[1]),
            "action_dim": int(pred_arr.shape[2]),
            "sign_eps": args.sign_eps,
            "action_stats_file": os.environ.get("RYNNVLA_ACTION_STATS_FILE", ""),
            "action_min": np.round(min_values, 6).tolist(),
            "action_max": np.round(max_values, 6).tolist(),
            "summary": summary,
            "per_step": records,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved JSON report to {out_path}")


if __name__ == "__main__":
    main()
