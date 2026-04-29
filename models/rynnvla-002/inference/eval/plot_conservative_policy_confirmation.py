#!/usr/bin/env python3
"""
Render a compact confirmation plot for the conservative low-amplitude policy story.

Inputs:
  - action_motion_report/summary.json
  - one or more episode_batch_eval JSON reports

Output:
  - single PNG that shows:
      1) training low-motion chunk fraction distribution
      2) evaluated-window GT vs predicted arm mean |action|
      3) per-joint magnitude ratios on those evaluated windows
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

JOINT_LABELS = ("joint_0", "joint_1", "joint_2", "joint_3", "joint_4")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot conservative-policy confirmation figure.")
    parser.add_argument(
        "--motion-summary",
        type=Path,
        default=Path("training_output/screwdriver_so101/action_motion_report/summary.json"),
    )
    parser.add_argument(
        "--eval-report",
        type=Path,
        action="append",
        default=[],
        help="Path to an episode_batch_eval JSON. Pass multiple times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_output/screwdriver_so101/conservative_policy_confirmation.png"),
    )
    return parser.parse_args()


def _load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _report_label(path: Path, payload: dict) -> str:
    episode = Path(payload["episode"]).name.replace("episode_", "ep")
    ckpt = Path(payload["checkpoint"]).name
    return f"{ckpt}\n{episode}"


def _plot(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    motion = _load_json(args.motion_summary)
    eval_payloads = [_load_json(path) for path in args.eval_report]

    low_frac = np.asarray([row["arm_low_motion_chunk_fraction"] for row in motion], dtype=np.float32)
    arm_p10 = np.asarray([row["arm_p10_chunk_mean_abs"] for row in motion], dtype=np.float32)

    labels = [_report_label(path, payload) for path, payload in zip(args.eval_report, eval_payloads)]
    gt_overall = np.asarray([p["summary"]["overall"]["magnitude_ratio"] for p in eval_payloads], dtype=np.float32)
    mean_abs_gt = np.asarray(
        [np.mean([p["summary"]["per_joint"]["mean_abs_gt"][joint] for joint in JOINT_LABELS]) for p in eval_payloads],
        dtype=np.float32,
    )
    mean_abs_pred = np.asarray(
        [np.mean([p["summary"]["per_joint"]["mean_abs_pred"][joint] for joint in JOINT_LABELS]) for p in eval_payloads],
        dtype=np.float32,
    )
    per_joint_ratio = np.asarray(
        [[p["summary"]["per_joint"]["magnitude_ratio"][joint] for joint in JOINT_LABELS] for p in eval_payloads],
        dtype=np.float32,
    )

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1], width_ratios=[1.0, 1.2], hspace=0.28, wspace=0.22)

    ax = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0.0, max(0.26, float(low_frac.max()) + 0.01), 12)
    ax.hist(low_frac, bins=bins, color="#4f6d7a", edgecolor="white")
    ax.axvline(float(np.median(low_frac)), color="#c8553d", linewidth=2, label=f"median={np.median(low_frac):.3f}")
    ax.axvline(float(np.mean(low_frac)), color="#2a9d8f", linewidth=2, linestyle="--", label=f"mean={np.mean(low_frac):.3f}")
    ax.set_title("Training Supervision: Low-Motion Chunk Fraction")
    ax.set_xlabel("fraction of chunks with arm mean |action| < 0.01")
    ax.set_ylabel("episode count")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(len(labels))
    width = 0.32
    ax.bar(x - width / 2, mean_abs_gt, width=width, color="#264653", label="training target mean |action|")
    ax.bar(x + width / 2, mean_abs_pred, width=width, color="#e76f51", label="model prediction mean |action|")
    for idx, ratio in enumerate(gt_overall):
        ax.text(idx, max(mean_abs_gt[idx], mean_abs_pred[idx]) + 0.003, f"ratio={ratio:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, labels=labels)
    ax.set_ylabel("mean |action| across arm joints")
    ax.set_title("Offline Eval: Predicted Actions Are Smaller Than Training Targets")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[1, :])
    im = ax.imshow(per_joint_ratio, aspect="auto", interpolation="nearest", cmap="RdYlBu_r", vmin=0.0, vmax=1.25)
    ax.set_xticks(np.arange(len(JOINT_LABELS)), labels=JOINT_LABELS)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_title("Per-Joint Magnitude Ratio = mean |pred| / mean |target|")
    for row_idx in range(per_joint_ratio.shape[0]):
        for col_idx in range(per_joint_ratio.shape[1]):
            ax.text(col_idx, row_idx, f"{per_joint_ratio[row_idx, col_idx]:.2f}", ha="center", va="center", fontsize=9, color="black")
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("magnitude ratio")

    fig.suptitle(
        "Conservative Policy Confirmation\nTraining has some low-information supervision, and evaluated predictions under-shoot target action scale.",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)

    print(f"wrote {args.output}")
    print(f"training_low_frac_mean={np.mean(low_frac):.4f}")
    print(f"training_low_frac_median={np.median(low_frac):.4f}")
    for label, gt_val, pred_val, ratio in zip(labels, mean_abs_gt, mean_abs_pred, gt_overall):
        print(f"{label}: target_mean_abs={gt_val:.4f} pred_mean_abs={pred_val:.4f} ratio={ratio:.4f}")


def main() -> None:
    args = _parse_args()
    if not args.eval_report:
        raise SystemExit("pass at least one --eval-report")
    _plot(args)


if __name__ == "__main__":
    main()
