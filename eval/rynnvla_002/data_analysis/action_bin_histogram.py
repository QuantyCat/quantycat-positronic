#!/usr/bin/env python3
"""
Plot per-joint bin distributions for RynnVLA training data.

For each joint, shows how the 256 discrete action/state bins are populated
across all training samples. Reveals bin starvation, imbalance, and task-specific
pose distributions.

Reads:
    task_dir/episode_*/abs_action/action_*/*.npy   — action delta targets
    task_dir/episode_*/state/state_*.npy           — absolute state values

Outputs:
    out_dir/action_bins.png    — per-joint action bin histograms
    out_dir/state_bins.png     — per-joint state bin histograms
    out_dir/summary.json       — bins occupied, peak bin, concentration stats

Run from repo root:
    python3 models/rynnvla-002/eval/data_analysis/action_bin_histogram.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

JOINT_LABELS = ("joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "gripper")
N_BINS = 256
BIN_EDGES = np.linspace(-1.0, 1.0, N_BINS)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0  # 255 centers


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training data bin distributions.")
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=Path("my_data/training_pipeline/training_data/Put_the_screwdriver_into_the_cup"),
        help="Directory containing episode_* folders.",
    )
    parser.add_argument(
        "--action-stats",
        type=Path,
        default=Path("my_data/training_pipeline/min_max_action.txt"),
        help="min_max_action.txt produced by step4.",
    )
    parser.add_argument(
        "--state-stats",
        type=Path,
        default=Path("my_data/training_pipeline/min_max_state.txt"),
        help="min_max_state.txt produced by step4.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("eval_output/screwdriver_so101/data_analysis/action_bin_histogram"),
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=0,
        help="Max action vectors to load (0 = all).",
    )
    return parser.parse_args()


def _parse_stats_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mins: list[float] = []
    maxs: list[float] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if "|" not in line:
                continue
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
            if len(nums) < 3:
                continue
            mins.append(float(nums[1]))
            maxs.append(float(nums[2]))
    if len(mins) != len(JOINT_LABELS):
        raise SystemExit(f"expected {len(JOINT_LABELS)} stat rows in {path}, got {len(mins)}")
    return np.asarray(mins, dtype=np.float32), np.asarray(maxs, dtype=np.float32)


def _norm(x: np.ndarray, mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
    return np.clip(2.0 * (x - mn) / (mx - mn + 1e-8) - 1.0, -1.0, 1.0)


def _to_bins(normed: np.ndarray) -> np.ndarray:
    indices = np.digitize(normed, BIN_EDGES) - 1
    return np.clip(indices, 0, N_BINS - 1)


def _load_vectors(paths: list[Path]) -> np.ndarray:
    vecs = []
    for p in paths:
        arr = np.load(p).astype(np.float32)
        if arr.shape == (len(JOINT_LABELS),):
            vecs.append(arr)
    if not vecs:
        raise SystemExit(f"no valid {len(JOINT_LABELS)}-dim vectors found")
    return np.stack(vecs, axis=0)


def _bin_stats(counts: np.ndarray, mn: float, mx: float) -> dict[str, Any]:
    total = counts.sum()
    occupied = int((counts > 0).sum())
    nz = np.where(counts > 0)[0]
    peak_bin = int(counts.argmax())
    peak_norm = float(-1.0 + peak_bin * 2.0 / (N_BINS - 1))
    top1_pct = float(counts.max() / total * 100)
    top5_pct = float(counts[counts.argsort()[-5:]].sum() / total * 100)
    return {
        "bins_occupied": occupied,
        "bins_occupied_pct": round(occupied / N_BINS * 100, 1),
        "range": [int(nz[0]), int(nz[-1])] if len(nz) else [0, 0],
        "peak_bin": peak_bin,
        "peak_norm": round(peak_norm, 4),
        "peak_raw": round(peak_norm / 2.0 * (mx - mn) + (mn + mx) / 2.0, 5),
        "top1_bin_pct": round(top1_pct, 1),
        "top5_bins_pct": round(top5_pct, 1),
    }


def _plot_histograms(
    all_counts: np.ndarray,
    stat_mins: np.ndarray,
    stat_maxs: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_joints = len(JOINT_LABELS)
    fig, axes = plt.subplots(n_joints, 1, figsize=(12, 3 * n_joints))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for j, (ax, label) in enumerate(zip(axes, JOINT_LABELS)):
        counts = all_counts[j]
        total = counts.sum()
        x = np.arange(N_BINS)

        # colour bars by density: grey = rare, blue = moderate, red = heavy
        colours = []
        for c in counts:
            pct = c / total * 100
            if pct == 0:
                colours.append("#e0e0e0")
            elif pct < 1.0:
                colours.append("#a8c8e8")
            elif pct < 5.0:
                colours.append("#3a88c8")
            else:
                colours.append("#c03030")

        ax.bar(x, counts, width=1.0, color=colours, linewidth=0)

        occupied = int((counts > 0).sum())
        peak_bin = int(counts.argmax())
        peak_pct = float(counts[peak_bin] / total * 100)

        # mark the raw-zero position (where "no movement" maps)
        zero_norm = float(_norm(np.array([0.0]), np.array([stat_mins[j]]), np.array([stat_maxs[j]]))[0])
        zero_bin = int(np.clip(np.digitize([zero_norm], BIN_EDGES)[0] - 1, 0, N_BINS - 1))
        ax.axvline(zero_bin, color="green", linewidth=1.5, linestyle="--", label=f"raw zero → bin {zero_bin}")

        ax.set_title(
            f"{label}  |  {occupied}/256 bins occupied  |  peak bin {peak_bin} ({peak_pct:.1f}%)",
            fontsize=10,
        )
        ax.set_xlabel("bin index  (0 = −1.0 normalized,  255 = +1.0 normalized)")
        ax.set_ylabel("count")
        ax.set_xlim(-1, N_BINS)
        ax.legend(fontsize=8)

        # secondary x-axis showing normalized value
        ax2 = ax.twiny()
        ax2.set_xlim(-1, N_BINS)
        tick_bins = [0, 64, 128, 192, 255]
        ax2.set_xticks(tick_bins)
        ax2.set_xticklabels([f"{-1.0 + b * 2.0 / 255:.2f}" for b in tick_bins], fontsize=8)
        ax2.set_xlabel("normalized value", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main() -> None:
    args = _parse_args()

    action_min, action_max = _parse_stats_file(args.action_stats)
    state_min, state_max = _parse_stats_file(args.state_stats)

    # --- load actions ---
    action_paths = sorted(args.task_dir.glob("episode_*/abs_action/action_*/*.npy"))
    if not action_paths:
        raise SystemExit(f"no action files found under {args.task_dir}")
    if args.sample_limit > 0:
        action_paths = action_paths[: args.sample_limit]
    print(f"loading {len(action_paths)} action vectors …")
    actions = _load_vectors(action_paths)

    # --- load states ---
    state_paths = sorted(args.task_dir.glob("episode_*/state/state_*.npy"))
    if not state_paths:
        raise SystemExit(f"no state files found under {args.task_dir}")
    if args.sample_limit > 0:
        state_paths = state_paths[: args.sample_limit]
    print(f"loading {len(state_paths)} state vectors …")
    states = _load_vectors(state_paths)

    # --- compute bin counts ---
    action_counts = np.zeros((len(JOINT_LABELS), N_BINS), dtype=np.int64)
    state_counts = np.zeros((len(JOINT_LABELS), N_BINS), dtype=np.int64)

    for j in range(len(JOINT_LABELS)):
        a_norm = _norm(actions[:, j], action_min[j:j+1], action_max[j:j+1])
        action_counts[j] = np.bincount(_to_bins(a_norm), minlength=N_BINS)

        s_norm = _norm(states[:, j], state_min[j:j+1], state_max[j:j+1])
        state_counts[j] = np.bincount(_to_bins(s_norm), minlength=N_BINS)

    # --- plots ---
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("plotting …")
    _plot_histograms(
        action_counts, action_min, action_max,
        f"Action delta bin distributions  (n={len(actions):,})",
        args.out_dir / "action_bins.png",
    )
    _plot_histograms(
        state_counts, state_min, state_max,
        f"State bin distributions  (n={len(states):,})",
        args.out_dir / "state_bins.png",
    )

    # --- summary JSON ---
    summary: dict[str, Any] = {
        "n_action_samples": len(actions),
        "n_state_samples": len(states),
        "n_bins": N_BINS,
        "action_stats_file": str(args.action_stats.resolve()),
        "state_stats_file": str(args.state_stats.resolve()),
        "actions": {},
        "states": {},
    }
    for j, label in enumerate(JOINT_LABELS):
        summary["actions"][label] = _bin_stats(action_counts[j], float(action_min[j]), float(action_max[j]))
        summary["states"][label] = _bin_stats(state_counts[j], float(state_min[j]), float(state_max[j]))

    out_json = args.out_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"  saved {out_json}")

    # --- console summary ---
    print("\n=== ACTION BINS ===")
    print(f"  {'joint':<12} {'occupied':>8}  {'top1%':>6}  {'top5%':>6}  {'peak_bin':>8}  zero_bin")
    for j, label in enumerate(JOINT_LABELS):
        s = summary["actions"][label]
        zero_norm = float(_norm(np.array([0.0]), action_min[j:j+1], action_max[j:j+1])[0])
        zero_bin = int(np.clip(np.digitize([zero_norm], BIN_EDGES)[0] - 1, 0, N_BINS - 1))
        print(f"  {label:<12} {s['bins_occupied']:>5}/256  {s['top1_bin_pct']:>5.1f}%  {s['top5_bins_pct']:>5.1f}%  {s['peak_bin']:>8}  {zero_bin}")

    print("\n=== STATE BINS ===")
    print(f"  {'joint':<12} {'occupied':>8}  {'top1%':>6}  {'top5%':>6}  {'peak_bin':>8}")
    for j, label in enumerate(JOINT_LABELS):
        s = summary["states"][label]
        print(f"  {label:<12} {s['bins_occupied']:>5}/256  {s['top1_bin_pct']:>5.1f}%  {s['top5_bins_pct']:>5.1f}%  {s['peak_bin']:>8}")


if __name__ == "__main__":
    main()
