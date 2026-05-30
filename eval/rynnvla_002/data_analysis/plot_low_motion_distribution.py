#!/usr/bin/env python3
"""
Plot the distribution of low-motion chunk fractions across training episodes.

Reads action_motion_report/summary.json and saves a histogram showing what
fraction of each episode's action chunks are near-static (arm mean |action| < threshold).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot low-motion chunk fraction distribution.")
    parser.add_argument(
        "--motion-summary",
        type=Path,
        default=Path("eval_output/screwdriver_so101/data_analysis/action_motion_report/summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_output/screwdriver_so101/data_analysis/low_motion_distribution.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.motion_summary.exists():
        raise SystemExit(f"motion summary not found: {args.motion_summary}\nRun action_episode_motion_report.py first.")

    motion = json.loads(args.motion_summary.read_text())
    low_frac = np.asarray([row["arm_low_motion_chunk_fraction"] for row in motion], dtype=np.float32)
    threshold = motion[0]["arm_low_motion_threshold"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0.0, max(0.26, float(low_frac.max()) + 0.02), 14)
    ax.hist(low_frac, bins=bins, color="#4f6d7a", edgecolor="white")
    ax.axvline(float(np.median(low_frac)), color="#c8553d", linewidth=2, label=f"median={np.median(low_frac):.3f}")
    ax.axvline(float(np.mean(low_frac)), color="#2a9d8f", linewidth=2, linestyle="--", label=f"mean={np.mean(low_frac):.3f}")
    ax.set_title("Training Supervision: Low-Motion Chunk Fraction")
    ax.set_xlabel(f"fraction of chunks with arm mean |action| < {threshold:.3f}")
    ax.set_ylabel("episode count")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=10)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"wrote {args.output}")
    print(f"episodes={len(low_frac)}  mean={np.mean(low_frac):.3f}  median={np.median(low_frac):.3f}  max={np.max(low_frac):.3f}")


if __name__ == "__main__":
    main()
