#!/usr/bin/env python3
"""
Create visual per-episode motion reports for saved RynnVLA action chunks.

The report answers: how much did each joint move in each demonstration episode?
It reads:

    task_dir/episode_XXXXXX/abs_action/action_T/0.npy ... N.npy

Despite the directory name, saved `abs_action` files are already the training
targets: relative deltas for joints 0-4 and absolute gripper targets. This report
must analyze those saved values directly, with no state subtraction.

and writes PNG plots plus a compact HTML index.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np

JOINT_LABELS = ("joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "gripper")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build visual episode motion report.")
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=Path("my_data/training_pipeline/training_data/Put_the_screwdriver_into_the_cup"),
        help="Directory containing episode_* folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("eval_output/screwdriver_so101/data_analysis/action_motion_report"),
        help="Output directory for PNG/HTML files.",
    )
    parser.add_argument(
        "--static-threshold",
        type=float,
        default=0.01,
        help="Chunk arm mean absolute motion below this is counted as low-motion.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip regenerating PNG plots. Useful when matplotlib is unavailable.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def _import_pyplot():
    import matplotlib.pyplot as plt

    return plt


def _load_chunk(action_dir: Path) -> np.ndarray:
    files = sorted(action_dir.glob("*.npy"), key=lambda p: int(p.stem))
    if not files:
        raise ValueError(f"no .npy files under {action_dir}")
    return np.asarray([np.load(p).astype(np.float32) for p in files], dtype=np.float32)


def _load_episode_chunks(episode_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    action_root = episode_dir / "abs_action"
    action_dirs = sorted(action_root.glob("action_*"), key=lambda p: int(p.name.split("_")[1]))
    steps: list[int] = []
    chunks: list[np.ndarray] = []
    for action_dir in action_dirs:
        try:
            chunks.append(_load_chunk(action_dir))
            steps.append(int(action_dir.name.split("_")[1]))
        except Exception as exc:
            print(f"warning: skipping {action_dir}: {exc}")
    if not chunks:
        raise ValueError(f"no chunks found for {episode_dir}")
    return np.asarray(steps, dtype=np.int32), np.stack(chunks, axis=0)


def _run_lengths(mask: np.ndarray) -> list[int]:
    runs: list[int] = []
    current = 0
    for value in mask:
        if value:
            current += 1
            continue
        if current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def _episode_summary(episode: str, steps: np.ndarray, chunks: np.ndarray, static_threshold: float) -> dict[str, Any]:
    arm = chunks[:, :, :-1]
    flat = chunks.reshape(-1, chunks.shape[-1])
    first = chunks[:, 0, :]
    arm_mean_by_chunk = np.mean(np.abs(arm), axis=(1, 2))
    low_motion_mask = arm_mean_by_chunk < static_threshold
    low_motion_runs = _run_lengths(low_motion_mask)
    return {
        "episode": episode,
        "num_chunks": int(chunks.shape[0]),
        "start_step": int(steps[0]),
        "end_step": int(steps[-1]),
        "arm_mean_abs": float(np.mean(np.abs(arm))),
        "arm_p10_chunk_mean_abs": float(np.percentile(arm_mean_by_chunk, 10)),
        "arm_p90_chunk_mean_abs": float(np.percentile(arm_mean_by_chunk, 90)),
        "arm_max_abs": float(np.max(np.abs(arm))),
        "arm_low_motion_threshold": float(static_threshold),
        "arm_low_motion_chunk_fraction": float(np.mean(low_motion_mask)),
        "arm_low_motion_run_count": int(len(low_motion_runs)),
        "arm_low_motion_run_p95_chunks": float(np.percentile(low_motion_runs, 95)) if low_motion_runs else 0.0,
        "arm_low_motion_run_max_chunks": int(max(low_motion_runs, default=0)),
        "joint_mean_abs": {name: float(np.mean(np.abs(flat[:, i]))) for i, name in enumerate(JOINT_LABELS)},
        "first_step_joint_mean_abs": {name: float(np.mean(np.abs(first[:, i]))) for i, name in enumerate(JOINT_LABELS)},
        "near_zero_fraction": {name: float(np.mean(np.abs(flat[:, i]) < 0.01)) for i, name in enumerate(JOINT_LABELS)},
    }


def _plot_episode(out_path: Path, episode: str, steps: np.ndarray, chunks: np.ndarray) -> None:
    plt = _import_pyplot()
    flat_mean_abs = np.mean(np.abs(chunks), axis=1)
    first = chunks[:, 0, :]
    arm_mean = np.mean(np.abs(chunks[:, :, :-1]), axis=(1, 2))
    gripper = chunks[:, :, -1]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"{episode} action motion", fontsize=15)

    ax = axes[0]
    for i, name in enumerate(JOINT_LABELS[:-1]):
        ax.plot(steps, flat_mean_abs[:, i], linewidth=1.5, label=name)
    ax.set_ylabel("chunk mean |action|")
    ax.set_title("Per-joint motion magnitude across the 5-step chunk")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=5, fontsize=9)

    ax = axes[1]
    for i, name in enumerate(JOINT_LABELS[:-1]):
        ax.plot(steps, first[:, i], linewidth=1.2, label=name)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("first action")
    ax.set_title("Signed first-step relative action, joints 0-4")
    ax.grid(True, alpha=0.25)

    ax = axes[2]
    ax.plot(steps, arm_mean, color="#202020", linewidth=1.6, label="arm mean |action|")
    ax.plot(steps, np.mean(np.abs(gripper), axis=1), color="#c43b3b", linewidth=1.4, label="gripper mean |action|")
    ax.axhline(0.01, color="#777777", linestyle="--", linewidth=1.0, label="0.01 near-static threshold")
    ax.set_xlabel("episode step")
    ax.set_ylabel("magnitude")
    ax.set_title("Overall arm vs gripper motion")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_overview(out_path: Path, summaries: list[dict[str, Any]]) -> None:
    plt = _import_pyplot()
    episodes = [s["episode"].replace("episode_", "ep") for s in summaries]
    labels = list(JOINT_LABELS)
    values = np.asarray([[s["joint_mean_abs"][name] for name in labels] for s in summaries], dtype=np.float32)

    fig, axes = plt.subplots(2, 1, figsize=(16, 11))
    ax = axes[0]
    im = ax.imshow(values.T, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xticks(np.arange(len(episodes)), labels=episodes, rotation=90)
    ax.set_title("Mean absolute action by episode and joint")
    ax.set_ylabel("joint")
    fig.colorbar(im, ax=ax, label="mean |action|")

    ax = axes[1]
    x = np.arange(len(summaries))
    width = 0.13
    for i, name in enumerate(labels[:-1]):
        ax.bar(x + (i - 2) * width, values[:, i], width=width, label=name)
    ax.plot(x, values[:, -1], color="#c43b3b", linewidth=2, marker="o", markersize=3, label="gripper")
    ax.set_xticks(x, episodes, rotation=90)
    ax.set_ylabel("mean |action|")
    ax.set_title("Per-episode mean motion")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=6, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_low_motion_overview(out_path: Path, summaries: list[dict[str, Any]], static_threshold: float) -> None:
    plt = _import_pyplot()
    episodes = [s["episode"].replace("episode_", "ep") for s in summaries]
    x = np.arange(len(summaries))
    low_frac = np.asarray([s["arm_low_motion_chunk_fraction"] for s in summaries], dtype=np.float32)
    low_run_p95 = np.asarray([s["arm_low_motion_run_p95_chunks"] for s in summaries], dtype=np.float32)
    low_run_max = np.asarray([s["arm_low_motion_run_max_chunks"] for s in summaries], dtype=np.float32)
    arm_p10 = np.asarray([s["arm_p10_chunk_mean_abs"] for s in summaries], dtype=np.float32)
    arm_mean = np.asarray([s["arm_mean_abs"] for s in summaries], dtype=np.float32)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    ax = axes[0]
    ax.bar(x, low_frac, color="#496a81", label=f"fraction of chunks with arm mean < {static_threshold:.3f}")
    ax.set_ylabel("low-motion chunk fraction")
    ax.set_title("Arm low-motion prevalence by episode")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=9, loc="upper right")

    ax2 = ax.twinx()
    ax2.plot(x, arm_p10, color="#c95f2d", linewidth=1.8, marker="o", markersize=3, label="arm p10 chunk mean |action|")
    ax2.plot(x, arm_mean, color="#202020", linewidth=1.6, marker="o", markersize=3, label="arm mean |action|")
    ax2.set_ylabel("motion magnitude")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, fontsize=9, loc="upper left")

    ax = axes[1]
    width = 0.38
    ax.bar(x - width / 2, low_run_p95, width=width, color="#7a9e7e", label="low-motion run p95")
    ax.bar(x + width / 2, low_run_max, width=width, color="#b55d4c", label="low-motion run max")
    ax.set_ylabel("contiguous low-motion chunks")
    ax.set_title("Arm low-motion run lengths by episode")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=9)
    ax.set_xticks(x, episodes, rotation=90)
    ax.set_xlabel("episode")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _write_html(out_dir: Path, summaries: list[dict[str, Any]], top_k: int, static_threshold: float) -> None:
    rows = []
    for s in summaries:
        cells = [
            html.escape(s["episode"]),
            str(s["num_chunks"]),
            f"{s['arm_mean_abs']:.4f}",
            f"{s['arm_p10_chunk_mean_abs']:.4f}",
            f"{s['arm_p90_chunk_mean_abs']:.4f}",
            f"{s['arm_max_abs']:.4f}",
            f"{s['arm_low_motion_chunk_fraction']:.3f}",
            f"{s['arm_low_motion_run_p95_chunks']:.1f}",
            str(s["arm_low_motion_run_max_chunks"]),
        ]
        cells.extend(f"{s['joint_mean_abs'][name]:.4f}" for name in JOINT_LABELS)
        img = f"episodes/{s['episode']}.png"
        cells.append(f'<a href="{img}">plot</a>')
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")

    top = sorted(summaries, key=lambda s: s["arm_mean_abs"], reverse=True)[:top_k]
    low = sorted(summaries, key=lambda s: s["arm_mean_abs"])[:top_k]
    longest_low_runs = sorted(summaries, key=lambda s: s["arm_low_motion_run_max_chunks"], reverse=True)[:top_k]
    highest_low_fraction = sorted(summaries, key=lambda s: s["arm_low_motion_chunk_fraction"], reverse=True)[:top_k]

    def _rank_list(items: list[dict[str, Any]]) -> str:
        return "<ol>" + "".join(
            f'<li><a href="episodes/{s["episode"]}.png">{html.escape(s["episode"])}</a>: '
            f'arm mean |action| {s["arm_mean_abs"]:.4f}</li>'
            for s in items
        ) + "</ol>"

    def _low_rank_list(items: list[dict[str, Any]]) -> str:
        return "<ol>" + "".join(
            f'<li><a href="episodes/{s["episode"]}.png">{html.escape(s["episode"])}</a>: '
            f'low frac {s["arm_low_motion_chunk_fraction"]:.3f}, '
            f'max run {s["arm_low_motion_run_max_chunks"]} chunks</li>'
            for s in items
        ) + "</ol>"

    headers = [
        "episode",
        "chunks",
        "arm mean",
        "arm p10",
        "arm p90",
        "arm max",
        f"low frac < {static_threshold:.3f}",
        "low run p95",
        "low run max",
        *JOINT_LABELS,
        "plot",
    ]
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Action Motion Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #202124; }}
    h1, h2 {{ margin-bottom: 8px; }}
    img {{ max-width: 100%; border: 1px solid #ddd; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ position: sticky; top: 0; background: white; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  </style>
</head>
<body>
  <h1>Action Motion Report</h1>
  <p>Each value is mean absolute saved target action. Joints 0-4 are relative joint targets; gripper is absolute target.</p>
  <p>Low-motion metrics are computed from per-chunk arm mean |action| with threshold &lt; {static_threshold:.3f}. Run lengths are contiguous low-motion chunks.</p>
  <h2>Overview</h2>
  <img src="overview.png" alt="overview">
  <h2>Arm Low-Motion Overview</h2>
  <img src="arm_low_motion_overview.png" alt="arm low motion overview">
  <div class="grid">
    <section>
      <h2>Highest Arm Motion</h2>
      {_rank_list(top)}
    </section>
    <section>
      <h2>Lowest Arm Motion</h2>
      {_rank_list(low)}
    </section>
  </div>
  <div class="grid">
    <section>
      <h2>Longest Low-Motion Runs</h2>
      {_low_rank_list(longest_low_runs)}
    </section>
    <section>
      <h2>Highest Low-Motion Fraction</h2>
      {_low_rank_list(highest_low_fraction)}
    </section>
  </div>
  <h2>All Episodes</h2>
  <table>
    <thead><tr>{''.join(f'<th>{html.escape(h)}</th>' for h in headers)}</tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html_text)


def main() -> None:
    args = _parse_args()
    task_dir = args.task_dir.resolve()
    out_dir = args.out_dir.resolve()
    episode_plot_dir = out_dir / "episodes"
    episode_plot_dir.mkdir(parents=True, exist_ok=True)

    episode_dirs = sorted(p for p in task_dir.iterdir() if p.is_dir() and p.name.startswith("episode_"))
    if not episode_dirs:
        raise SystemExit(f"no episode directories found under {task_dir}")

    summaries: list[dict[str, Any]] = []
    for episode_dir in episode_dirs:
        steps, chunks = _load_episode_chunks(episode_dir)
        summary = _episode_summary(episode_dir.name, steps, chunks, args.static_threshold)
        summaries.append(summary)
        if not args.skip_plots:
            _plot_episode(episode_plot_dir / f"{episode_dir.name}.png", episode_dir.name, steps, chunks)
        print(
            f"{episode_dir.name}: arm_mean={summary['arm_mean_abs']:.4f} "
            f"arm_p10={summary['arm_p10_chunk_mean_abs']:.4f} "
            f"low_frac={summary['arm_low_motion_chunk_fraction']:.3f} "
            f"low_run_max={summary['arm_low_motion_run_max_chunks']} "
            f"j0={summary['joint_mean_abs']['joint_0']:.4f} "
            f"j1={summary['joint_mean_abs']['joint_1']:.4f} "
            f"j2={summary['joint_mean_abs']['joint_2']:.4f} "
            f"j3={summary['joint_mean_abs']['joint_3']:.4f} "
            f"j4={summary['joint_mean_abs']['joint_4']:.4f} "
            f"grip={summary['joint_mean_abs']['gripper']:.4f}"
        )

    summaries.sort(key=lambda s: s["episode"])
    if not args.skip_plots:
        _plot_overview(out_dir / "overview.png", summaries)
        _plot_low_motion_overview(out_dir / "arm_low_motion_overview.png", summaries, args.static_threshold)
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    print()
    print(f"wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
