#!/usr/bin/env python3
"""Analyze where j2/elbow action supervision comes from in extracted demos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[3]
TASK_DIR = (
    REPO
    / "models/rynnvla-002/training_pipeline/training_data/Put_the_screwdriver_into_the_cup"
)
OUTPUT_DIR = REPO / "eval_output/screwdriver_so101/data_analysis/j2_action_signal"
JOINT_NAMES = ["j0_shoulder_pan", "j1_shoulder_lift", "j2_elbow_flex", "j3_wrist_flex", "j4_wrist_roll", "j5_gripper"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, default=TASK_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--motion-threshold-deg", type=float, default=2.0)
    parser.add_argument("--pause-threshold-deg", type=float, default=0.25)
    parser.add_argument("--large-j2-threshold-deg", type=float, default=2.0)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=30)
    return parser.parse_args()


def _sorted_numbered_files(path: Path, prefix: str) -> list[Path]:
    return sorted(path.glob(f"{prefix}_*.npy"), key=lambda p: int(p.stem.split("_")[-1]))


def _sorted_action_dirs(path: Path) -> list[Path]:
    return sorted(path.glob("action_*"), key=lambda p: int(p.name.split("_")[-1]))


def _load_states_deg(episode_dir: Path) -> np.ndarray:
    files = _sorted_numbered_files(episode_dir / "state", "state")
    states = np.asarray([np.load(path).astype(np.float32)[:6] for path in files], dtype=np.float32)
    # Extracted RynnVLA/OpenPI data is in radians. Keep the fallback harmless if
    # a future dataset is already in degrees.
    if np.nanmax(np.abs(states[:, :5])) < 7.0:
        states[:, :5] = np.rad2deg(states[:, :5])
    return states


def _load_actions_deg(episode_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    action_dirs = _sorted_action_dirs(episode_dir / "abs_action")
    steps = np.asarray([int(path.name.split("_")[-1]) for path in action_dirs], dtype=np.int32)
    chunks = []
    for action_dir in action_dirs:
        files = sorted(action_dir.glob("*.npy"), key=lambda p: int(p.stem))
        chunks.append(np.asarray([np.load(path).astype(np.float32) for path in files], dtype=np.float32))
    actions = np.asarray(chunks, dtype=np.float32)
    if np.nanmax(np.abs(actions[..., :5])) < 7.0:
        actions[..., :5] = np.rad2deg(actions[..., :5])
    return steps, actions


def _first_motion_frame(states: np.ndarray, threshold_deg: float) -> int | None:
    origin = states[0, :5]
    deltas = np.abs(states[:, :5] - origin[None, :])
    hits = np.where(np.max(deltas, axis=1) >= threshold_deg)[0]
    return int(hits[0]) if len(hits) else None


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return None
    aa = a[mask]
    bb = b[mask]
    if float(np.std(aa)) < 1e-8 or float(np.std(bb)) < 1e-8:
        return None
    return float(np.corrcoef(aa, bb)[0, 1])


def _safe_slope(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return None
    xx = x[mask]
    yy = y[mask]
    denom = float(np.dot(xx, xx))
    if denom < 1e-8:
        return None
    return float(np.dot(xx, yy) / denom)


def _realized_delta(states: np.ndarray, step: int, horizon_idx: int, joint: int) -> float | None:
    target_step = step + horizon_idx + 1
    if step >= len(states) or target_step >= len(states):
        return None
    return float(states[target_step, joint] - states[step, joint])


def _segment(start: int, first_motion: int | None, n_states: int) -> str:
    if first_motion is not None and start < first_motion:
        return "opening_hold"
    if start >= max(0, n_states - 90):
        return "tail"
    return "active_or_mid"


def _episode_rows(
    episode_dir: Path,
    motion_threshold_deg: float,
    pause_threshold_deg: float,
    large_j2_threshold_deg: float,
    window_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    states = _load_states_deg(episode_dir)
    steps, actions = _load_actions_deg(episode_dir)
    first_motion = _first_motion_frame(states, motion_threshold_deg)
    n_steps, chunk_len, _ = actions.shape

    rows = []
    flat_action_j2 = []
    flat_realized_j2 = []
    flat_shoulder = []
    flat_state_j2 = []
    large_count = 0
    large_poor_realized = 0
    sign_mismatch = 0
    near_pause_large = 0
    pos_count = 0
    neg_count = 0

    state_step_delta = np.diff(states[:, :5], axis=0, prepend=states[:1, :5])
    arm_speed = np.linalg.norm(state_step_delta[:, :5], axis=1)

    for local_idx, step in enumerate(steps):
        if step >= len(states):
            continue
        for h in range(chunk_len):
            action = float(actions[local_idx, h, 2])
            realized = _realized_delta(states, int(step), h, 2)
            if realized is None:
                continue
            flat_action_j2.append(action)
            flat_realized_j2.append(realized)
            flat_shoulder.append(float(states[step, 1]))
            flat_state_j2.append(float(states[step, 2]))
            if abs(action) >= large_j2_threshold_deg:
                large_count += 1
                if abs(realized) < pause_threshold_deg or abs(realized) < 0.25 * abs(action):
                    large_poor_realized += 1
                if np.sign(action) != 0 and np.sign(realized) != 0 and np.sign(action) != np.sign(realized):
                    sign_mismatch += 1
                if arm_speed[min(step, len(arm_speed) - 1)] < pause_threshold_deg:
                    near_pause_large += 1
                if action > 0:
                    pos_count += 1
                elif action < 0:
                    neg_count += 1

    action_arr = np.asarray(flat_action_j2, dtype=np.float32)
    realized_arr = np.asarray(flat_realized_j2, dtype=np.float32)
    shoulder_arr = np.asarray(flat_shoulder, dtype=np.float32)
    state_j2_arr = np.asarray(flat_state_j2, dtype=np.float32)

    abs_j2_by_step = np.mean(np.abs(actions[:, :, 2]), axis=1)
    for start_idx in range(0, max(0, n_steps - window_size + 1)):
        stop_idx = start_idx + window_size
        step_start = int(steps[start_idx])
        step_end = int(steps[stop_idx - 1])
        window_actions = actions[start_idx:stop_idx, :, 2].reshape(-1)
        realized = []
        for local_idx in range(start_idx, stop_idx):
            step = int(steps[local_idx])
            for h in range(chunk_len):
                delta = _realized_delta(states, step, h, 2)
                if delta is not None:
                    realized.append(delta)
        realized_window = np.asarray(realized, dtype=np.float32)
        rows.append(
            {
                "episode": episode_dir.name,
                "start": step_start,
                "end": step_end,
                "segment": _segment(step_start, first_motion, len(states)),
                "score_mean_abs_j2_action": float(abs_j2_by_step[start_idx:stop_idx].mean()),
                "mean_j2_action": float(window_actions.mean()),
                "mean_abs_j2_realized": float(np.mean(np.abs(realized_window))) if len(realized_window) else None,
                "realized_to_action_abs_ratio": (
                    float(np.mean(np.abs(realized_window)) / (np.mean(np.abs(window_actions)) + 1e-8))
                    if len(realized_window)
                    else None
                ),
                "j2_action_sign_balance_pos_frac": float(np.mean(window_actions > 0.0)),
                "mean_arm_speed": float(arm_speed[step_start : min(step_end + 1, len(arm_speed))].mean()),
            }
        )

    summary = {
        "episode": episode_dir.name,
        "frames": int(len(states)),
        "action_steps": int(n_steps),
        "chunk_len": int(chunk_len),
        "first_motion_frame": first_motion,
        "j2_action_abs_mean": float(np.mean(np.abs(action_arr))),
        "j2_action_abs_p50": float(np.percentile(np.abs(action_arr), 50)),
        "j2_action_abs_p90": float(np.percentile(np.abs(action_arr), 90)),
        "j2_action_abs_p99": float(np.percentile(np.abs(action_arr), 99)),
        "j2_realized_abs_mean": float(np.mean(np.abs(realized_arr))),
        "realized_vs_action_corr": _safe_corr(action_arr, realized_arr),
        "realized_vs_action_slope": _safe_slope(action_arr, realized_arr),
        "action_vs_shoulder_corr": _safe_corr(action_arr, shoulder_arr),
        "action_vs_j2_state_corr": _safe_corr(action_arr, state_j2_arr),
        "large_j2_count": int(large_count),
        "large_j2_poor_realized_count": int(large_poor_realized),
        "large_j2_poor_realized_frac": float(large_poor_realized / large_count) if large_count else None,
        "large_j2_sign_mismatch_count": int(sign_mismatch),
        "large_j2_near_pause_count": int(near_pause_large),
        "large_j2_near_pause_frac": float(near_pause_large / large_count) if large_count else None,
        "large_j2_positive_count": int(pos_count),
        "large_j2_negative_count": int(neg_count),
    }
    return summary, rows


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    agg = payload["aggregate"]
    lines = [
        "# J2 Action Signal Analysis",
        "",
        f"Task dir: `{payload['task_dir']}`",
        "",
        "## Aggregate",
        "",
        f"- episodes: {agg['episode_count']}",
        f"- action samples: {agg['sample_count']}",
        f"- j2 |action| mean / p90 / p99: {agg['j2_action_abs_mean']:.3f} / {agg['j2_action_abs_p90']:.3f} / {agg['j2_action_abs_p99']:.3f} deg",
        f"- j2 |realized| mean: {agg['j2_realized_abs_mean']:.3f} deg",
        f"- realized-vs-action corr / slope: {agg['realized_vs_action_corr']:.3f} / {agg['realized_vs_action_slope']:.3f}",
        f"- action-vs-shoulder corr: {agg['action_vs_shoulder_corr']:.3f}",
        f"- action-vs-j2-state corr: {agg['action_vs_j2_state_corr']:.3f}",
        f"- large j2 samples with poor realized motion: {agg['large_j2_poor_realized_frac']:.3f}",
        f"- large j2 samples during near-pauses: {agg['large_j2_near_pause_frac']:.3f}",
        "",
        "## Top J2 Supervision Windows",
        "",
        "| rank | episode | frames | segment | mean abs j2 action | mean abs j2 realized | realized/action | pos frac | arm speed |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(payload["top_windows"], start=1):
        lines.append(
            "| "
            f"{idx} | {row['episode']} | {row['start']}-{row['end']} | {row['segment']} | "
            f"{row['score_mean_abs_j2_action']:.3f} | "
            f"{row['mean_abs_j2_realized']:.3f} | "
            f"{row['realized_to_action_abs_ratio']:.3f} | "
            f"{row['j2_action_sign_balance_pos_frac']:.3f} | "
            f"{row['mean_arm_speed']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Episodes With Weakest Realized J2 Response",
            "",
            "| episode | first motion | j2 action mean | j2 realized mean | corr | slope | poor large frac | near-pause large frac |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["weak_response_episodes"]:
        lines.append(
            "| "
            f"{row['episode']} | {row['first_motion_frame']} | "
            f"{row['j2_action_abs_mean']:.3f} | {row['j2_realized_abs_mean']:.3f} | "
            f"{row['realized_vs_action_corr']:.3f} | {row['realized_vs_action_slope']:.3f} | "
            f"{row['large_j2_poor_realized_frac']:.3f} | {row['large_j2_near_pause_frac']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    task_dir = args.task_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_dirs = sorted(path for path in task_dir.glob("episode_*") if path.is_dir())
    episode_summaries = []
    all_windows = []
    all_actions = []
    all_realized = []
    all_shoulder = []
    all_state_j2 = []
    large_count = 0
    large_poor = 0
    large_pause = 0

    for episode_dir in episode_dirs:
        summary, windows = _episode_rows(
            episode_dir,
            args.motion_threshold_deg,
            args.pause_threshold_deg,
            args.large_j2_threshold_deg,
            args.window_size,
        )
        episode_summaries.append(summary)
        all_windows.extend(windows)

        states = _load_states_deg(episode_dir)
        steps, actions = _load_actions_deg(episode_dir)
        for local_idx, step in enumerate(steps):
            for h in range(actions.shape[1]):
                realized = _realized_delta(states, int(step), h, 2)
                if realized is None:
                    continue
                action = float(actions[local_idx, h, 2])
                all_actions.append(action)
                all_realized.append(realized)
                all_shoulder.append(float(states[min(int(step), len(states) - 1), 1]))
                all_state_j2.append(float(states[min(int(step), len(states) - 1), 2]))
                if abs(action) >= args.large_j2_threshold_deg:
                    large_count += 1
                    if abs(realized) < args.pause_threshold_deg or abs(realized) < 0.25 * abs(action):
                        large_poor += 1
                    state_delta = np.diff(states[:, :5], axis=0, prepend=states[:1, :5])
                    arm_speed = np.linalg.norm(state_delta[:, :5], axis=1)
                    if arm_speed[min(int(step), len(arm_speed) - 1)] < args.pause_threshold_deg:
                        large_pause += 1

    action_arr = np.asarray(all_actions, dtype=np.float32)
    realized_arr = np.asarray(all_realized, dtype=np.float32)
    shoulder_arr = np.asarray(all_shoulder, dtype=np.float32)
    state_j2_arr = np.asarray(all_state_j2, dtype=np.float32)
    all_windows.sort(key=lambda row: row["score_mean_abs_j2_action"], reverse=True)
    weak_response = sorted(
        [row for row in episode_summaries if row["large_j2_count"] > 0],
        key=lambda row: (row["realized_vs_action_slope"] if row["realized_vs_action_slope"] is not None else 999),
    )[: args.top_k]

    payload = {
        "task_dir": str(task_dir),
        "joint_names": JOINT_NAMES,
        "parameters": {
            "motion_threshold_deg": args.motion_threshold_deg,
            "pause_threshold_deg": args.pause_threshold_deg,
            "large_j2_threshold_deg": args.large_j2_threshold_deg,
            "window_size": args.window_size,
        },
        "aggregate": {
            "episode_count": len(episode_summaries),
            "sample_count": int(len(action_arr)),
            "j2_action_abs_mean": float(np.mean(np.abs(action_arr))),
            "j2_action_abs_p50": float(np.percentile(np.abs(action_arr), 50)),
            "j2_action_abs_p90": float(np.percentile(np.abs(action_arr), 90)),
            "j2_action_abs_p99": float(np.percentile(np.abs(action_arr), 99)),
            "j2_realized_abs_mean": float(np.mean(np.abs(realized_arr))),
            "realized_vs_action_corr": _safe_corr(action_arr, realized_arr),
            "realized_vs_action_slope": _safe_slope(action_arr, realized_arr),
            "action_vs_shoulder_corr": _safe_corr(action_arr, shoulder_arr),
            "action_vs_j2_state_corr": _safe_corr(action_arr, state_j2_arr),
            "large_j2_count": int(large_count),
            "large_j2_poor_realized_frac": float(large_poor / large_count) if large_count else None,
            "large_j2_near_pause_frac": float(large_pause / large_count) if large_count else None,
        },
        "episode_summaries": episode_summaries,
        "top_windows": all_windows[: args.top_k],
        "weak_response_episodes": weak_response,
    }

    json_path = output_dir / "j2_action_signal_summary.json"
    md_path = output_dir / "j2_action_signal_summary.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(md_path, payload)

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    agg = payload["aggregate"]
    print(
        "aggregate "
        f"samples={agg['sample_count']} "
        f"action_abs_mean={agg['j2_action_abs_mean']:.3f} "
        f"realized_abs_mean={agg['j2_realized_abs_mean']:.3f} "
        f"corr={agg['realized_vs_action_corr']:.3f} "
        f"slope={agg['realized_vs_action_slope']:.3f} "
        f"poor_large_frac={agg['large_j2_poor_realized_frac']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
