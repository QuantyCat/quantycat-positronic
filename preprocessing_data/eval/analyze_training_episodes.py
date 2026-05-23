#!/usr/bin/env python3
"""Comprehensive training episode analysis for the screwdriver-into-cup dataset.

Covers:
  - Episode length distribution
  - Hold (countdown) duration per episode
  - Gripper timing: grasp frame, release frame, hold duration
  - Episode completion: where does the end-effector finish?
  - End-effector trajectory variance across episodes
  - Action smoothness / jitter per joint
  - State-action tracking lag (servo performance)
  - Outlier episodes

Output: terminal report + JSON + per-episode markdown table
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATASET_ROOT = REPO / "my_data/input_data"
OUTPUT_DIR = REPO / "eval_output/screwdriver_so101/data_analysis/episode_deep_audit"

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# Gripper: ~0 = closed, max ~0.489 rad = open. Threshold for "grasping":
GRIPPER_CLOSED_THRESHOLD = 0.08   # rad; below this = closed
GRIPPER_OPEN_THRESHOLD   = 0.25   # rad; above this = open

# Shoulder_lift delta threshold for "first real motion" (radians, ~2°)
FIRST_MOTION_LIFT_THRESHOLD_RAD = np.deg2rad(2.0)


def load_episode(path: Path):
    df = pd.read_parquet(path)
    states  = np.stack(df["observation.state"].apply(np.asarray).values).astype(np.float32)
    actions = np.stack(df["action"].apply(np.asarray).values).astype(np.float32)
    obs_ee  = np.stack(df["observation.end_effector_pose"].apply(np.asarray).values).astype(np.float32)
    act_ee  = np.stack(df["action.end_effector_pose"].apply(np.asarray).values).astype(np.float32)
    ts      = df["timestamp"].values.astype(np.float32)
    return states, actions, obs_ee, act_ee, ts


def first_motion_frame(states, actions, threshold=FIRST_MOTION_LIFT_THRESHOLD_RAD):
    """First frame where |action_lift - state_lift| > threshold."""
    lift_delta = np.abs(actions[:, 1] - states[:, 1])
    frames = np.where(lift_delta > threshold)[0]
    return int(frames[0]) if len(frames) > 0 else -1


def gripper_events(actions):
    """Return (grasp_frame, release_frame) based on gripper action trajectory.
    grasp_frame: first frame where gripper transitions from open to closed.
    release_frame: first frame after grasp where gripper transitions back to open (-1 if never).
    """
    gripper = actions[:, 5]
    n = len(gripper)

    # Find grasp: first time gripper drops below CLOSED threshold after being open
    grasp_frame = -1
    for i in range(1, n):
        if gripper[i - 1] > GRIPPER_OPEN_THRESHOLD and gripper[i] < GRIPPER_CLOSED_THRESHOLD:
            grasp_frame = i
            break

    # If no clear open->close transition, find first sustained closed period
    if grasp_frame == -1:
        closed = gripper < GRIPPER_CLOSED_THRESHOLD
        for i in range(n):
            if closed[i] and (i == 0 or not closed[i - 1]):
                # Check this is sustained (not a flicker)
                if i + 5 < n and np.all(closed[i:i + 5]):
                    grasp_frame = i
                    break

    # Find release: first time gripper opens after grasp
    release_frame = -1
    if grasp_frame >= 0:
        for i in range(grasp_frame + 1, n):
            if actions[i - 1, 5] < GRIPPER_CLOSED_THRESHOLD and actions[i, 5] > GRIPPER_OPEN_THRESHOLD:
                release_frame = i
                break

    return grasp_frame, release_frame


def action_jitter(actions):
    """Mean absolute second difference of each joint's action trajectory (smoothness measure).
    High = jerky, low = smooth.
    """
    if len(actions) < 3:
        return np.zeros(6)
    second_diff = np.diff(actions[:, :5], n=2, axis=0)  # exclude gripper from smoothness
    return np.mean(np.abs(second_diff), axis=0)


def state_action_lag(states, actions):
    """Mean absolute difference between commanded action and observed state per joint.
    High for a joint = servo couldn't keep up.
    """
    err = np.abs(actions - states)
    return np.mean(err, axis=0)


def analyze_episode(path: Path) -> dict:
    ep_idx = int(path.stem.split("_")[1])
    states, actions, obs_ee, act_ee, ts = load_episode(path)
    n = len(states)
    duration = float(ts[-1] - ts[0]) if n > 1 else 0.0

    hold_frame = first_motion_frame(states, actions)
    hold_s = float(ts[hold_frame]) if hold_frame >= 0 else float("nan")
    active_frames = n - hold_frame if hold_frame >= 0 else n

    grasp_frame, release_frame = gripper_events(actions)
    grasp_s = float(ts[grasp_frame]) if grasp_frame >= 0 else float("nan")
    release_s = float(ts[release_frame]) if release_frame >= 0 else float("nan")

    # Gripper range actually used
    gripper_min = float(actions[:, 5].min())
    gripper_max = float(actions[:, 5].max())
    gripper_range = gripper_max - gripper_min

    # End-effector position at start, grasp, and end of episode
    ee_start = obs_ee[0, :3].tolist()
    ee_end   = obs_ee[-1, :3].tolist()
    ee_grasp = obs_ee[grasp_frame, :3].tolist() if grasp_frame >= 0 else [float("nan")] * 3

    # Trajectory smoothness (arm joints 0-4 only)
    jitter = action_jitter(actions)

    # State-action tracking
    lag = state_action_lag(states, actions)

    # Peak motion per joint (max |action - state|)
    peak_lag = np.max(np.abs(actions - states), axis=0)

    # Episode task phases relative timing
    frames_to_grasp = (grasp_frame - hold_frame) if (grasp_frame >= 0 and hold_frame >= 0) else -1
    frames_grasp_to_release = (release_frame - grasp_frame) if (grasp_frame >= 0 and release_frame >= 0) else -1
    frames_active_after_grasp = (n - grasp_frame) if grasp_frame >= 0 else -1

    # How much wrist_roll varies (joint 4) — often a source of noise
    wrist_roll_std = float(np.std(actions[:, 4]))
    wrist_roll_range = float(actions[:, 4].max() - actions[:, 4].min())

    # Action variance per joint (useful for checking multi-modal behavior)
    action_std = actions.std(axis=0)

    return {
        "episode": ep_idx,
        "n_frames": n,
        "duration_s": round(duration, 2),
        "active_frames": active_frames,
        "hold_frame": hold_frame,
        "hold_s": round(hold_s, 2) if not np.isnan(hold_s) else None,
        "grasp_frame": grasp_frame,
        "grasp_s": round(grasp_s, 2) if not np.isnan(grasp_s) else None,
        "release_frame": release_frame,
        "release_s": round(release_s, 2) if not np.isnan(release_s) else None,
        "frames_reach_to_grasp": frames_to_grasp,
        "frames_held": frames_grasp_to_release,
        "gripper_range_rad": round(gripper_range, 4),
        "gripper_min_rad": round(gripper_min, 4),
        "gripper_max_rad": round(gripper_max, 4),
        "ee_start_xyz": [round(v, 4) for v in ee_start],
        "ee_grasp_xyz": [round(v, 4) for v in ee_grasp],
        "ee_end_xyz": [round(v, 4) for v in ee_end],
        "jitter_joints04_mean_rad": [round(float(v), 6) for v in jitter],
        "tracking_lag_mean_rad": [round(float(v), 6) for v in lag],
        "tracking_lag_peak_rad": [round(float(v), 6) for v in peak_lag],
        "wrist_roll_std_rad": round(wrist_roll_std, 5),
        "wrist_roll_range_rad": round(wrist_roll_range, 4),
        "action_std_per_joint": [round(float(v), 5) for v in action_std],
    }


def print_report(episodes: list[dict]):
    n = len(episodes)
    print(f"\n{'='*80}")
    print(f"TRAINING EPISODE DEEP AUDIT  ({n} episodes)")
    print(f"{'='*80}\n")

    # --- Episode lengths ---
    lengths = [e["n_frames"] for e in episodes]
    durations = [e["duration_s"] for e in episodes]
    print("EPISODE LENGTHS")
    print(f"  frames:   min={min(lengths)}  max={max(lengths)}  mean={np.mean(lengths):.0f}  std={np.std(lengths):.0f}")
    print(f"  duration: min={min(durations):.1f}s  max={max(durations):.1f}s  mean={np.mean(durations):.1f}s  std={np.std(durations):.1f}s")
    outlier_len = [e["episode"] for e in episodes if abs(e["n_frames"] - np.mean(lengths)) > 2 * np.std(lengths)]
    if outlier_len:
        print(f"  LENGTH OUTLIERS (>2σ): episodes {outlier_len}")
    print()

    # --- Hold / countdown ---
    holds = [e["hold_frame"] for e in episodes if e["hold_frame"] >= 0]
    hold_ss = [e["hold_s"] for e in episodes if e["hold_s"] is not None]
    print("COUNTDOWN HOLD")
    print(f"  frames: min={min(holds)}  max={max(holds)}  mean={np.mean(holds):.0f}  std={np.std(holds):.0f}")
    print(f"  time:   min={min(hold_ss):.2f}s  max={max(hold_ss):.2f}s  mean={np.mean(hold_ss):.2f}s")
    print()

    # --- Gripper / grasp timing ---
    grasp_frames = [e["grasp_frame"] for e in episodes if e["grasp_frame"] >= 0]
    grasp_ss = [e["grasp_s"] for e in episodes if e["grasp_s"] is not None]
    no_grasp = [e["episode"] for e in episodes if e["grasp_frame"] < 0]
    print("GRIPPER / GRASP TIMING")
    if no_grasp:
        print(f"  WARNING: {len(no_grasp)} episodes with no detected grasp: {no_grasp}")
    if grasp_frames:
        print(f"  grasp frame: min={min(grasp_frames)}  max={max(grasp_frames)}  mean={np.mean(grasp_frames):.0f}  std={np.std(grasp_frames):.0f}")
        print(f"  grasp time:  min={min(grasp_ss):.2f}s  max={max(grasp_ss):.2f}s  mean={np.mean(grasp_ss):.2f}s  std={np.std(grasp_ss):.2f}s")

    release_frames = [e["release_frame"] for e in episodes if e["release_frame"] >= 0]
    no_release = [e["episode"] for e in episodes if e["release_frame"] < 0]
    if no_release:
        print(f"  NOTE: {len(no_release)} episodes with no detected release (gripper stays closed): {no_release[:10]}{'...' if len(no_release) > 10 else ''}")
    if release_frames:
        release_ss = [e["release_s"] for e in episodes if e["release_s"] is not None]
        print(f"  release frame: min={min(release_frames)}  max={max(release_frames)}  mean={np.mean(release_frames):.0f}  std={np.std(release_frames):.0f}")
        print(f"  release time:  min={min(release_ss):.2f}s  max={max(release_ss):.2f}s  mean={np.mean(release_ss):.2f}s")

    reach_times = [e["frames_reach_to_grasp"] for e in episodes if e["frames_reach_to_grasp"] >= 0]
    if reach_times:
        print(f"  frames hold→grasp: min={min(reach_times)}  max={max(reach_times)}  mean={np.mean(reach_times):.0f}  std={np.std(reach_times):.0f}")

    gripper_ranges = [e["gripper_range_rad"] for e in episodes]
    print(f"  gripper range used: min={min(gripper_ranges):.3f}  max={max(gripper_ranges):.3f}  mean={np.mean(gripper_ranges):.3f} rad")
    low_gripper = [e["episode"] for e in episodes if e["gripper_range_rad"] < 0.1]
    if low_gripper:
        print(f"  WARNING: episodes with very low gripper range (<0.1 rad, may be failed grasp): {low_gripper}")
    print()

    # --- End-effector at episode end ---
    ee_ends = np.array([e["ee_end_xyz"] for e in episodes])
    print("END-EFFECTOR POSITION AT EPISODE END (xyz, meters)")
    print(f"  x: mean={ee_ends[:,0].mean():.4f}  std={ee_ends[:,0].std():.4f}  range=[{ee_ends[:,0].min():.4f}, {ee_ends[:,0].max():.4f}]")
    print(f"  y: mean={ee_ends[:,1].mean():.4f}  std={ee_ends[:,1].std():.4f}  range=[{ee_ends[:,1].min():.4f}, {ee_ends[:,1].max():.4f}]")
    print(f"  z: mean={ee_ends[:,2].mean():.4f}  std={ee_ends[:,2].std():.4f}  range=[{ee_ends[:,2].min():.4f}, {ee_ends[:,2].max():.4f}]")
    end_spread = np.linalg.norm(ee_ends - ee_ends.mean(axis=0), axis=1)
    outlier_end = [episodes[i]["episode"] for i in range(n) if end_spread[i] > 0.1]
    if outlier_end:
        print(f"  END POSITION OUTLIERS (>10cm from mean): episodes {outlier_end}")
    print()

    # --- End-effector at grasp ---
    ee_grasps = np.array([e["ee_grasp_xyz"] for e in episodes if e["grasp_frame"] >= 0 and not np.isnan(e["ee_grasp_xyz"][0])])
    if len(ee_grasps) > 0:
        print("END-EFFECTOR POSITION AT GRASP (xyz, meters)")
        print(f"  x: mean={ee_grasps[:,0].mean():.4f}  std={ee_grasps[:,0].std():.4f}  range=[{ee_grasps[:,0].min():.4f}, {ee_grasps[:,0].max():.4f}]")
        print(f"  y: mean={ee_grasps[:,1].mean():.4f}  std={ee_grasps[:,1].std():.4f}  range=[{ee_grasps[:,1].min():.4f}, {ee_grasps[:,1].max():.4f}]")
        print(f"  z: mean={ee_grasps[:,2].mean():.4f}  std={ee_grasps[:,2].std():.4f}  range=[{ee_grasps[:,2].min():.4f}, {ee_grasps[:,2].max():.4f}]")
        grasp_spread = np.linalg.norm(ee_grasps - ee_grasps.mean(axis=0), axis=1)
        high_var = np.percentile(grasp_spread, 75)
        print(f"  grasp position spread 75th pct: {high_var:.4f}m  max: {grasp_spread.max():.4f}m")
        print()

    # --- Action smoothness ---
    all_jitter = np.array([e["jitter_joints04_mean_rad"] for e in episodes])
    print("ACTION SMOOTHNESS (mean |Δ²action| per joint, lower=smoother)")
    print(f"  {'Joint':<14} {'mean':>8} {'max':>8} {'std':>8}")
    for j, name in enumerate(JOINT_NAMES[:5]):
        col = all_jitter[:, j]
        print(f"  {name:<14} {col.mean():>8.5f} {col.max():>8.5f} {col.std():>8.5f}")
    jitter_totals = all_jitter.mean(axis=1)
    jittery = [episodes[i]["episode"] for i in np.argsort(jitter_totals)[-5:]]
    print(f"  5 jitteriest episodes (by mean across joints): {jittery}")
    print()

    # --- State-action tracking lag ---
    all_lag = np.array([e["tracking_lag_mean_rad"] for e in episodes])
    all_peak_lag = np.array([e["tracking_lag_peak_rad"] for e in episodes])
    print("STATE-ACTION TRACKING LAG (mean |action - state| per joint, radians)")
    print(f"  {'Joint':<14} {'mean':>8} {'max_ep':>8} {'peak_rad':>10}")
    for j, name in enumerate(JOINT_NAMES):
        col = all_lag[:, j]
        peak_col = all_peak_lag[:, j]
        print(f"  {name:<14} {col.mean():>8.4f} {col.max():>8.4f} {peak_col.max():>10.4f}")
    high_lag_eps = [episodes[i]["episode"] for i in range(n) if all_lag[i].mean() > np.percentile(all_lag.mean(axis=1), 90)]
    print(f"  Top 10% highest tracking lag episodes: {high_lag_eps}")
    print()

    # --- Wrist roll ---
    wr_std = [e["wrist_roll_std_rad"] for e in episodes]
    wr_range = [e["wrist_roll_range_rad"] for e in episodes]
    print("WRIST ROLL (joint 4) VARIABILITY")
    print(f"  std: mean={np.mean(wr_std):.4f}  max={max(wr_std):.4f}  (per-episode std of wrist_roll action)")
    print(f"  range: mean={np.mean(wr_range):.4f}rad={np.rad2deg(np.mean(wr_range)):.1f}°  max={max(wr_range):.4f}rad={np.rad2deg(max(wr_range)):.1f}°")
    high_wr = [episodes[i]["episode"] for i in range(n) if wr_range[i] > np.deg2rad(20)]
    if high_wr:
        print(f"  Episodes with >20° wrist_roll range: {high_wr}")
    print()

    # --- Per-episode table ---
    print("PER-EPISODE SUMMARY TABLE")
    header = f"{'ep':>3}  {'frames':>6}  {'hold_fr':>7}  {'grasp_fr':>8}  {'rel_fr':>6}  {'grip_rng':>8}  {'ee_end_z':>8}  {'jitter':>7}  {'lag':>7}"
    print(header)
    print("-" * len(header))
    for e in episodes:
        grasp_fr = str(e["grasp_frame"]) if e["grasp_frame"] >= 0 else "none"
        rel_fr   = str(e["release_frame"]) if e["release_frame"] >= 0 else "-"
        jit_mean = np.mean(e["jitter_joints04_mean_rad"]) if e["jitter_joints04_mean_rad"] else 0
        lag_mean = np.mean(e["tracking_lag_mean_rad"][:5]) if e["tracking_lag_mean_rad"] else 0
        ee_end_z = e["ee_end_xyz"][2]
        print(f"{e['episode']:>3}  {e['n_frames']:>6}  {e['hold_frame']:>7}  {grasp_fr:>8}  {rel_fr:>6}  {e['gripper_range_rad']:>8.4f}  {ee_end_z:>8.4f}  {jit_mean:>7.5f}  {lag_mean:>7.4f}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-save", action="store_true", help="Don't save JSON output")
    args = parser.parse_args()

    data_dir = args.dataset_root / "data/chunk-000"
    episodes_paths = sorted(data_dir.glob("episode_*.parquet"))
    print(f"Analyzing {len(episodes_paths)} episodes...", file=sys.stderr)

    results = []
    for path in episodes_paths:
        ep_idx = int(path.stem.split("_")[1])
        print(f"  ep{ep_idx:03d}...", end="\r", file=sys.stderr)
        results.append(analyze_episode(path))

    print(f"  Done.          ", file=sys.stderr)

    print_report(results)

    if not args.no_save:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / "report.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON report to {out_path}")


if __name__ == "__main__":
    main()
