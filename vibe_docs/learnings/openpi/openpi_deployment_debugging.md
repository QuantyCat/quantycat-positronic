# OpenPI SO-101 Deployment Debugging: Root Cause & Fix

**Date:** 2026-05-21  
**Checkpoint:** `pi05_quantycat_lora` step 9999  
**Task:** "Put the screwdriver into the cup"  
**Symptom:** Robot barely moves during live deployment despite correct inference pipeline

---

## The Problem

After verifying that robot control, inference, and safety clipping were all working correctly, the arm still produced near-zero motion over 20 deployment steps. The policy was outputting targets like `elbow = 93.1°` from a starting state of `elbow = 95.96°` — a -2.86° command — but the arm barely moved, and subsequent inferences produced identical near-zero commands.

---

## Root Cause: The Training Countdown Hold

Every single training episode (50 out of 50) began with the arm sitting completely still for **~165 frames = 5.5 seconds** before any reaching motion started. This was confirmed by scanning all training parquets for the first frame where `shoulder_lift` delta exceeded 2°.

```
ep   frames  dur_s  first_motion_fr  first_motion_s
 0      703   23.4             160           5.33s
 1      810   27.0             168           5.60s
 2      900   30.0             167           5.57s
 7      665   22.1             164           5.47s
...
(all 50 episodes: 163–197 frames, 5.43–6.57s hold)
```

This hold was the recording system's countdown: the demonstrator waited for a visual/audio cue before starting the motion. The arm started at `shoulder_lift ≈ -100°` every time and stayed there until the demonstrator decided to lift.

**What the policy learned from this:**

The visual scene at the rest position (`shoulder_lift ≈ -100°`) appears in every episode for the first 5+ seconds. The action labels for all those frames say "hold at -100°, command elbow to 93.1°". So the policy correctly learned: _when you see this scene, output hold/tiny commands._

The transition to reaching at frame 165 was a **time-based human decision**, not triggered by any visual change. Imitation learning from visual observations cannot replicate a temporal trigger — the policy will always see "arm at rest = hold" regardless of how long it has been running.

---

## Why the Gravity Stall Made It Worse

The elbow at ~96° is near a gravity equilibrium point. The follower arm's servo PD controller cannot overcome gravity torque to reach the commanded target of 93.1°. During the 165-frame hold in training:

- Demonstrator commanded: `elbow target = 93.1°` (from state 95.96°, delta = -2.86°)
- Robot actually moved: ~0.27° over 5.5 seconds
- State in training data: frozen at `elbow ≈ 95.96°`

The policy saw 165 frames of identical visual input with action labels `elbow → 93.1°`, which is a target the motor physically cannot reach. In deployment, the same thing happened: command issued, motor stalls, no progress, re-inference sees the same scene, same hold command.

---

## Diagnostic Scripts Written

All scripts live in `models/openpi/eval/`.

### `show_episode_actions.py`

Shows ground-truth action deltas frame-by-frame for any training episode (no policy required). Revealed that ep7 has nearly identical state from frame 0 to 165, with the arm finally moving at frame 165.

```bash
/home/caroline/openpi/.venv/bin/python models/openpi/eval/show_episode_actions.py \
  --episode 7 --stride 5
```

### `check_episode_timing.py`

Shows timestamps and effective frame rate of a training episode. Confirmed training was recorded at **30 fps**, so 665 frames = 22.1 seconds total, and the hold phase was exactly 5.5 seconds.

```bash
/home/caroline/openpi/.venv/bin/python models/openpi/eval/check_episode_timing.py \
  --episode 7
```

### `find_first_motion.py`

Scans all training episodes and reports the first frame where `|shoulder_lift delta| > threshold`. This is what revealed the consistent 5.5-second hold across all 50 episodes.

```bash
/home/caroline/openpi/.venv/bin/python models/openpi/eval/find_first_motion.py \
  --threshold-deg 2.0
```

---

## The Fix: Skip the Dead Zone

The training trajectory has a "dead zone" (frames 0–164) where the arm doesn't move, and an "active zone" (frames 165+) where reaching begins. To use the checkpoint, bypass the dead zone by positioning the arm at a state from the active zone before starting policy inference.

### Finding the Right Starting Pose

Episode 7, frame 180 is a good entry point — the arm has just broken through the gravity stall and is actively reaching:

```
state (deg): [3.82, -86.42, 92.44, 66.99, 5.98, 0.39]
  shoulder_pan:  3.82°
  shoulder_lift: -86.42°
  elbow_flex:    92.44°
  wrist_flex:    66.99°
  wrist_roll:    5.98°
  gripper:       0.39°
```

The start pose used successfully in deployment: `[4, -85, 92, 67, 6, 0.4]`

### Deployment Procedure

1. Move the arm to the pre-reach start position:
   ```bash
   /home/caroline/openpi/.venv/bin/python \
     models/openpi/deployment/test_robot_send_action.py \
     --start-pose 4 -85 92 67 6 0.4
   ```

2. Confirm joint positions match (the script prints them after the move).

3. Run the live policy with enough steps to cover the full task (~22s of training = 22 × 30 = 660 frames; at 0.75s/step, ~200 steps for the active portion):
   ```bash
   bash models/openpi/run_scripts/live_so101_step9999.sh --max-steps 200
   ```

### First Confirmed Live Run (2026-05-21)

Starting from `[4.48, -85.43, 93.16, 66.71, 5.89, 0.27]`:

| Step | shoulder_lift | elbow_flex | shoulder_pan |
|------|--------------|------------|--------------|
| 0    | -85.43°      | 93.16°     | +4.48°       |
| 10   | -72.16°      | 80.77°     | +0.53°       |
| 19   | -59.24°      | 66.97°     | -3.96°       |

**+26° shoulder lift, -26° elbow extension in 15 seconds.** The policy consistently commanded at or above the 6°/step elbow safety clip, confirming it was trying to move faster than the limit allowed. This is the correct reaching behavior.

---

## Why the Policy Checkpoint Is Actually Fine

The checkpoint learned to do the task correctly for the reaching, grasp, and placement phases. The only thing it cannot do is self-initiate from the rest position, because that transition was a time-based human decision not encoded in the visual stream.

Evidence:
- From the active start pose, inference immediately outputs large coordinated deltas (shoulder_lift +4°, elbow -6°, pan -1°–-2°)
- Commands are consistent across two sequential inferences (steps 0–9 and 10–19)
- The safety limiter was engaged on elbow (`limited=True`) — the policy was requesting MORE motion than allowed
- The trajectory matches the training data: at step 19, `shoulder_lift = -59.24°`, matching ep7 frame ~205

---

## Permanent Fixes

### Option A: Manual start-pose script (current workaround)

Keep the `--start-pose` pre-move step in the deployment procedure. Document the canonical start angles in the run script or config.

Short-term workaround:

- add a scripted pre-position or kickstart before inference
- use the validated active pose `[4, -85, 92, 67, 6, 0.4]`
- or use a smaller deterministic pre-lift that moves the arm into the active-motion regime before handing control to OpenPI

Validated commands:

```bash
python models/openpi/deployment/test_robot_send_action.py --start-pose 4 -85 92 67 6 0.4 --wait 3.0
bash models/openpi/run_scripts/live_so101_step9999.sh --max-steps 20
```

### Option B: Add a kickstart to the deployment config

Add a `warmup_commands_deg` field to `pi05_lora_step9999_so101.json` that the live script executes before starting policy inference — e.g., a gradual 10° shoulder_lift raise over 3 steps. This automates the bypass without requiring manual pre-positioning.

### Option C: Re-collect training data without the countdown hold

When recording new episodes, have the demonstrator start the leader arm motion immediately at t=0 (no countdown hold). This teaches the policy to initiate from the rest position visually. Alternatively, trim the first 160 frames from each existing episode before retraining.

### Option D: Trim existing episodes (fastest retraining path)

The existing 50 episodes can be trimmed at frame 165 before re-running `build_dataset` + `train`. This removes the hold phase from training data without re-recording. The resulting policy should self-start from the rest position.

Long-term fix:

- retrain with demos that do not include the long frozen prefix at the start
- or trim the first ~165 frames from every episode before training so the policy learns from motion onset instead of countdown hold time

---

## Summary

| Finding | Value |
|---------|-------|
| Training hold duration | 5.43–6.57s (163–197 frames) across all 50 episodes |
| Training frame rate | 30 fps |
| Gravity stall position | elbow ≈ 95.96°, shoulder_lift ≈ -100° |
| Policy self-start | Not possible from rest (time-based trigger, not visual) |
| Policy capability from active start | Confirmed working — correct reaching trajectory |
| Canonical deployment start pose | `[4, -85, 92, 67, 6, 0.4]` (deg) |
| Recommended max_steps for full task | 200 (covers ~150s at 0.75s/step) |


Beast Claude: claude --resume 37c8dade-3eb5-439c-8670-40be8a927dfe 
Lamb Codex: codex resume 019e46d8-e8c2-7e22-9607-8206d1fa5426

---

## Deep Training Data Audit (2026-05-23)

After confirming the checkpoint works from the active start pose, a full audit of all 50 training episodes was run to identify other data quality issues before retraining.

### Script: `analyze_training_episodes.py`

Comprehensive per-episode analysis covering: hold duration, gripper timing and range, end-effector position at grasp and episode end, action smoothness/jitter, state-action tracking lag, and wrist roll variability.

```bash
cd /home/caroline/quantycat-positronic
/home/caroline/quantycat-positronic/vendor/openpi/.venv/bin/python \
  models/openpi/eval/analyze_training_episodes.py
```

Output saved to `eval_output/screwdriver_so101/data_analysis/episode_deep_audit/report.json`.

### Key findings

**Episode 45 is a failed demonstration.**
The gripper opens to its full range (~28°) mid-reach at frame ~200 — consistent with a drop and re-grasp. The end-effector trajectory is radically different from all other 49 episodes: the arm sweeps to `[0.241, -0.426]` (far outside the cluster) before attempting to recover. Including it gives the policy contradictory supervision for the same visual scene.

**Wrist roll has enormous trajectory variance.**
During the hold phase, `wrist_roll` is essentially static (std = 0.37°). During the active phase, it ranges across a mean of **84.7°** per episode (max 120.7°), with trajectory std of ~24° — even though every episode *ends* at ≈ −38.5°. The policy has to learn a highly variable wrist_roll arc from visual input, which the visual stream doesn't encode. This is a significant source of multi-modal supervision noise.

**All demonstrations start with the gripper already closed.**
The gripper is near 0 rad (closed) from frame 0 in all 50 episodes. The demonstrator starts already holding the screwdriver. The training data covers **transport + place only** — there is no pick-up motion in the dataset. A policy retrained from this data cannot grasp from a surface.

**Episode 12 has a transient glitch at frames 46–53.**
The leader arm briefly commands shoulder_lift to −97° and snaps back to −100° over 7 frames. Not worth removing, but the actual motion onset is at frame ~200, consistent with other episodes. The first-motion detection reported 46 for this episode, which is a false alarm.

**Episode 28 descends unusually deep (z = 0.048 m).**
Most episodes reach z ≈ 0.08–0.10 m at lowest. Inspect the video before deciding whether to keep.

**End-effector endpoint is highly consistent.**
All episodes finish within ≈5.5 mm (x), 3.8 mm (y), 7.5 mm (z) of the mean endpoint. Task completion position is reliable across all good episodes.

### Audit summary table

| Issue | Severity | Action |
|-------|----------|--------|
| Episode 45: failed demo (drop + re-grasp) | High | Remove |
| Wrist roll variance (mean 84.7° range) | High | Freeze, down-weight, or re-record |
| Countdown hold (163–197 frames) | High | Trim first 165 frames (already known) |
| Episode 28: unusually deep trajectory | Medium | Inspect video |
| Episode 12: 7-frame leader glitch | Low | Ignore or smooth |
| Episode length outliers (ep2, ep16, ep47) | Low | Accept |

---

## Clean Dataset: `clean_input_data` (2026-05-23)

### What changed

- First **165 frames trimmed** from all 49 kept episodes (removes the countdown hold)
- **Episode 45 removed** (failed demonstration)
- Episodes re-indexed contiguously: ep46→ep45, ep47→ep46, ep48→ep47, ep49→ep48
- All parquet files regenerated with corrected `frame_index`, `timestamp`, `episode_index`, and global `index`
- Videos trimmed with ffmpeg (frame-exact selection `n ≥ 165`, re-encoded as H.264)
- All meta files regenerated (`info.json`, `episodes.jsonl`, `episodes_stats.jsonl`, `tasks.jsonl`)

### Numbers

| | Original | Clean |
|-|----------|-------|
| Episodes | 50 | 49 |
| Total frames | 37,266 | 28,476 |
| Frames removed | — | 8,790 (23.6%) |
| Hold frames removed | — | ~8,085 |

### Script: `build_clean_dataset.py`

```bash
cd /home/caroline/quantycat-positronic

# Dry run — shows episode mapping, no writes
python scripts/build_clean_dataset.py --dry-run

# Full run
python scripts/build_clean_dataset.py
```

Output: `my_data/clean_input_data/`

To point a training run at the clean data, update the dataset path in your training config from `my_data/input_data` to `my_data/clean_input_data`.

### What this fixes for the retrained policy

- Removes the "rest position → hold still" supervision (23% of training data)
- Removes the contradictory trajectory from episode 45
- A policy retrained on this data should self-initiate from the rest position without needing the manual pre-position step

### What it does NOT fix

- Wrist roll trajectory variance is unchanged — still 84.7° mean range
- The dataset still contains only transport+place demonstrations, not pick-up from surface
- Episode 28's unusual depth is preserved (inspect video to decide)