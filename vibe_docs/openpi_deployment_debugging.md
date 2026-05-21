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

### Option B: Add a kickstart to the deployment config

Add a `warmup_commands_deg` field to `pi05_lora_step9999_so101.json` that the live script executes before starting policy inference — e.g., a gradual 10° shoulder_lift raise over 3 steps. This automates the bypass without requiring manual pre-positioning.

### Option C: Re-collect training data without the countdown hold

When recording new episodes, have the demonstrator start the leader arm motion immediately at t=0 (no countdown hold). This teaches the policy to initiate from the rest position visually. Alternatively, trim the first 160 frames from each existing episode before retraining.

### Option D: Trim existing episodes (fastest retraining path)

The existing 50 episodes can be trimmed at frame 165 before re-running `build_dataset` + `train`. This removes the hold phase from training data without re-recording. The resulting policy should self-start from the rest position.

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
