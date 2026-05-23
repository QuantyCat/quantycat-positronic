# OpenPI SO-101 Live Deployment

This directory contains the control-machine wrapper for testing the OpenPI
pi0.5 LoRA `9999` checkpoint on the live SO-101 screwdriver task.

The checkpoint itself is intentionally not committed. Copy or download it onto
the robot control computer separately, then edit or override
`model.checkpoint_path` in:

```text
models/openpi/deployment/pi05_lora_step9999_so101.json
```

## Required Pieces

- This `quantycat-positronic` repo.
- A patched OpenPI checkout, default path `/home/caroline/openpi`.
- The OpenPI checkpoint directory for step `9999`, including both `params/` and
  `assets/`.
- LeRobot-compatible SO-101 control environment with the same camera names and
  motor observation keys used by the existing RynnVLA live script.

The OpenPI checkout must expose the `pi05_quantycat_lora` config and
`openpi.policies.quantycat_policy`. The quantycat copy of the policy transform
is kept at:

```text
models/openpi/training_config/quantycat_policy.py
```

## Recommended First Commands

From the repo root on the control computer:

```bash
cd /home/caroline/quantycat-positronic

# Fast config/path check without loading model weights.
/home/caroline/openpi/.venv/bin/python \
  models/openpi/deployment/live_so101_openpi.py \
  --check-only \
  --skip-policy-load

# Full policy-load check. This should verify the OpenPI config and checkpoint.
/home/caroline/openpi/.venv/bin/python \
  models/openpi/deployment/live_so101_openpi.py \
  --check-only
```

If the checkpoint is somewhere else:

```bash
/home/caroline/openpi/.venv/bin/python \
  models/openpi/deployment/live_so101_openpi.py \
  --check-only \
  --checkpoint /path/to/9999
```

## Dry Run

Use dry-run first. It connects to the robot, reads cameras/state, runs the
policy, applies gain and safety clipping, writes logs, and prints the command it
would send, but does not call `robot.send_action()`.

```bash
/home/caroline/openpi/.venv/bin/python \
  models/openpi/deployment/live_so101_openpi.py \
  --dry-run \
  --max-steps 5
```

## Current Config Defaults

- `max_steps`: `0` (runs until Ctrl+C)
- `execute_steps_per_inference`: `20` (execute full action horizon before re-inferring)
- `control_period_s`: `0.033` (30 Hz, matching training fps)
- max command delta: `4 deg` on arm joints, `6 deg` elbow, `2 deg` gripper
- deployment gain vector: `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]` (all 1.0 — recalibrate after first run)

Run a short live test with an explicit step cap:

```bash
/home/caroline/openpi/.venv/bin/python \
  models/openpi/deployment/live_so101_openpi.py \
  --max-steps 60
```

## Self-Start Workaround

From the default rest pose the model may predict near-hold actions. Pre-position
the arm to an active in-trajectory pose before handing control to the policy:

```bash
cd /home/caroline/quantycat-positronic
python models/openpi/deployment/test_robot_send_action.py --start-pose 4 -85 92 67 6 0.4 --wait 3.0
bash models/openpi/run_scripts/live_so101_step9999.sh
```

This was a known issue with the previous checkpoint trained on unclean data
(countdown hold included). The v2 checkpoint trained on `clean_input_data3`
(hold trimmed, pauses removed, actions smoothed) should self-start more
reliably.

## Action Convention

OpenPI policy inference returns absolute 6-D targets in model units. The live
runner converts the current LeRobot state from degrees to radians, then applies
gain in delta space:

```text
delta = predicted_target - current_state
calibrated_target = current_state + delta * gain_vector
```

After gain, the runner clips per-command deltas and final absolute targets,
converts back to live robot units, and sends the command dictionary.

## Logs

Each rollout writes:

```text
run_logs/openpi_live_so101/<timestamp>/deployment_config.json
run_logs/openpi_live_so101/<timestamp>/rollout.jsonl
```

The JSONL log records raw state, model-unit state, policy target, final clipped
target, and whether safety clipping changed the command.
