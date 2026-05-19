# SO101 OpenPI Debug Handoff

Date: 2026-05-19

This handoff is for the Quantycat SO101 OpenPI fine-tuning / live inference debug.
The main goal is to explain why the robot barely moves during live inference even
though the checkpoint loads and the physical start pose looks similar to demos.

## Repos And Machines

Local machine paths inspected here:

- Quantycat repo: `/home/caroline/quantycat-positronic`
- OpenPI repo: `/home/caroline/openpi`
- Illia examples: `/home/caroline/pi-examples`

Live machine paths from pasted logs:

- Quantycat repo on lamb: `/home/caroline/Desktop/quantycat-positronic`
- OpenPI repo on lamb: `/home/caroline/Desktop/OpenPi`
- Live checkpoint on lamb:
  `/home/caroline/Desktop/fine_tuning/screwdriver_so101/05172026_pi05_h20_lora`

Important: do not assume local `/home/caroline/quantycat-positronic` is identical
to lamb's `/home/caroline/Desktop/quantycat-positronic`.

## Current Model/Config Facts

OpenPI config:

- Config name used for live inference: `pi05_quantycat_lora`
- Local definition:
  `/home/caroline/openpi/src/openpi/training/config.py`
- Dataset used by local OpenPI config:
  `/home/caroline/quantycat-positronic/my_data/input_data`
- LoRA checkpoint is real LoRA; checkpoint metadata includes `lora_a` / `lora_b`
  tensors.

Action convention:

- Dataset actions are absolute 6-D joint/gripper targets.
- OpenPI training transform converts joints 0-4 to deltas relative to current
  state.
- Gripper is left absolute.
- OpenPI output transform converts joints 0-4 back to absolute targets.
- Live script treats policy output as absolute model-space targets and sends
  target-current deltas through safety/gain logic.
- This action convention appears correct.

Relevant code:

- Local OpenPI data config:
  `/home/caroline/openpi/src/openpi/training/config.py`
- Quantycat policy transform:
  `/home/caroline/openpi/src/openpi/policies/quantycat_policy.py`
- Local copy in Quantycat repo:
  `/home/caroline/quantycat-positronic/models/openpi/training_config/quantycat_policy.py`
- Live deployment script:
  `/home/caroline/quantycat-positronic/models/openpi/deployment/live_so101_openpi.py`

## Training Data Facts

Dataset:

- `/home/caroline/quantycat-positronic/my_data/input_data`
- LeRobot v2.1
- 50 episodes
- 37,266 frames
- 30 FPS
- Robot type: `so101_follower`
- Videos: front and wrist, 640x360
- Parquet columns include:
  - `observation.state`
  - `action`
  - `observation.images.front`
  - `observation.images.wrist`
  - camera pose columns

No robot calibration JSON is stored in the dataset. OpenPI trained on already
calibrated `observation.state` / `action` values. The original LeRobot robot
calibration used during recording cannot be reconstructed exactly from OpenPI
artifacts alone.

## Live State And Calibration Findings

Live sample state from user/lamb logs:

```text
live state deg = [11.34, -101.60, 96.68, 74.88, -26.63, 1.08]
```

Earlier live diagnostic:

```text
raw ticks      = [2232, 815, 3160, 2845, 1744, 2048]
calibrated deg = [12.13, -101.34, 96.50, 75.06, -26.72, 1.08]
```

Closest local training frame found:

```text
episode_000001 frame 27
training state deg = [13.67, -100.48, 95.87, 73.93, -28.57, 0.46]
```

Difference is small:

```text
live - training approx = [-2.33, -1.12, +0.81, +0.95, +1.94, +0.62] deg
```

Conclusion: the live numeric joint state is close to at least one real early
training frame. The evidence does not support "the state is geometrically 30 deg
wrong" as the main cause.

## Important Diagnostic Added

Added diagnostic script:

```text
/home/caroline/quantycat-positronic/models/openpi/eval/compare_live_training_observation.py
```

Purpose:

- Does not connect to robot.
- Does not modify calibration.
- Runs the same policy/checkpoint on controlled combinations of training vs live
  images and training vs live state.

It compares:

- A: training images + training state
- B: training images + live state
- C: live images + training state
- D: live images + live state
- E/F: mixed front/wrist ablations

Example run used locally:

```bash
/home/caroline/openpi/.venv/bin/python \
  /home/caroline/quantycat-positronic/models/openpi/eval/compare_live_training_observation.py \
  --live-run /home/caroline/quantycat-positronic/my_data/training_pipeline/openpi/sample_run/20260519T045833Z \
  --episode 1 \
  --frame 27 \
  --sample-steps 10
```

## Sample Run Inspected

User-provided sample run path:

```text
/home/caroline/quantycat-positronic/my_data/training_pipeline/openpi/sample_run/20260519T045833Z
```

Files:

- `deployment_config.json`
- `latest_front.npy`
- `latest_wrist.npy`
- `latest_state_model.npy`
- `latest_observation.png`
- `rollout.jsonl`

Generated comparison artifacts:

- `training_vs_live_ep001_frame027.png`
- `live_wrist_nearest_training_sheet.png`
- `visual_nearest_training_sample.json`

## Controlled Replay Results

For `episode_000001 frame 27`:

```text
training state deg:
[13.67, -100.48, 95.87, 73.93, -28.57, 0.46]

dataset action delta deg:
[3.93, 0.57, -0.95, 1.36, 10.64, 0.15]
```

Policy outputs:

```text
A training images + training state
  h9 wrist_roll delta: +5.26 deg
  chunk max wrist_roll delta: +23.76 deg

B training images + live state
  h9 wrist_roll delta: +13.77 deg
  chunk max wrist_roll delta: +15.96 deg

C live images + training state
  h9 wrist_roll delta: -1.26 deg

D live images + live state
  h9 wrist_roll delta: -1.21 deg
```

Interpretation:

- The checkpoint can produce meaningful motion.
- The live state does not kill the command.
- The output changes substantially when live images are used.
- Main issue is in visual observation distribution, not numeric state.

Important wording:

- This does **not** mean every joint output becomes zero.
- Wrist-roll was used as the easiest task-relevant indicator for this frame
  because the dataset action has a large wrist-roll component:
  `[3.93, 0.57, -0.95, 1.36, 10.64, 0.15] deg`.
- Other joints still move in the live-image prediction. The specific problem is
  that the strong training-image action pattern, most visibly wrist-roll,
  disappears or flips sign under live images.

## New Dry-Run On Lamb: 20260519T161437Z

User ran on lamb:

```bash
cd /home/caroline/Desktop/quantycat-positronic
bash models/openpi/run_scripts/live_so101_step9999.sh \
  --dry-run \
  --checkpoint /home/caroline/Desktop/fine_tuning/screwdriver_so101/05172026_pi05_h20_lora \
  --max-steps 5
```

Checkpoint directory on lamb is valid and contains:

```text
assets  _CHECKPOINT_METADATA  params  train_state
```

Run log:

```text
OpenPI policy loaded
state_stats: /home/caroline/Desktop/fine_tuning/screwdriver_so101/05172026_pi05_h20_lora/assets/openpi/norm_stats.json
Robot connected. dry_run=True.
Logs: /home/caroline/Desktop/quantycat-positronic/run_logs/openpi_live_so101/20260519T161437Z
```

The dry-run produced nonzero targets and no safety clipping:

```text
step 0 state  = [11.43, -101.51, 96.68, 74.88, -26.63, 1.08]
step 0 target = [10.94, -101.18, 97.72, 76.99, -29.01, 0.38]
step 0 delta  = [-0.49, +0.34, +1.04, +2.11, -2.37, -0.70] deg
limited=False
```

Interpretation:

- The OpenPI policy loads on lamb.
- The live script is not outputting zeros.
- The script's safety limiter is not suppressing this dry-run command.
- Since `--dry-run` was used, no action should be sent to motors; the script only
  logs the target commands.

The new dry-run sample was copied/available locally at:

```text
/home/caroline/quantycat-positronic/my_data/training_pipeline/openpi/sample_run/20260519T161437Z
```

Files:

```text
deployment_config.json
latest_front.npy
latest_wrist.npy
latest_state_model.npy
latest_observation.png
rollout.jsonl
```

Reran the same controlled replay locally on this new sample:

```bash
/home/caroline/openpi/.venv/bin/python \
  /home/caroline/quantycat-positronic/models/openpi/eval/compare_live_training_observation.py \
  --live-run /home/caroline/quantycat-positronic/my_data/training_pipeline/openpi/sample_run/20260519T161437Z \
  --episode 1 \
  --frame 27 \
  --sample-steps 10
```

Key output:

```text
A training images + training state
  h9_delta_deg:    [0.84, -0.07, 0.24, 2.29, 5.26, 0.03]
  chunk_max_delta: [2.98, 0.19, 0.43, 2.97, 23.76, 0.10]

B training images + live state
  h9_delta_deg:    [3.19, 0.58, 0.31, 2.35, 13.77, -0.21]
  chunk_max_delta: [3.94, 0.83, 0.80, 3.04, 15.96, 0.34]

C live images + training state
  h9_delta_deg:    [-1.17, 0.44, 2.47, 3.22, -1.64, 0.16]
  chunk_max_delta: [-0.86, 0.58, 2.66, 3.30, -0.68, 0.22]

D live images + live state
  h9_delta_deg:    [0.96, 0.58, 1.37, 1.13, -1.76, 0.30]
  chunk_max_delta: [1.11, 0.68, 1.42, 1.28, -1.70, 0.34]
```

This reproduced the earlier finding on a newer lamb dry-run sample:

- Training images + live state still gives a strong positive wrist-roll chunk.
- Live images + live state gives a small negative wrist-roll chunk.
- The state is not the thing that kills that action pattern.

The new `latest_observation.png` is not simply dark. It is visually quite
different from the earlier dark sample: the front/high image is overexposed or
blown out, and the wrist image is bright. Therefore "the wrist image is just
dark" is not a sufficient explanation for the newer sample, though exposure,
white balance, and camera preprocessing remain plausible.

## Inference Script Review

The dry-run path in:

```text
/home/caroline/quantycat-positronic/models/openpi/deployment/live_so101_openpi.py
```

was reviewed.

Findings:

- `--checkpoint` correctly overrides the JSON checkpoint path.
- `--dry-run` still connects to robot/cameras and runs policy inference, but the
  only motor-send call is guarded by:

```python
if not args.dry_run:
    _send_robot_action(robot, target_live)
```

- Policy input keys are the expected already-repacked keys:

```python
{
    "observation/images/front": front,
    "observation/images/wrist": wrist,
    "observation/state": state_model,
    "prompt": prompt,
}
```

- This is correct because `policy_config.create_trained_policy(...)` does not
  automatically apply the training `repack_transforms` during inference unless
  extra repack transforms are explicitly passed.
- OpenPI output is treated as absolute model-space targets. This is correct for
  this config because `AbsoluteActions` is in the policy output transform chain.
- The live runner computes `delta = pred_abs - current_state`, applies
  gain/safety clipping, converts back to live units, and sends/logs absolute
  targets.
- The action convention in the inference script appears correct.

Caveat:

- For the old LeRobot API path, `_send_robot_action` sends a tensor, which
  matches old `ManipulatorRobot.send_action`.
- For a newer LeRobot API path, `_send_robot_action` sends a dict with keys like
  `shoulder_pan.pos`. This is probably correct if live observations use matching
  keys, but the safest live-machine check is to print `robot.action_features`
  and confirm the action keys exactly match.

## Camera / Image Findings

`latest_observation.png` showed:

- Left half: live front view. It roughly sees task scene.
- Right half: live wrist view. It mostly sees dark floor / robot side, not the
  same wrist view as training.

Training comparison image:

- Training wrist sees tabletop, gripper, and screwdriver area.
- Live wrist does not look like that.

Nearest-neighbor image diagnostic:

- Live front's closest sampled matches are from training front stream.
- Live wrist's closest sampled matches are not from training wrist; closest
  sampled matches were poor and came from training front.

This points to a live visual-observation mismatch. Do not overstate this as
definitely "wrong camera pose": the user believes the physical wrist camera is
the same and may be right. The mismatch may be exposure/white-balance/noise,
image preprocessing, object/scene placement, camera pose/FOV, or some
combination. The key evidence is that the model's action changes a lot when live
images replace training images.

## Brightness / Color Tests

Tested offline:

- Original live images.
- Brightened live images.
- Gamma-corrected live images.
- Per-camera mean/std matched live images to training frame.

None recovered the training-like wrist-roll command on the earlier saved sample.
Brightness may still be part of the issue, especially on the real robot, but
the existing evidence does not prove that a simple brightness fix is sufficient.
The newer 20260519T161437Z sample was bright/overexposed rather than dark and
still showed the same train-image vs live-image action split.

## Camera Index Swap Tests

Tested offline:

- Original live front + live wrist.
- Swapped live wrist as front and live front as wrist.
- Duplicated live front into both slots.
- Duplicated live wrist into both slots.
- Training front + live front as wrist.

Swapping or duplicating live images did not recover the training-like command.
This makes a simple camera-index swap less likely, though live machine camera
index probing is still useful.

Clarification:

- In `training_vs_live_ep001_frame027.png`, the first image is the high/table
  camera and the second image is the wrist camera.
- In this codebase that high/table camera stream is named `front`, which is
  confusing but appears consistent across training and live runner.
- A direct live swap test did not make the output look like the training-image
  output, so a simple `front`/`wrist` key swap is unlikely to be the whole cause.

## Illia Example Findings

Illia local example files:

- `/home/caroline/pi-examples/lerobot_random/vla/pi/evaluate_pi0.py`
- `/home/caroline/pi-examples/lerobot_random/vla/pi/config.py`
- `/home/caroline/pi-examples/lerobot_random/vla/pi/lekiwi_policy.py`

Illia wrapper does:

- Predict action chunks.
- Execute only `ACTIONS_TO_EXECUTE` actions from each chunk.
- Uses `ACTIONS_TO_EXECUTE = 20` for action horizon 30 in one example.

Your live script:

- Also predicts chunks.
- Uses `execute_steps_per_inference = 10`.
- For action horizon 20, executing 10 is not obviously wrong.

Illia delta-action note:

- He warns that only appropriate action dimensions should be delta transformed.
- Your Quantycat config does this correctly for joints 0-4 and leaves gripper
  absolute.

Important image-slot difference:

Illia's `LeKiwiInputs` maps:

```python
base_0_rgb = top
left_wrist_0_rgb = wrist
right_wrist_0_rgb = front
```

Your Quantycat transform maps:

```python
base_0_rgb = front
left_wrist_0_rgb = wrist
right_wrist_0_rgb = wrist
```

So the idea of filling a missing camera slot came from Illia, but Illia fills
the missing right-wrist slot with a different useful view, not the same wrist
image twice.

Official OpenPI UR5 example instead uses zeros for missing right wrist and masks
that slot false for PI0-style models.

This image-slot mapping should be tested next.

## No Calibration Offset Should Be Added

Do not add an arbitrary joint offset. User explicitly does not want offset hacks.
The controlled replay says live state is not the main problem.

## Slot Mapping Ablation (2026-05-19)

Ran offline using the T161437Z lamb dry-run sample and episode_000001 frame 27.

Script:

```bash
cd /home/caroline/quantycat-positronic
/home/caroline/openpi/.venv/bin/python \
  models/openpi/eval/compare_live_training_observation.py \
  --live-run my_data/training_pipeline/openpi/sample_run/20260519T161437Z \
  --episode 1 --frame 27 --sample-steps 10 \
  --output-json eval_output/screwdriver_so101/model_eval/slot_mapping_test/slot_mapping_results.json
```

Full results saved to:

```text
/home/caroline/quantycat-positronic/eval_output/screwdriver_so101/model_eval/slot_mapping_test/slot_mapping_results.json
```

### Key Numbers

Training ground truth j4 (wrist_roll) delta: `+10.64 deg`

| Case | Front | Wrist | j4 chunk extreme |
|---|---|---|---|
| A training + training | training | training | +23.76 deg |
| B training + live state | training | training | +15.96 deg |
| C live images + training state | live | live | −3.02 deg |
| D live images + live state | live | live | −1.91 deg |
| E live front + training wrist | **live** | training | −1.34 deg |
| F training front + live wrist | training | **live** | **+6.46 deg** |

### New Finding: Front Camera Is The Main Problem

Case E vs F reverses the earlier suspicion about the wrist camera.

- Case E (live front + training wrist): j4 = **−1.34 deg** — weak, wrong sign.
- Case F (training front + live wrist): j4 = **+6.46 deg** — strong, correct sign.

The live front image in the T161437Z sample is overexposed / blown out (confirmed
from `latest_observation.png`). Replacing it with a training front image recovers
most of the wrist_roll command even though the live wrist is present. The wrist
camera is providing useful signal; the front camera is not.

### Slot Mapping Comparison (live images + live state)

All three variants were tested with live images and live state (the deployment
condition). j4 training ground truth is +10.64 deg.

| Slot variant | j4 chunk extreme |
|---|---|
| wrist dup (current, trained) | **−9.47 deg** |
| front (Illia-style) | −2.55 deg |
| zeros + mask=False (UR5-style) | −3.50 deg |

The current wrist-duplicate slot amplifies the wrong-direction j4 from −1.91 to
−9.47. Illia-style (right_wrist=front) reduces it to −2.55. The sign remains
wrong in all variants because the underlying problem is the overexposed front
image, not the slot mapping. But the wrist-duplicate is making things actively
worse.

Note: none of the slot variants recover the correct sign, because the root cause
(front camera exposure) is not addressed by slot remapping alone.

### Slot Mapping Caveat

The checkpoint was trained with `right_wrist_source="wrist"`. Any inference-only
slot change is a training/inference mismatch. The Illia-style fix is pragmatic
for the short term; the durable fix is to retrain with the chosen slot mapping.

## Changes Made In Repo

Added / modified (chronological):

```text
/home/caroline/quantycat-positronic/models/openpi/eval/compare_live_training_observation.py
/home/caroline/quantycat-positronic/models/openpi/HANDOFF_SO101_OPENPI_DEBUG.md
```

Updated (slot mapping work, 2026-05-19):

```text
/home/caroline/openpi/src/openpi/policies/quantycat_policy.py
  — QuantycatInputs gains right_wrist_source param ("wrist"|"front"|"zeros")

/home/caroline/openpi/src/openpi/training/config.py
  — LeRobotQuantycatDataConfig gains right_wrist_source field, threads to QuantycatInputs

/home/caroline/quantycat-positronic/models/openpi/deployment/live_so101_openpi.py
  — _load_policy reads cfg["model"].get("right_wrist_source", "wrist") and applies
    dataclasses.replace if not "wrist"; _validate_config prints it

/home/caroline/quantycat-positronic/models/openpi/deployment/pi05_lora_step9999_so101.json
  — "right_wrist_source": "front" added (Illia-style, active on lamb)

/home/caroline/quantycat-positronic/models/openpi/eval/compare_live_training_observation.py
  — extended with slot-variant cases (G/H) and slot comparison table
```

Generated under sample run:

```text
/home/caroline/quantycat-positronic/my_data/training_pipeline/openpi/sample_run/20260519T045833Z/training_vs_live_ep001_frame027.png
/home/caroline/quantycat-positronic/my_data/training_pipeline/openpi/sample_run/20260519T045833Z/live_wrist_nearest_training_sheet.png
/home/caroline/quantycat-positronic/my_data/training_pipeline/openpi/sample_run/20260519T045833Z/visual_nearest_training_sample.json

/home/caroline/quantycat-positronic/eval_output/screwdriver_so101/model_eval/slot_mapping_test/slot_mapping_results.json
```

## What Is Still Plausible

- **Front camera exposure / auto-gain** — strongest current suspect. The T161437Z
  front image is blown out. Training data used normal exposure. Fixing exposure
  on lamb is the highest-priority next action.
- Compression or color-space differences between dataset videos and live frames.
- Camera orientation, crop, FOV, or physical viewpoint differences (front camera).
- Object/scene placement differences.
- Lamb checkpoint or OpenPI checkout differing from local copies.
- New vs old LeRobot runtime API differences.
- Model underfit on some joints (j2, j4 slopes 0.70 / 0.76 in offline eval).

Less likely based on current evidence:

- The live wrist camera being the main visual problem (case F contradicts this).
- The live numeric joint state being far outside training.
- Double-applying or forgetting action deltas in the live script.
- Safety clipping suppressing motion in the dry-run.
- A simple front/wrist camera key swap.

## Next Steps

### 1. Fix Front Camera Exposure On Lamb (highest priority)

Check whether the front camera is overexposed on lamb:

```bash
python3 -c "
import cv2, numpy as np
cap = cv2.VideoCapture(2)  # front_camera_index from deploy config
ok, frame = cap.read(); cap.release()
print('front shape:', frame.shape, 'mean:', np.mean(frame), 'max:', np.max(frame))
cv2.imwrite('/tmp/live_front_check.png', frame)
"
```

Mean > 200 or max == 255 everywhere = overexposed. Remedies:

- Disable auto-exposure: `cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)` then
  `cap.set(cv2.CAP_PROP_EXPOSURE, <value>)` (value depends on driver).
- Match exposure to training recording conditions (lighting, camera settings).
- If the camera gain/white-balance is controlled by LeRobot, check its camera
  config for exposure or gain fields.

### 2. Test The Slot Fix On Lamb (already deployed)

The deploy JSON now has `"right_wrist_source": "front"`. On lamb, after pulling:

```bash
bash models/openpi/run_scripts/live_so101_step9999.sh --check-only --skip-policy-load
# should print: right_wrist_source: front

bash models/openpi/run_scripts/live_so101_step9999.sh --dry-run --max-steps 5 \
  --checkpoint /home/caroline/Desktop/fine_tuning/screwdriver_so101/05172026_pi05_h20_lora
```

Compare j4 (wrist_roll) magnitude in the dry-run rollout.jsonl to the earlier
wrist-dup run. Expect smaller negative values (~2–3 deg instead of ~9 deg) but
sign may still be wrong until the front camera exposure is fixed.

### 3. Probe All Camera Indices On Lamb

Confirm which camera index each physical camera is on and whether the front
camera view matches training:

```bash
cd /home/caroline/Desktop/quantycat-positronic
/home/caroline/Desktop/OpenPi/.venv/bin/python - <<'PY'
import cv2
from pathlib import Path

out = Path("run_logs/camera_index_probe")
out.mkdir(parents=True, exist_ok=True)
for idx in range(8):
    cap = cv2.VideoCapture(idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print(idx, "no frame")
        continue
    path = out / f"camera_{idx}.png"
    cv2.imwrite(str(path), frame)
    print(idx, frame.shape, "mean:", frame.mean().round(1))
PY
```

Compare `camera_*.png` to training frames in:

```text
/home/caroline/quantycat-positronic/my_data/training_pipeline/openpi/sample_run/20260519T045833Z/training_vs_live_ep001_frame027.png
```

### 4. If Exposure Fix + Slot Change Helps But Does Not Fully Recover

Retrain with `right_wrist_source="front"` in the data config (already supported
via `LeRobotQuantycatDataConfig(right_wrist_source="front", ...)`). This makes
training and inference consistent and removes the slot mismatch.

## Short Root-Cause Summary

```text
The live policy is not failing because the numeric joint state is far from
training. The live front camera image is overexposed / blown out in the sampled
run, which is the primary cause of collapsed action output. The wrist camera
appears to provide useful signal (case F). The current wrist-duplicate slot
mapping amplifies the wrong-direction wrist_roll from −2 to −9 deg; switching
to Illia-style (right_wrist=front) reduces this to −3 deg as an interim measure.
The durable fix is: (1) fix front camera exposure, (2) retrain with the chosen
slot mapping for training/inference consistency.
```
