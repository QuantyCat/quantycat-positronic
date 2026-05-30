Robot demos -> LeRobot v3.0 dataset -> PI 0.5 LoRA fine-tune -> offline eval -> iron-fleet robot test.

**Current priority:** prepare the next Physical Intelligence `pi05` run through the
LeRobot v3.0 pipeline. Older OpenPI/RynnVLA paths remain useful for comparison, but
the next production path is:

```text
~/quantycat-data/datasets/<curated-v3-dataset>/
    -> models/lerobot/run_scripts/training.sh
    -> ~/quantycat-data/checkpoints/lerobot/pi05/<exp-name>/
    -> models/lerobot/eval/lerobot_lora_high_motion_eval.py
    -> ~/quantycat-iron-fleet for live robot testing
```

This repo owns dataset preparation, training, and offline eval. Live robot rollout
belongs in `~/quantycat-iron-fleet`.

---

## Current Repo Structure

```text
quantycat-positronic/
  models/
    lerobot/            current PI 0.5 LoRA training path
      run_scripts/
        setup.sh
        training.sh
      eval/
        lerobot_lora_high_motion_eval.py
    openpi/             historical OpenPI path and config patches
    rynnvla-002/        historical RynnVLA experiments
    act/
  eval/
    openpi/             shared offline eval utilities and older OpenPI evals
    rynnvla_002/
  preprocessing_data/   mostly older LeRobot v2.1-style transforms
  vendor/
    lerobot/.venv       LeRobot 0.5.1 environment for pi05
    openpi/             historical OpenPI dependency
  vibe_docs/
```

Data and checkpoints live outside the repo by default:

```text
~/quantycat-data/
  datasets/
  checkpoints/
  eval_output/
  logs/
```

`QUANTYCAT_DATA_HOME` overrides this root.

---

## LeRobot v3.0 Dataset Contract

The active raw dataset is:

```text
~/quantycat-data/datasets/screwdriver_so101/
```

Current metadata:

```text
LeRobot version: v3.0
episodes:        50
frames:          37,266
fps:             30
task:            Put the screwdriver into the cup
```

LeRobot v3.0 stores data differently from the older v2.1 datasets:

```text
<dataset>/
  data/chunk-*/file-*.parquet
  videos/observation.images.front/chunk-000/file-000.mp4
  videos/observation.images.wrist/chunk-000/file-000.mp4
  meta/info.json
  meta/stats.json
```

Do not run the old `preprocessing_data/trim_dataset.py`,
`remove_pauses.py`, or `smooth_actions.py` directly on this v3 dataset until
they are ported or wrapped for v3. Those scripts expect older per-episode files
like `data/chunk-000/episode_000000.parquet` and
`videos/chunk-000/<camera>/episode_000000.mp4`.

The canonical joint order is:

```text
0 shoulder_pan.pos
1 shoulder_lift.pos
2 elbow_flex.pos
3 wrist_flex.pos
4 wrist_roll.pos
5 gripper.pos
```

The dataset stores model-space joint positions in radians. iron-fleet reads live
SO-101 state in degrees, converts it to radians for the policy, and converts
policy targets back to degrees before commanding the robot.

---

## Action Semantics For The Next PI 0.5 Run

The next run should not train on naive commanded-target labels.

What worked better in the OpenPI experiments was **achieved motion**:

```text
arm target for supervision = future achieved observation.state - current observation.state
```

For the LeRobot `pi05` policy, prefer using the built-in relative-action
processor instead of making the deployed policy output raw deltas:

```text
raw dataset action for joints 0-4: absolute future achieved joint state
pi05 preprocessor:                 action - current state
pi05 postprocessor:                predicted relative action + current state
live policy output:                absolute joint target
gripper:                           stays absolute unless explicitly changed
```

That implies two required checks before training:

1. The curated training dataset should rewrite arm actions to achieved future
   states, not stale commanded controller targets. A one-step version is:

   ```text
   action[t, 0:5] = observation.state[t + 1, 0:5]
   action[t, 5]   = original gripper action
   ```

   For chunked training, the LeRobot dataloader then builds the horizon from
   these absolute future achieved states.

2. Training should enable relative actions and exclude the gripper:

   ```bash
   --policy.use_relative_actions=true \
   --policy.relative_exclude_joints='["gripper.pos"]'
   ```

`meta/stats.json` must match that preprocessor order. LeRobot pi05 applies:

```text
raw action -> relative conversion -> normalize
```

So if `use_relative_actions=true`, recompute dataset stats with the same
relative-action settings. A good preflight is that arm action stats look like
small deltas around zero, not absolute joint positions centered near the robot's
home pose.

---

## Lessons To Preserve

These are not optional cleanup items; they are failure modes we have already hit.

### Use raw physical zero, not normalized zero

Do sign, damping, and gain analysis in raw model units after unnormalizing. A
normalized value of `0` is the center of the normalization range, not the
physical joint's zero angle. Comparing signs in normalized space caused false
sign conclusions before.

For arm joints, the sign to audit is:

```text
raw_delta = target_or_future_state - current_state
```

not:

```text
normalized_action - 0
```

### Remove no-motion preroll

The original screwdriver dataset had about 5.5 seconds of no-motion countdown
hold at the start of every episode, roughly frames 163-197. A previous live
checkpoint learned to hold still from the rest scene because that scene was
overrepresented with near-zero labels.

Before the next run, remeasure first motion in the v3 dataset and trim by actual
episode motion, not by assumption. The historical constant trim was 165 frames,
but the v3 curation step should produce a report.

### Remove or quarantine known bad demonstrations

Historical audit found:

```text
episode 45: failed demo/drop/re-grasp; remove
episode 28: unusually deep trajectory; inspect before deciding
episode 12: short leader glitch; probably tolerable or smoothable
```

Episode numbers must be interpreted before any reindexing. Keep a mapping from
original episode index to curated episode index.

### Reduce low-motion contamination

RynnVLA ablations showed that the problem is not only the initial countdown.
Mid-episode pauses and low-motion tails can push models toward conservative
template actions. For the next PI 0.5 run, build an active-motion report that
counts low-motion spans throughout each episode.

Do not blindly delete every pause if the pause is part of task semantics. The
goal is to remove recording artifacts and indecision, not the stable placement
phase.

### Treat wrist roll as noisy until proven otherwise

The old audit found very high wrist-roll path variance even though final
endpoints were consistent. Keep j4 in eval, but do not let a strong aggregate
loss hide poor j4 sign/correlation. Consider a run variant that downweights,
freezes, or smooths wrist roll only after the baseline curated run.

### Dataset scope is transport/place, not full pickup

The demonstrations start with the screwdriver already held. A policy trained on
this data should be evaluated as a transport/place policy, not a pick-from-table
policy.

---

## Current LeRobot Baseline

Two default LeRobot pi05 LoRA runs exist:

```text
~/quantycat-data/checkpoints/lerobot/pi05/05282026_pi05_lerobot/checkpoints/003000/pretrained_model
~/quantycat-data/checkpoints/lerobot/pi05/05282026_1657_pi05_lerobot/checkpoints/010000/pretrained_model
```

Their high-motion evals are under:

```text
~/quantycat-data/eval_output/lerobot/pi05/lerobot_lora_checkpoints_003000/
~/quantycat-data/eval_output/lerobot/pi05/lerobot_lora_checkpoints_010000/
```

The 10k default run is not a good enough deployment candidate. Diagonal
high-motion metrics are weak or near template-like:

| Focus | Sign | Corr | Slope | Norm MAE |
|---|---:|---:|---:|---:|
| j0 | 0.453 | -0.088 | -0.071 | 0.451 |
| j1 | 0.492 | 0.120 | 0.155 | 0.425 |
| j2 | 0.677 | 0.023 | 0.038 | 0.379 |
| j3 | 0.752 | 0.275 | 0.282 | 0.311 |
| j4 | 0.643 | 0.100 | 0.089 | 0.371 |

Use this as a regression baseline: the next run should beat it because it fixes
data semantics and curation, not because it merely trains longer.

---

## Training Commands

Install the LeRobot environment once:

```bash
cd /home/caroline/quantycat-positronic
bash models/lerobot/run_scripts/setup.sh
```

Current launcher:

```bash
DATASET_REPO_ID=screwdriver_so101 \
EXP_NAME=05302026_pi05_lerobot_baseline \
bash models/lerobot/run_scripts/training.sh
```

The launcher currently uses:

```text
policy.type=pi05
policy.pretrained_path=lerobot/pi05_base
peft.method_type=LORA
steps=10000
batch_size=2
dtype=bfloat16
compile_model=true
gradient_checkpointing=true
wandb.enable=true
```

Before the next serious run, update the launcher or pass overrides so it trains
on the curated dataset with relative actions:

```bash
DATASET_REPO_ID=screwdriver_so101_pi05_active_achieved_v1 \
EXP_NAME=05302026_pi05_lerobot_active_achieved_v1 \
bash models/lerobot/run_scripts/training.sh
```

and ensure the `lerobot-train` invocation includes:

```text
--policy.use_relative_actions=true
--policy.relative_exclude_joints='["gripper.pos"]'
```

---

## Offline Eval Gate

Run the LeRobot LoRA high-motion eval after each candidate checkpoint:

```bash
vendor/lerobot/.venv/bin/python models/lerobot/eval/lerobot_lora_high_motion_eval.py \
  --dataset-root ~/quantycat-data/datasets/<curated-v3-dataset> \
  --checkpoint ~/quantycat-data/checkpoints/lerobot/pi05/<exp-name>/checkpoints/<step>/pretrained_model \
  --label <exp-name>_<step> \
  --save-traces
```

Evaluate train and heldout splits separately. Do not call a checkpoint
live-testable based only on training-set replay.

Minimum gates before iron-fleet:

```text
- no negative diagonal slopes on j0-j4 high-motion windows
- raw sign agreement clearly above the default-run baseline
- j1/j2 shoulder/elbow correlations are meaningfully positive
- no normalized-zero sign audit failures
- gripper remains absolute and is not accidentally relative-normalized
- predicted deltas are not mostly template outputs with tiny variance
```

Then run gain calibration on heldout traces only. Gains must scale deltas:

```text
target = current_state + (policy_target - current_state) * gain_vector
```

Never multiply absolute joint targets directly.

---

## iron-fleet Deployment Contract

`~/quantycat-iron-fleet` is the standard place for physical SO-101 testing.
Current relevant files:

```text
scripts/run_policy.py
scripts/run_openpi.sh
iron_fleet/deployment/state_adapter.py
iron_fleet/deployment/gain_clip.py
robots/so101/config.json
robots/so101/openpi_screwdriver.json
robots/so101/go_home.py
```

iron-fleet currently has an OpenPI policy adapter. A LeRobot pi05 LoRA
checkpoint will need either:

```text
1. a new iron_fleet/policies/lerobot_pi05.py adapter that loads the LeRobot
   checkpoint's pretrained_model/ directory, or
2. an explicit export/conversion path into a format the existing OpenPI adapter
   can load.
```

Prefer option 1 for the next run because `models/lerobot/eval/lerobot_lora_high_motion_eval.py`
already demonstrates how to load `PI05Policy`, `PeftModel`, and the saved
LeRobot pre/postprocessors.

The iron-fleet runner should preserve the existing safety behavior:

```text
- live deg -> model rad state conversion
- policy absolute target in model units
- gain applied to target-current delta
- per-command delta clipping
- optional target limits
- raw policy target, gained target, clipped target, and achieved state logged
```

Typical live-test progression:

```bash
cd ~/quantycat-iron-fleet

# 1. Known physical start.
bash robots/so101/run_scripts/go_home.sh

# 2. Config validation, first without loading GPU weights.
bash scripts/run_openpi.sh --check-only --skip-policy-load

# 3. Full policy load check.
bash scripts/run_openpi.sh --check-only

# 4. Dry-run inference and inspect logged deltas.
bash scripts/run_openpi.sh --dry-run --max-steps 5

# 5. Short live run with conservative limits.
bash scripts/run_openpi.sh --max-steps 10
```

For a LeRobot adapter, keep the same sequence but use a LeRobot-specific config
and wrapper name.

---

## Plan Of Attack For The Next Run

### 1. Freeze and audit the current baseline

- Record `meta/info.json`, `meta/stats.json`, current training launcher args,
  and the 3k/10k eval summaries.
- Treat the default 10k LeRobot run as a baseline to beat, not as a candidate
  to deploy.

### 2. Build a v3 curation tool/report

- Read consolidated v3 parquet files from `data/chunk-*/file-*.parquet`.
- Group by `episode_index` and preserve original episode IDs.
- Remeasure first motion per episode using raw arm deltas.
- Detect low-motion spans throughout each episode.
- Flag known issues: episode 45 removal, episode 28 inspection, episode 12
  glitch, wrist-roll variance, gripper behavior.
- Write a curation report before writing a new dataset.

### 3. Produce the curated achieved-action dataset

Recommended dataset name:

```text
~/quantycat-data/datasets/screwdriver_so101_pi05_active_achieved_v1
```

Required properties:

- LeRobot v3.0 layout remains valid.
- No-motion preroll removed based on measured motion.
- Episode 45 removed unless a new video inspection overturns the old audit.
- Arm actions are absolute future achieved states.
- Gripper action remains absolute.
- `meta/info.json` and episode mappings are regenerated.
- `meta/stats.json` is recomputed for the exact relative-action setting used
  in training.

### 4. Create an episode-level heldout split

Use the heldout split to choose data semantics and gains. After that, a final
production run can train on all curated good episodes.

Recommended split:

```text
train:   curated good episodes minus heldout
heldout: 10 original episodes, fixed seed, recorded in a split manifest
```

### 5. Run one controlled training job

Start with the same model hyperparameters as the default LeRobot 10k run. The
main experimental change should be data quality and action semantics:

```text
pi05_base + LoRA
batch_size=2
steps=10000
bfloat16
compile_model=true
gradient_checkpointing=true
use_relative_actions=true
relative_exclude_joints=["gripper.pos"]
```

Avoid changing LoRA rank, LR, smoothing, wrist-roll weighting, and dataset
filtering all at once.

### 6. Evaluate before live testing

- Run high-motion eval on train and heldout.
- Run normalized-zero sign audit in raw units.
- Run gain calibration only after heldout replay is credible.
- Promote a checkpoint only if it improves over the current 10k default run and
  does not regress the prior achieved-delta lessons on j1/j2.

### 7. Add LeRobot pi05 support to iron-fleet

- Add a LeRobot policy adapter or conversion path.
- Add a SO-101 LeRobot deployment config with checkpoint path, prompt, safety,
  and gain vector.
- Keep check-only, dry-run, short-run, and rollout logging behavior identical
  to the OpenPI runner.

### 8. Physical test sequence

- Home the arm.
- Run dry-run and inspect predicted raw deltas.
- Run 5-10 live steps with conservative clips.
- Compare rollout logs against offline expected direction and magnitude.
- Only then run longer transport/place trials.
