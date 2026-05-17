# OpenPI Parallel Development Strategy

Date: 2026-05-16

## Goal

Run OpenPI/pi0.5 development in parallel with the current RynnVLA-002 screwdriver
work, using the same SO-101 demonstrations where possible, while keeping the two
model tracks comparable enough to make useful decisions.

## Current Local State

The runnable OpenPI checkout is:

```text
/home/caroline/openpi
```

The Quantycat wrapper notes/scripts are:

```text
/home/caroline/quantycat-positronic/models/openpi
```

The existing OpenPI config path is already wired:

```text
config: pi05_quantycat
source: /home/caroline/openpi/src/openpi/training/config.py
policy transforms: /home/caroline/openpi/src/openpi/policies/quantycat_policy.py
```

The config currently uses:

- model: `Pi0Config(pi05=True, action_horizon=20)`
- data: `/home/caroline/quantycat-positronic/my_data/input_data`
- base checkpoint: `gs://openpi-assets/checkpoints/pi05_base/params`
- train steps: `5000`
- batch size: `4`
- save interval: `1000`
- norm stats: `/home/caroline/quantycat-positronic/my_data/training_pipeline/openpi/norm_stats.json`

## Can We Reuse the RynnVLA Examples?

Yes, but the reusable artifact is the original LeRobot dataset, not the
RynnVLA-tokenized training split.

Use this for OpenPI:

```text
/home/caroline/quantycat-positronic/my_data/input_data
```

This is a LeRobot-style dataset with parquet episodes and metadata:

```text
data/chunk-000/episode_*.parquet
meta/info.json
meta/episodes.jsonl
meta/tasks.jsonl
meta/episodes_stats.jsonl
```

Do not use these directly for OpenPI training:

```text
my_data/training_pipeline/tokens/vla_data/...
my_data/training_pipeline/conversations/...
```

Those are RynnVLA-specific token/conversation products. They are still useful as
metadata for selecting subsets, but OpenPI should consume the source LeRobot
episodes.

## Data Semantics

The OpenPI Quantycat transform currently expects/repackages:

```text
observation.images.front -> observation/images/front
observation.images.wrist -> observation/images/wrist
observation.state        -> observation/state
action                   -> action
prompt                   -> prompt
```

The policy transform maps SO-101 into OpenPI inputs:

```text
state: 6-D absolute joint/gripper position
images:
  base_0_rgb: front camera
  left_wrist_0_rgb: wrist camera
  right_wrist_0_rgb: duplicated wrist camera
actions: 6-D action
```

The current OpenPI config then applies:

```text
DeltaActions(mask=[j0,j1,j2,j3,j4], gripper_absolute=True)
```

So OpenPI trains joints 0-4 as deltas relative to the current state and leaves
the gripper absolute.

This is close enough to the RynnVLA action objective to compare behavior, but it
is not identical: RynnVLA has been training 5-step chunks and custom weighted
joint losses, while the OpenPI config uses a 20-step action horizon and the
standard OpenPI objective.

## Recommended First Track: pi0.5

Start with pi0.5, not pi0, because the local config is already `pi05_quantycat`
and uses the pi0.5 base checkpoint path. This minimizes setup risk.

Baseline run:

```bash
bash models/openpi/run_scripts/preprocess.sh
bash models/openpi/run_scripts/training.sh
```

The first milestone should be an overfit/sanity run, not full model selection:

1. Confirm the local LeRobot dataset loads without HuggingFace path issues.
2. Confirm norm stats are used and action/state normalization is reasonable.
3. Confirm a checkpoint is saved at step 1000.
4. Run a small teacher-forced or offline action trace eval on the same selected
   high-motion windows used for RynnVLA.

## Evaluation Plan

To compare OpenPI with RynnVLA, build an OpenPI eval path that mirrors the
existing RynnVLA high-motion eval:

1. Reuse selected windows from the RynnVLA eval buckets:
   - `broad_j0_high_motion_top50`
   - `broad_j1_high_motion_top50`
   - `broad_j2_high_motion_top50`
   - `broad_j3_high_motion_top50`
   - `broad_j4_high_motion_top50`
2. For each window, feed observations through an OpenPI policy checkpoint.
3. Extract the first relevant action or the full action chunk, depending on
   deployment behavior.
4. Compute the same diagonal metrics:
   - sign agreement
   - correlation
   - fitted slope
   - normalized MAE

This will make OpenPI results directly comparable to the current RynnVLA tables.

## Risks and Required Work

1. Dataset loading may still be fragile.
   The README notes that OpenPI's pinned LeRobot may require a local-path patch
   in `openpi.training.data_loader.create_torch_dataset`. The current code still
   constructs `LeRobotDatasetMetadata(repo_id)` and `LeRobotDataset(repo_id)`;
   this must be validated with a real preprocessing/training command.

2. Horizon mismatch may make early comparisons noisy.
   OpenPI is set to `action_horizon=20`; RynnVLA is currently chunk size 5. For
   fair comparison, evaluate both first-action behavior and short-horizon
   integrated behavior.

3. Loss objectives differ.
   RynnVLA is using custom sign/motion/magnitude/quiet losses. OpenPI will not
   reproduce those without model/training changes. Treat initial OpenPI as a
   model-family comparison, not a loss-ablation comparison.

4. Camera mapping is approximate.
   SO-101 has one wrist camera; OpenPI receives it in both wrist slots. This is
   acceptable for a first pass but should be recorded in every result table.

5. Normalization may dominate.
   OpenPI docs warn that poor norm stats can destabilize training. Inspect
   `norm_stats.json`, especially low-variance action dimensions, before drawing
   conclusions from a bad run.

## Suggested Development Order

1. Validate current `pi05_quantycat` end to end for 1000 steps.
2. Add an OpenPI high-motion eval wrapper using the same selected-window JSONs.
3. Compare step-1000 and step-5000 OpenPI checkpoints against:
   - RynnVLA baseline epoch 4
   - RynnVLA KI epoch 4
   - RynnVLA sign-route epoch 4 when available
4. If OpenPI shows useful slopes, add deployment-path inference for SO-101.
5. Only then consider pi0 or pi0-fast variants.

## Decision Criteria

OpenPI is worth parallel investment if, on the same high-motion windows, it
achieves one of:

- better `j2`/`j4` slopes than RynnVLA without collapsing `j0`, or
- materially better first-action deployment behavior, or
- cleaner physical rollouts even if offline slope metrics are similar.

If it cannot train/eval cleanly on the existing LeRobot dataset within the
current local setup, the immediate next task is fixing the dataset loader path,
not changing model architecture.
