# Action Plan: Make Achieved-Delta OpenPI the Standard Pipeline

**Date:** 2026-05-24
**Current best checkpoint:** `models/openpi/training_pipeline/checkpoints/pi05_quantycat_lora_achieved_delta/05242026_pi05_lora_achieved_delta/4999`
**Current best config:** `pi05_quantycat_lora_achieved_delta`
**Current best dataset:** `my_data/clean_input_data_achieved_delta`

## Goal

Promote the achieved-delta label convention from an experimental variant to the default SO-101/OpenPI training path.

This is a label-semantics fix, not an ablation. For arm joints 0-4, the model should train on physically achieved motion:

```text
future_achieved_state - current_state
```

not commanded controller targets:

```text
commanded_target - current_state
```

The commanded-target labels can disagree with the robot's actual motion because of gravity, servo lag, safety clipping, saturation, or stalls. The achieved-delta pipeline aligns the supervised target with what the robot actually did.

## 1. Make Achieved-Delta the Documented Default

Update the OpenPI pipeline docs so future runs start from `clean_input_data_achieved_delta` or a newly generated achieved-delta dataset, not `clean_input_data`.

Files to update:

- `vibe_docs/models/openpi_pipeline.md`
- Any runbook that still references `pi05_quantycat_lora` as the default training config
- Any notes that describe `clean_input_data` as the final training dataset

Required doc changes:

- State that SO-101/OpenPI training labels should use achieved arm deltas.
- Explain that raw dataset `action` is still absolute before OpenPI transforms.
- Explain the convention:

```text
raw action[t, 0:5] = observation.state[t + 1, 0:5]
OpenPI DeltaActions => state[t + h + 1, 0:5] - state[t, 0:5]
```

- Preserve the warning that gripper action remains in the original absolute-action convention unless separately changed.

Done when:

- A new OpenPI training runbook points at the achieved-delta dataset/config by default.
- The old commanded-target path is documented as historical/debug only.

## 2. Update Training and Preprocess Entry Points

Make it difficult to accidentally train on the old commanded-target labels.

Current achieved-delta builder:

```text
models/openpi/preprocess_achieved_delta_dataset.py
```

Current achieved-delta config:

```text
pi05_quantycat_lora_achieved_delta
```

Recommended changes:

- Add a standard preprocessing launcher for achieved-delta generation.
- Update `models/openpi/run_scripts/preprocess.sh` or add a new standard script that computes norm stats for the achieved-delta config.
- Update `models/openpi/run_scripts/training.sh` defaults or add a new standard training launcher that defaults to:

```text
CONFIG_NAME=pi05_quantycat_lora_achieved_delta
EXP_NAME=<date>_pi05_lora_achieved_delta
```

- Add a preflight check that rejects `clean_input_data` for the standard SO-101 training path unless explicitly overridden.

Done when:

- A normal "train OpenPI SO-101" command uses achieved-delta labels without manual config edits.
- The old `pi05_quantycat_lora` path is still available for reproduction but no longer the default.

## 3. Rename or Canonicalize the Config/Dataset

Once the live robot result confirms the offline eval, decide whether to keep the explicit `achieved_delta` name or rename the standard path.

Option A: Keep explicit names.

```text
pi05_quantycat_lora_achieved_delta
clean_input_data_achieved_delta
```

Pros:

- Label convention is obvious.
- Harder to confuse with previous commanded-target runs.

Option B: Promote to canonical names.

```text
pi05_quantycat_lora_so101
clean_input_data_so101
```

Pros:

- Cleaner long-term naming.
- Treats achieved-delta as the normal SO-101 convention.

Recommended short-term choice:

Keep `achieved_delta` in names until after the first successful live run. Then either keep it permanently for clarity or create canonical aliases/docs that point to the achieved-delta paths.

Done when:

- New training docs have one blessed config name.
- Checkpoint, norm-stats, eval, and live deployment docs all refer to the same canonical name.

## 4. Update Live Inference Config

Before live robot testing, point the OpenPI live config at the achieved-delta checkpoint and choose the deployment gain vector.

Candidate checkpoint:

```text
models/openpi/training_pipeline/checkpoints/pi05_quantycat_lora_achieved_delta/05242026_pi05_lora_achieved_delta/4999
```

Offline eval baseline:

| Joint | Sign | Slope | Corr | Norm MAE |
|---|---:|---:|---:|---:|
| j0 | 0.996 | 0.969 | 0.975 | 0.062 |
| j1 | 0.996 | 0.893 | 0.934 | 0.053 |
| j2 | 0.997 | 0.862 | 0.967 | 0.070 |
| j3 | 0.933 | 0.776 | 0.859 | 0.101 |
| j4 | 0.988 | 0.810 | 0.907 | 0.098 |

Conservative calibrated gains from offline trace replay:

```text
[1.0, 1.0, 1.1, 1.0, 1.2, 1.0]
```

Recommended live-test sequence:

1. Run config validation/check-only against the new checkpoint.
2. Run dry-run inference and inspect raw target deltas.
3. First live run can use either all gains at `1.0` or the conservative gain vector above.
4. Log raw policy target, gained target, clipped target, and achieved state at every step.

Done when:

- Live config points at the achieved-delta checkpoint.
- Gain vector decision is documented.
- Live rollout logs clearly preserve raw vs gained vs clipped commands.

## Acceptance Criteria

The achieved-delta pipeline can be considered standard when:

- Docs call achieved-delta labels the default for SO-101/OpenPI.
- Preprocess/training launchers default to the achieved-delta dataset/config.
- Offline eval scripts compare against achieved-delta labels, not stale commanded-target labels.
- Live config points at the achieved-delta checkpoint.
- First live run shows motion consistent with the offline eval, especially for j2 elbow_flex.

## Related Artifacts

- Achieved-delta dataset builder: `models/openpi/preprocess_achieved_delta_dataset.py`
- Achieved-delta dataset report: `my_data/clean_input_data_achieved_delta/achieved_delta_report/summary.md`
- Achieved-delta eval script: `models/openpi/eval/model_eval/openpi_lerobot_high_motion_eval.py`
- Achieved-delta eval summary: `vibe_docs/learnings/model_development/openpi/05242026_pi05_lora_achieved_delta.md`
- Gain calibration summary: `eval_output/screwdriver_so101/model_eval/openpi_05242026_pi05_lora_achieved_delta_step4999_top20_gain_calibration/gain_calibration_summary.md`
