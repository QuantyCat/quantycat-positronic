# Quantycat OpenPI Pi05

This directory holds the Quantycat helper scripts and notes for openpi/pi05.
It does not contain a second runnable OpenPI training config.

The runnable source of truth is the patched OpenPI checkout under:

```text
/home/caroline/openpi
```

In particular, the active config is:

```text
/home/caroline/openpi/src/openpi/training/config.py
```

and the active policy transform is:

```text
/home/caroline/openpi/src/openpi/policies/quantycat_policy.py
```

## Files

- `run_scripts/setup.sh` - install/sync the OpenPI environment.
- `run_scripts/preprocess.sh` - compute OpenPI norm stats.
- `run_scripts/training.sh` - launch training with `pi05_quantycat`.
- `run_scripts/live_so101_step9999.sh` - launch the live SO-101 OpenPI wrapper.
- `deployment/` - live robot config, runner, and control-machine checklist.
- `training_config/quantycat_policy.py` - optional readable copy of the policy
  transform. The live copy is in `/home/caroline/openpi`.

## Commands

From `/home/caroline/quantycat-positronic`:

```bash
bash models/openpi/run_scripts/setup.sh
bash models/openpi/run_scripts/preprocess.sh
bash models/openpi/run_scripts/training.sh
```

## Live SO-101 Deployment

The current deployment candidate is the OpenPI pi0.5 LoRA continuation checkpoint
at step `9999`, using config name:

```text
pi05_quantycat_lora
```

The checkpoint is not stored in this repo. Download/copy it separately onto the
robot control computer, then update or override:

```text
models/openpi/deployment/pi05_lora_step9999_so101.json
```

Run the deployment checks from the repo root:

```bash
bash models/openpi/run_scripts/live_so101_step9999.sh --check-only --skip-policy-load
bash models/openpi/run_scripts/live_so101_step9999.sh --check-only --checkpoint /path/to/9999
```

Then dry-run before sending commands:

```bash
bash models/openpi/run_scripts/live_so101_step9999.sh --dry-run --checkpoint /path/to/9999 --max-steps 5
```

The default live gain vector is:

```text
[1.000, 1.000, 1.000, 1.000, 1.025, 1.000]
```

Gain is applied to `predicted_target - current_state`, then the runner clips
per-command deltas and final targets before sending actions to SO-101.

## Inference Input Keys

Training uses `LeRobotQuantycatDataConfig.repack_transforms`, but inference does
not automatically run that training repack transform. An inference caller should
send observations using the keys expected by `QuantycatInputs`:

```python
{
    "observation/state": state,
    "observation/images/front": front_image,
    "observation/images/wrist": wrist_image,
    "prompt": "Put the screwdriver into the cup",
}
```

## Notes

The dataset is local:

```text
/home/caroline/quantycat-positronic/my_data/input_data
```

Normalization stats are saved and loaded from:

```text
/home/caroline/quantycat-positronic/my_data/training_pipeline/openpi/norm_stats.json
```

If openpi's pinned LeRobot version refuses local dataset paths, patching
`openpi.training.data_loader.create_torch_dataset` to pass a local `root=...`
or uploading the dataset to HuggingFace will be the next step.

The missing right wrist camera is represented by duplicating the real wrist
camera into `right_wrist_0_rgb` with `image_mask["right_wrist_0_rgb"] = True`.
This satisfies openpi's required three image keys without relying on masked
camera behavior.
