# Quantycat OpenPI Pi05 Config

This directory holds the Quantycat SO-101 configuration layer for openpi/pi05.
It does not modify the upstream openpi checkout by itself.

The runnable source of truth is the patched checkout under:

```text
/home/caroline/openpi
```

The files in this directory are reference snapshots for the Quantycat project.
If behavior changes, update the live openpi files first and then refresh these
snapshots.

## Files

- `config.yaml` - local run settings for the SO-101 screwdriver dataset.
- `training_config/quantycat_policy.py` - SO-101 input/output transforms.
- `training_config/quantycat_openpi_config.py` - source snippet for the matching openpi `TrainConfig`.

## Intended OpenPI Changes

Install the policy transform into openpi:

```bash
cp models/openpi/training_config/quantycat_policy.py \
  /home/caroline/openpi/src/openpi/policies/quantycat_policy.py
```

Then add the contents of `LeRobotQuantycatDataConfig` and
`PI05_QUANTYCAT_CONFIG` from `training_config/quantycat_openpi_config.py` to:

```text
/home/caroline/openpi/src/openpi/training/config.py
```

Also add this import near the other policy imports:

```python
import openpi.policies.quantycat_policy as quantycat_policy
```

## Commands After Installation

From `/home/caroline/openpi`:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_quantycat
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_quantycat --exp-name=screwdriver_so101_pi05_h20_v1 --overwrite
```

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
