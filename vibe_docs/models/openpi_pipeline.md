Robot demos → OpenPI π0.5 fine-tune → trained checkpoint → inference on SO-101.

This pipeline fine-tunes Physical Intelligence `openpi` on the local Quantycat
SO-101 screwdriver dataset. Unlike the RynnVLA pipeline, the input LeRobot
dataset is used directly: there is no conversion into per-episode image/action
folders and no pretokenization step.

---

## Full Pipeline

```
my_data/input_data/                         (LeRobot v2.1 format — never modified)
    data/chunk-000/episode_*.parquet
    videos/chunk-000/observation.images.front/*.mp4
    videos/chunk-000/observation.images.wrist/*.mp4
    meta/*.json*
        ↓  setup.sh
/home/caroline/openpi/.venv                 (uv-managed openpi environment)
        ↓  preprocess.sh
my_data/training_pipeline/openpi/norm_stats.json
        ↓  training.sh
my_data/training_pipeline/openpi/checkpoints/pi05_quantycat/screwdriver_so101_pi05_h20_v1/
        ↓  policy server / inference client
SO-101 robot
```

---

## Key Commands To Run

From the Quantycat repo root:

```bash
cd /home/caroline/quantycat-positronic
bash models/openpi/run_scripts/setup.sh
bash models/openpi/run_scripts/preprocess.sh
bash models/openpi/run_scripts/training.sh
```

Use `tmux` for training so it survives disconnects:

```bash
tmux new -s openpi_train
cd /home/caroline/quantycat-positronic
bash models/openpi/run_scripts/training.sh
```

---

## Dataset

Input dataset:

```text
/home/caroline/quantycat-positronic/my_data/input_data
```

Schema:

- Robot: SO-101 follower arm
- Task: `Put the screwdriver into the cup`
- Episodes: 50
- Frames: about 37k
- FPS: 30
- Cameras:
  - `observation.images.front`
  - `observation.images.wrist`
- State: 6-D joint/gripper positions
- Action: 6-D absolute joint/gripper target positions

Action order:

```text
shoulder_pan.pos
shoulder_lift.pos
elbow_flex.pos
wrist_flex.pos
wrist_roll.pos
gripper.pos
```

---

## What Was Added To OpenPI

Live OpenPI checkout:

```text
/home/caroline/openpi
```

Changed files:

- `/home/caroline/openpi/src/openpi/policies/quantycat_policy.py`
  - New SO-101 policy transform.
  - Maps `observation/images/front` to `base_0_rgb`.
  - Maps `observation/images/wrist` to `left_wrist_0_rgb`.
  - Duplicates the wrist camera into `right_wrist_0_rgb`.
  - Sets all three image masks to `True`.
  - Maps 6-D `observation/state` into `state`.
  - Maps 6-D `action` into `actions`.
  - Does not pad state/actions early; OpenPI pads later after normalization.

- `/home/caroline/openpi/src/openpi/training/config.py`
  - Added `import openpi.policies.quantycat_policy as quantycat_policy`.
  - Added `LeRobotQuantycatDataConfig`.
  - Registered `TrainConfig(name="pi05_quantycat", ...)`.
  - Points `repo_id` to the local LeRobot dataset.
  - Uses `make_bool_mask(5, -1)`.
    - Joints 0-4 are converted to delta targets.
    - Gripper dim 5 stays absolute.
  - Uses `pi0_config.Pi0Config(pi05=True, action_horizon=20)`.
  - Uses base checkpoint `gs://openpi-assets/checkpoints/pi05_base/params`.

- `/home/caroline/openpi/scripts/compute_norm_stats.py`
  - Patched output path logic so absolute local `repo_id` does not cause norm
    stats to be written into `my_data/input_data`.
  - For this config, stats now save to `my_data/training_pipeline/openpi/norm_stats.json`.

- `/home/caroline/openpi/src/openpi/training/data_loader.py`
  - Added a compatibility alias for LeRobot v2.1 parquet metadata:
    `_type: "List"` → HuggingFace `datasets.Sequence`.
  - This fixes the `ValueError: Feature type 'List' not found` failure with
    OpenPI's pinned `datasets==3.6.0`.

Reference copies live under:

```text
/home/caroline/quantycat-positronic/models/openpi/
```

Those files are documentation snapshots. The live source of truth for training
is `/home/caroline/openpi`.

---

## Run Scripts

### 1. Setup

```bash
bash models/openpi/run_scripts/setup.sh
```

What it does:

- Verifies `/home/caroline/openpi` exists.
- Verifies `uv` is installed.
- Runs `GIT_LFS_SKIP_SMUDGE=1 uv sync`.
- Runs `GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .`.

If `uv` is missing:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

or:

```bash
python -m pip install uv
```

### 2. Preprocess / Norm Stats

```bash
bash models/openpi/run_scripts/preprocess.sh
```

This runs:

```bash
cd /home/caroline/openpi
uv run scripts/compute_norm_stats.py --config-name pi05_quantycat
```

Expected output:

```text
/home/caroline/quantycat-positronic/my_data/training_pipeline/openpi/norm_stats.json
```

Do not write generated files into:

```text
/home/caroline/quantycat-positronic/my_data/input_data
```

That directory is the raw LeRobot dataset and should remain unchanged.

### 3. Training

```bash
bash models/openpi/run_scripts/training.sh
```

This runs:

```bash
cd /home/caroline/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_quantycat \
  --exp-name=screwdriver_so101_pi05_h20_v1 \
  --overwrite
```

Expected checkpoints:

```text
/home/caroline/quantycat-positronic/my_data/training_pipeline/openpi/checkpoints/pi05_quantycat/screwdriver_so101_pi05_h20_v1
```

---

## Current Training Config

Config name:

```text
pi05_quantycat
```

Important settings:

```python
model=pi0_config.Pi0Config(pi05=True, action_horizon=20)
weight_loader=weight_loaders.CheckpointWeightLoader(
    "gs://openpi-assets/checkpoints/pi05_base/params"
)
num_train_steps=5_000
batch_size=8
num_workers=2
save_interval=1000
keep_period=5000
wandb_enabled=False
```

Notes:

- `batch_size=8` is intentionally kept because this is the value chosen for
  the first run.
- On a 32GB RTX 5090, full π0.5 fine-tuning may OOM. If that happens, reduce
  `batch_size` first, likely to `4` or `1`.
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` is set by `training.sh`.

---

## Inference Input Format

Training uses `LeRobotQuantycatDataConfig.repack_transforms`. Inference does
not automatically run that training repack transform. The inference caller must
send observations with the keys expected by `QuantycatInputs`:

```python
{
    "observation/state": state,
    "observation/images/front": front_image,
    "observation/images/wrist": wrist_image,
    "prompt": "Put the screwdriver into the cup",
}
```

`QuantycatInputs` then builds the OpenPI model image slots:

```python
{
    "base_0_rgb": front_image,
    "left_wrist_0_rgb": wrist_image,
    "right_wrist_0_rgb": wrist_image,
}
```

---

## Output Locations

OpenPI work root:

```text
/home/caroline/quantycat-positronic/my_data/training_pipeline/openpi
```

Norm stats:

```text
/home/caroline/quantycat-positronic/my_data/training_pipeline/openpi/norm_stats.json
```

Checkpoints:

```text
/home/caroline/quantycat-positronic/my_data/training_pipeline/openpi/checkpoints/pi05_quantycat/<exp-name>
```

Default experiment:

```text
/home/caroline/quantycat-positronic/my_data/training_pipeline/openpi/checkpoints/pi05_quantycat/screwdriver_so101_pi05_h20_v1
```

---

## Known Issues / Fixes

### HuggingFace `List` Feature Error

If norm stats fails with:

```text
ValueError: Feature type 'List' not found
```

the fix is already patched in:

```text
/home/caroline/openpi/src/openpi/training/data_loader.py
```

It aliases LeRobot v2.1 parquet metadata `_type: "List"` to
`datasets.Sequence`.

### Norm Stats Saved Into `input_data`

OpenPI's original `compute_norm_stats.py` used:

```python
output_path = config.assets_dirs / data_config.repo_id
```

With an absolute local `repo_id`, that accidentally resolved to
`my_data/input_data`. This is patched so `pi05_quantycat` writes to:

```text
my_data/training_pipeline/openpi/norm_stats.json
```

### Local Dataset Path

The config uses a local LeRobot dataset path:

```python
repo_id="/home/caroline/quantycat-positronic/my_data/input_data"
```

This has been verified far enough to load parquet files and compute norm stats.

---

## Minimal Checklist

```bash
cd /home/caroline/quantycat-positronic

# 1. Install/sync openpi environment.
bash models/openpi/run_scripts/setup.sh

# 2. Compute normalization stats.
bash models/openpi/run_scripts/preprocess.sh

# 3. Train.
bash models/openpi/run_scripts/training.sh
```
