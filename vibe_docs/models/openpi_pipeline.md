Robot demos → preprocess → OpenPI π0.5 LoRA fine-tune → trained checkpoint → inference on SO-101.

This pipeline fine-tunes Physical Intelligence `openpi` on the local Quantycat
SO-101 screwdriver dataset. Unlike the RynnVLA pipeline, the input LeRobot
dataset is used directly: there is no conversion into per-episode image/action
folders and no pretokenization step.

---

## Full Pipeline

```
my_data/input_data/                         (LeRobot v2.1 format — raw, never modified)
    data/chunk-000/episode_*.parquet
    videos/chunk-000/observation.images.front/*.mp4
    videos/chunk-000/observation.images.wrist/*.mp4
    meta/*.json*
        ↓  pipeline.sh (models/preprocessing_data/)
my_data/clean_input_data/                         (trimmed, pauses removed, actions smoothed)
        ↓  setup.sh
vendor/openpi/.venv                         (uv-managed openpi environment)
        ↓  preprocess.sh
models/openpi/training_pipeline/norm_stats.json
        ↓  training.sh
models/openpi/training_pipeline/checkpoints/pi05_quantycat_lora/05232026_pi05_lora/
        ↓  inference.sh
SO-101 robot
```

---

## Key Commands

From the repo root:

```bash
cd /home/caroline/quantycat-positronic

# 0. Preprocess raw data.
bash models/preprocessing_data/pipeline.sh \
    --src my_data/input_data \
    --dst my_data/clean_input_data \
    --trim-frames 165 \
    --remove-episodes "45" \
    --sigma 1.5

# 1. Install/sync openpi environment (once per machine).
bash models/openpi/run_scripts/setup.sh

# 2. Compute normalization stats.
bash models/openpi/run_scripts/preprocess.sh

# 3. Train.
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

Raw input (never modified):

```text
/home/caroline/quantycat-positronic/my_data/input_data
```

Preprocessed training input:

```text
/home/caroline/quantycat-positronic/my_data/clean_input_data
```

Schema:

- Robot: SO-101 follower arm
- Task: `Put the screwdriver into the cup`
- Episodes: 49 (episode 45 removed — failed grasp)
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

## Preprocessing Pipeline

Produces `clean_data` from raw `input_data`. Runs three steps in order:

1. **Trim**: removes 165-frame countdown hold from every episode start; drops episode 45.
2. **Remove pauses**: removes runs of ≥15 consecutive near-zero-motion frames (L2 arm delta < 0.01 rad/frame).
3. **Smooth actions**: Gaussian filter (σ=1.5) on joints 0–4; gripper untouched.

```bash
bash models/preprocessing_data/pipeline.sh \
    --src my_data/input_data \
    --dst my_data/clean_input_data \
    --trim-frames 165 \
    --remove-episodes "45" \
    --sigma 1.5

# Preview without writing:
bash models/preprocessing_data/pipeline.sh \
    --src my_data/input_data \
    --dst my_data/clean_input_data \
    --trim-frames 165 \
    --remove-episodes "45" \
    --dry-run
```

See `models/preprocessing_data/README.md` for per-script documentation and threshold tuning.

**Video timestamp fix**: `remove_pauses.py` uses `setpts=N/{FPS}/TB` + `-vsync cfr` to produce uniform sequential PTS after dropping frames. Earlier versions using `setpts=PTS-STARTPTS` + `-vsync 0` preserved original PTS, creating gaps that caused LeRobot's timestamp validator to fail during norm stats computation.

---

## What Was Added To OpenPI

The patched OpenPI checkout is at:

```text
/home/caroline/quantycat-positronic/vendor/openpi
```

(symlinked from `/home/caroline/openpi`)

Changed files:

- `src/openpi/policies/quantycat_policy.py`
  - New SO-101 policy transform.
  - Maps `observation/images/front` to `base_0_rgb`.
  - Maps `observation/images/wrist` to `left_wrist_0_rgb`.
  - Duplicates the wrist camera into `right_wrist_0_rgb`.
  - Sets all three image masks to `True`.
  - Maps 6-D `observation/state` into `state`.
  - Maps 6-D `action` into `actions`.

- `src/openpi/training/config.py`
  - Added `LeRobotQuantycatDataConfig` and `LeRobotQuantycatLoraDataConfig`.
  - Registered `TrainConfig(name="pi05_quantycat_lora", ...)` — active LoRA config.
  - Points `repo_id` to the local preprocessed dataset (`my_data/clean_input_data`).
  - Uses `make_bool_mask(5, -1)`: joints 0–4 converted to delta targets, gripper dim 5 stays absolute.
  - Uses `pi0_config.Pi0Config(pi05=True, action_horizon=20)`.
  - Uses base checkpoint `gs://openpi-assets/checkpoints/pi05_base/params`.

- `scripts/compute_norm_stats.py`
  - Patched output path logic so absolute local `repo_id` writes stats to:
    `models/openpi/training_pipeline/norm_stats.json` (not into `input_data`).

- `src/openpi/training/data_loader.py`
  - Compatibility alias: `_type: "List"` → `datasets.Sequence` for LeRobot v2.1 parquet metadata.
  - Fixes `ValueError: Feature type 'List' not found` with OpenPI's pinned `datasets==3.6.0`.

A readable copy of the policy transform is kept at:

```text
models/openpi/training_config/quantycat_policy.py
```

---

## Run Scripts

### 1. Setup

```bash
bash models/openpi/run_scripts/setup.sh
```

What it does:

- Verifies `vendor/openpi` exists.
- Verifies `uv` is installed.
- Runs `GIT_LFS_SKIP_SMUDGE=1 uv sync`.
- Runs `GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .`.

If `uv` is missing:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

### 2. Preprocess / Norm Stats

```bash
bash models/openpi/run_scripts/preprocess.sh
```

This runs:

```bash
cd vendor/openpi
uv run scripts/compute_norm_stats.py --config-name pi05_quantycat_lora
```

Expected output:

```text
/home/caroline/quantycat-positronic/models/openpi/training_pipeline/norm_stats.json
```

### 3. Training

```bash
bash models/openpi/run_scripts/training.sh
```

This runs:

```bash
cd vendor/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_quantycat_lora \
  --exp-name=05232026_pi05_lora \
  --overwrite
```

Expected checkpoints:

```text
/home/caroline/quantycat-positronic/models/openpi/training_pipeline/checkpoints/pi05_quantycat_lora/05232026_pi05_lora/
```

---

## Current Training Config

Config name: `pi05_quantycat_lora`

Important settings:

```python
model=pi0_config.Pi0Config(pi05=True, action_horizon=20)
weight_loader=weight_loaders.CheckpointWeightLoader(
    "gs://openpi-assets/checkpoints/pi05_base/params"
)
num_train_steps=5_000
batch_size=4
num_workers=2
save_interval=1000
keep_period=5000
wandb_enabled=False
```

Notes:

- LoRA fine-tune (`pi05_quantycat_lora`) rather than full fine-tune. Reduces memory and training time.
- `batch_size=4` for the RTX 5090 32GB. The full fine-tune config (`pi05_quantycat`) OOMed at batch size 8.
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` is set by `training.sh`.
- Checkpoints saved every 1000 steps; only the most recent kept-period checkpoint is retained.

---

## Live Deployment

> **Inference scripts have moved to `quantycat-iron-fleet`.**
>
> All commands below should be run from `~/quantycat-iron-fleet`.
> See `vibe_docs/inference.md` there for the full guide.

### Config File

```text
~/quantycat-iron-fleet/configs/openpi_so101.json
```

Key defaults (as of 2026-05-23):

| Parameter | Value | Notes |
|---|---|---|
| `config_name` | `pi05_quantycat_lora` | matches training config |
| `prompt` | `Put the screwdriver into the cup` | |
| `sample_steps` | `10` | diffusion steps at inference |
| `max_steps` | `0` | infinite loop (Ctrl+C to stop) |
| `execute_steps_per_inference` | `10` | action chunk steps before re-inferring |
| `control_period_s` | `0.033` | 1/30 fps — matches training fps |
| `gain_vector` | `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]` | recalibrate after first run |
| `max_delta_per_command_deg` | `[4, 4, 6, 4, 4, 2]` | per-joint safety clip |

### Check / Dry-Run / Live

```bash
cd ~/quantycat-iron-fleet

# Fast config check (no model load).
bash scripts/run_openpi.sh --check-only --skip-policy-load

# Full policy-load check.
bash scripts/run_openpi.sh --check-only

# Dry-run: connects to robot, runs policy, applies gain/clips, but does NOT send actions.
bash scripts/run_openpi.sh --dry-run --max-steps 5

# Live run with step cap.
bash scripts/run_openpi.sh --max-steps 60

# Infinite live run.
bash scripts/run_openpi.sh
```

### Self-Start Workaround

From the default rest pose the model may predict near-hold actions. Pre-position
the arm to an in-trajectory pose before handing control:

```bash
cd ~/quantycat-iron-fleet
python scripts/test_motors.py --start-pose 4 -85 92 67 6 0.4 --wait 3.0
bash scripts/run_openpi.sh
```

The v2 checkpoint (trained on `clean_data` with hold trimmed, pauses removed,
actions smoothed) should self-start more reliably than the v1 checkpoint.

### Action Convention

```text
delta = predicted_target - current_state
calibrated_target = current_state + delta * gain_vector
```

After gain, the runner clips per-command deltas and final absolute targets,
converts back to live robot units (degrees), and sends the command dictionary.

### Logs

Each rollout writes to `run_logs/openpi/` inside iron-fleet:

```text
~/quantycat-iron-fleet/run_logs/openpi/<timestamp>/deployment_config.json
~/quantycat-iron-fleet/run_logs/openpi/<timestamp>/rollout.jsonl
```

---

## Inference Input Format

Inference callers must send observations with the keys expected by `QuantycatInputs`:

```python
{
    "observation/state": state,          # 6-D joint/gripper positions
    "observation/images/front": front,   # (H, W, 3) uint8
    "observation/images/wrist": wrist,   # (H, W, 3) uint8
    "prompt": "Put the screwdriver into the cup",
}
```

`QuantycatInputs` builds the OpenPI model image slots internally:

```python
{
    "base_0_rgb": front,
    "left_wrist_0_rgb": wrist,
    "right_wrist_0_rgb": wrist,   # wrist duplicated — SO-101 has no right wrist camera
}
```

---

## Output Locations

OpenPI work root:

```text
/home/caroline/quantycat-positronic/models/openpi/training_pipeline/
```

Norm stats:

```text
models/openpi/training_pipeline/norm_stats.json
```

Checkpoints:

```text
models/openpi/training_pipeline/checkpoints/pi05_quantycat_lora/<exp-name>/
```

Current experiment:

```text
models/openpi/training_pipeline/checkpoints/pi05_quantycat_lora/05232026_pi05_lora/
```

---

## Known Issues / Fixes

### HuggingFace `List` Feature Error

If norm stats fails with:

```text
ValueError: Feature type 'List' not found
```

the fix is already patched in `vendor/openpi/src/openpi/training/data_loader.py`.
It aliases LeRobot v2.1 parquet metadata `_type: "List"` to `datasets.Sequence`.

### Norm Stats Saved Into `input_data`

OpenPI's original `compute_norm_stats.py` used:

```python
output_path = config.assets_dirs / data_config.repo_id
```

With an absolute local `repo_id`, that accidentally resolved into `my_data/input_data`.
Patched so `pi05_quantycat_lora` writes to `models/openpi/training_pipeline/norm_stats.json`.

### Video/Parquet Timestamp Mismatch After Pause Removal

If preprocess fails with:

```text
AssertionError: One or several query timestamps unexpectedly violate the tolerance
```

the fix is in `models/preprocessing_data/remove_pauses.py`. Uses `setpts=N/{FPS}/TB`
+ `-vsync cfr` + `-r {FPS}` to produce uniform sequential PTS. The original
`setpts=PTS-STARTPTS` + `-vsync 0` preserved gaps where pauses were removed.

---

## Minimal Checklist

```bash
# --- Training (in quantycat-positronic) ---
cd /home/caroline/quantycat-positronic

# 0. Preprocess raw recordings.
bash models/preprocessing_data/pipeline.sh \
    --src my_data/input_data --dst my_data/clean_input_data \
    --trim-frames 165 --remove-episodes "45" --sigma 1.5

# 1. Install/sync openpi environment.
bash models/openpi/run_scripts/setup.sh

# 2. Compute normalization stats.
bash models/openpi/run_scripts/preprocess.sh

# 3. Train.
bash models/openpi/run_scripts/training.sh

# --- Inference (in quantycat-iron-fleet) ---
cd /home/caroline/quantycat-iron-fleet

# 4. Check deployment config, then dry-run.
bash scripts/run_openpi.sh --check-only
bash scripts/run_openpi.sh --dry-run --max-steps 5
```
