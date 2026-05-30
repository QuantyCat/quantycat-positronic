Robot demos → LeRobot v3.0 dataset → pi05 LoRA fine-tune → trained checkpoint.

This pipeline fine-tunes Physical Intelligence `pi05` via the LeRobot framework on the
Quantycat SO-101 screwdriver dataset. The dataset is used directly in LeRobot v3.0 format
with no conversion step.

---

## Full Pipeline

```
~/quantycat-data/datasets/screwdriver_so101/   (LeRobot v3.0 — never modified)
    data/chunk-*/file-*.parquet
    videos/observation.images.front/chunk-000/file-000.mp4
    videos/observation.images.wrist/chunk-000/file-000.mp4
    meta/stats.json                            ← must have relative action stats if use_relative_actions=true
        ↓  setup.sh  (creates vendor/lerobot/.venv, installs lerobot[pi]==0.5.1 + peft)
vendor/lerobot/.venv
        ↓  training.sh
~/quantycat-data/checkpoints/lerobot/pi05/<exp-name>/
```

Data location is controlled by `QUANTYCAT_DATA_HOME` (default: `~/quantycat-data`).

---

## Key Commands

From the repo root:

```bash
cd /home/caroline/quantycat-positronic

# 1. Install lerobot venv (once per machine — skips if venv already exists).
bash models/lerobot/run_scripts/setup.sh

# 2. Recompute dataset stats (run once after recording or modifying dataset).
vendor/lerobot/.venv/bin/lerobot-edit-dataset \
    --operation.type=recompute_stats \
    --repo_id=screwdriver_so101 \
    --root=/home/caroline/quantycat-data/datasets

# 3. Train.
DATASET_REPO_ID=screwdriver_so101 bash models/lerobot/run_scripts/training.sh
```

Use `tmux` for training so it survives disconnects:

```bash
tmux new -s lerobot_train
cd /home/caroline/quantycat-positronic
DATASET_REPO_ID=screwdriver_so101 bash models/lerobot/run_scripts/training.sh
```

---

## Dataset

Location:

```text
~/quantycat-data/datasets/screwdriver_so101/
```

Schema:

- Format: LeRobot v3.0
- Robot: SO-101 follower arm
- Task: `Put the screwdriver into the cup`
- Episodes: 50
- FPS: 30
- Cameras:
  - `observation.images.front` (640×360, av1)
  - `observation.images.wrist` (640×360, av1)
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

## Setup

```bash
bash models/lerobot/run_scripts/setup.sh
```

Creates `vendor/lerobot/.venv` with Python 3.12 and installs `lerobot[pi]==0.5.1` and `peft`.
Skips creation if the venv already exists.

Requires `uv`. If missing:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

---

## Dataset Stats

LeRobot stores normalization stats in `meta/stats.json`. These are computed from the raw
dataset values (absolute joint positions). Recompute after any dataset change:

```bash
vendor/lerobot/.venv/bin/lerobot-edit-dataset \
    --operation.type=recompute_stats \
    --repo_id=screwdriver_so101 \
    --root=/home/caroline/quantycat-data/datasets
```

**Important:** The lerobot pi05 preprocessing pipeline runs in this order:
`raw action → relative conversion → normalize`. If `use_relative_actions=true` is set,
`stats.json` must contain stats for the **relative** (delta) actions, not absolute positions.
Pass `--operation.relative_action=true --operation.relative_exclude_joints='["gripper.pos"]'`
to `lerobot-edit-dataset` in that case.

---

## Training

```bash
DATASET_REPO_ID=screwdriver_so101 bash models/lerobot/run_scripts/training.sh
```

Environment variable overrides:

| Variable | Default | Purpose |
|---|---|---|
| `DATASET_REPO_ID` | `screwdriver_so101` | Dataset folder name under `HF_LEROBOT_HOME` |
| `EXP_NAME` | `<date_time>_pi05_lerobot` | Run name for W&B and checkpoint folder |
| `LEROBOT_VENV` | `vendor/lerobot/.venv` | Path to the lerobot venv |

Current training config:

| Parameter | Value |
|---|---|
| Policy | `pi05` |
| Pretrained base | `lerobot/pi05_base` |
| PEFT | LoRA (`--peft.method_type=LORA`) |
| Trainable params | ~1.3M of 4.1B (0.03%) |
| dtype | bfloat16 |
| Gradient checkpointing | true |
| Compile model | true |
| Steps | 3000 |
| Batch size | 2 |
| W&B | enabled |

---

## Output Locations

Checkpoints:

```text
~/quantycat-data/checkpoints/lerobot/pi05/<exp-name>/
```

The checkpoint directory contains a `pretrained_model/` subfolder with:
- `adapter_config.json` — LoRA adapter config
- `adapter_model.safetensors` — LoRA weights
- `policy_preprocessor.json` — preprocessor pipeline config
- `policy_postprocessor_step_0_unnormalizer_processor.safetensors` — normalization stats used during training

---

## Eval

```bash
vendor/lerobot/.venv/bin/python models/lerobot/eval/lerobot_lora_high_motion_eval.py \
    --checkpoint ~/quantycat-data/checkpoints/lerobot/pi05/<exp-name>/checkpoints/<step>/pretrained_model
```

Mirrors `eval/openpi/lerobot_high_motion_eval.py`: selects top-20 high-motion windows per joint,
runs the policy, and computes sign agreement / correlation against ground-truth achieved deltas.

---

## Known Issues / Notes

### OOM Without LoRA

Full fine-tuning of pi05 on a 32 GB GPU OOMs during the first optimizer step (Adam allocates
state for 4B parameters). Always use `--peft.method_type=LORA`.

### compile_model and CUDA Graphs

`--policy.compile_model=true` allocates ~4 GiB in CUDA Graph private pools but significantly
improves throughput (~8.7 steps/sec on RTX 5090). Keep enabled unless OOMing.

### loss_per_dim Warning

```text
WandB logging of key "loss_per_dim" was ignored as its type "<class 'list'>" is not handled
```

Harmless — lerobot passes a list to wandb which expects a scalar. Does not affect training.
