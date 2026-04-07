Robot demos → RynnVLA-002 fine-tune → trained checkpoint → inference on SO-101.

RynnVLA-002 has two components:
- **VLA Model (256×256)** — text + images → actions. This is what we fine-tune.
- **World Model (512×512)** — images + actions → next image frame. Optional, not needed for SO-101 inference.

---

## Forked repo

We forked the original to [github.com/QuantyCat/RynnVLA-002](https://github.com/QuantyCat/RynnVLA-002) to store dataset-specific changes. Lives at `~/RynnVLA-002/` on the training machine.

Three files changed from the original:

**`data_lerobot/item_processor.py`** — updated `MIN/MAX_VALUES_ACTION` and `MIN/MAX_VALUES_STATE` from LIBERO values to actual screwdriver dataset values. Used to normalize joint values to [-1, 1] during pretokenization.

**`eval_solver_lerobot_action_head_state.py`** — updated `unnorm_min_max()` with the same action min/max. Inference-time mirror — denormalizes predicted tokens back into real joint values before sending to the robot.

**`data_lerobot/pretoken_lerobot_state.py`** — two fixes:
- Subprocess Python invocation changed from hardcoded `"python"` to `sys.executable` so workers inherit the conda environment
- Worker count changed from hardcoded `32` to `num_gpus * 16` — tuned for RTX 5090 (each worker uses ~1.15GB, 16 workers ≈ 18GB)

---

## Full pipeline

```
my_data/input_data/                     (LeRobot v2.1 format — never modified)
    ↓  Step 1 — step1_convert_lerobot.py
my_data/training_pipeline/training_data/
    TASK_NAME/episode_000000/
        front_image/image_0.png ...
        wrist_image/image_0.png ...
        state/state_0.npy               shape (6,) joint positions
        abs_action/action_0/            relative actions, gripper absolute
            0.npy ... 19.npy            20 sub-actions per chunk
    ↓  Step 2 — step2_generate_conversations.py
my_data/training_pipeline/conversations/
    libero_<task_label>_his_<his>_train_img_state_abs_ck_1_<resolution>.json
    ↓  Step 3 — step3_verify.py
    ↓  Step 4 — step4_calculate_min_max.py
my_data/training_pipeline/min_max_action.txt
my_data/training_pipeline/min_max_state.txt
    ↓  Step 5 — step5_pretokenize.py
my_data/training_pipeline/tokens/vla_data/
    files/*.pkl
    ↓  Step 6 — step6_merge_records.py
my_data/training_pipeline/tokens/vla_data/record.json
    ↓  Step 7 — step7_update_train_config.py
~/RynnVLA-002/rynnvla-002/configs/lerobot/his_1_third_view_wrist_w_state_20_256_pretokenize.yaml
    ↓  Fine-tune — finetune.py
my_data/training_pipeline/fine_tuning/<task_label>_<robot>/
    ↓  Inference on SO-101
```

---

## Key Commands To Run

```bash
source models/rynnvla-002/run_scripts/setup.sh
source models/rynnvla-002/run_scripts/preprocess.sh
bash models/rynnvla-002/run_scripts/finetune.sh
```

---

## Config

All parameters live in `models/rynnvla-002/config.yaml`:

```yaml
input_dir: my_data/input_data
work_dir: my_data/training_pipeline
task_label: screwdriver    # short label for output filenames
robot: so101               # robot name — used in fine_tuning output folder

# Must be consistent across preprocessing, training, and inference
chunk_size: 20             # sub-actions per chunk (lerobot default — must match time_horizon)
action_dim: 6              # SO-101 has 6 joints
resolution: 256            # VLA model uses 256x256
his: 1                     # history length

# Training — all from RynnVLA-002 original paper
batch_size: 8              # reduce to 4 if OOM
num_workers: 16            # CPU workers for data loading
epochs: 40
lr: 5e-6
```

---

## Prerequisites — download model weights

Run once on the training machine before preprocessing:

```bash
source .env  # loads HF_TOKEN
python3 models/rynnvla-002/run_scripts/download_weights.py
```

Downloads to `~/RynnVLA-002/rynnvla-002/ckpts/`:
- `chameleon/tokenizer` — raw Chameleon tokenizer (vqgan.ckpt etc.)
- `chameleon/base_model` — Chameleon 7B weights in HuggingFace format (~14GB)
- `starting_point` — RynnVLA-002 pretrained checkpoint (~14GB)

---

## Steps 1–7 — Preprocessing

```bash
source models/rynnvla-002/run_scripts/setup.sh
source models/rynnvla-002/run_scripts/preprocess.sh
```

`preprocess.sh` runs all steps in order and skips any step whose output already exists:

1. Convert LeRobot dataset → per-episode `training_data/` folders
2. Generate conversation JSON from `training_data/`
3. Verify outputs
4. Calculate action and state min/max values → saved to `min_max_action.txt` / `min_max_state.txt`
5. Pretokenize — converts every episode frame into a `.pkl` token file
6. Merge per-worker records → `record.json`
7. Update training config YAML with `record.json` path

**What pretokenization does:** each frame is converted into a flat token sequence. Three things per frame:
- **Images** — front + wrist frames run through VQ-GAN, compressed into discrete tokens (slow GPU step)
- **Actions** — joint movements bucketed into discrete bins, encoded as tokens
- **States** — same for joint positions

Workers split frames evenly and run in parallel (`num_gpus × 16` workers). Logs go to `tokens/vla_data/logs/worker_N.log`. Each worker uses ~1.15GB VRAM — 16 workers ≈ 18GB on a 32GB card. Estimated time on RTX 5090: ~30–45 min for a ~36k frame dataset.

---

## Fine-tune

```bash
source models/rynnvla-002/run_scripts/setup.sh
bash models/rynnvla-002/run_scripts/finetune.sh
```

This calls `models/rynnvla-002/fine_tuning/finetune.py`, which reads all parameters from `config.yaml` and launches torchrun.

Checkpoint output: `$work_dir/fine_tuning/<task_label>_<robot>/`
Training log: `$work_dir/fine_tuning/<task_label>_<robot>/output.log`

**Key training flags:**

| Flag | Value | Source |
|---|---|---|
| `--action_dim` | 6 | config.yaml — SO-101 has 6 joints |
| `--time_horizon` | 20 | config.yaml — lerobot chunk size |
| `--init_from` | `ckpts/starting_point` | pretrained RynnVLA-002 checkpoint |
| `--epochs` | 40 | config.yaml |
| `--lr` | 5e-6 | config.yaml |
| `--batch_size` | 8 | config.yaml — reduce to 4 if OOM |
| `--checkpointing` | enabled | saves VRAM by recomputing activations |
| `--ckpt_max_keep` | 3 | keep last 3 checkpoints |

If you get CUDA OOM at startup, reduce `batch_size` to 4 in `config.yaml` and rerun.

---

## Inference on SO-101

```python
from rynnvla002 import Solver
solver = Solver(resume_path="path/to/your/checkpoint")

task = "Put the screwdriver into the cup"
while True:
    obs = robot.get_observation()
    action = solver.get_action_wrist_action_head_state(
        front_image=obs["observation.images.front"],
        wrist_image=obs["observation.images.wrist"],
        state=obs["observation.state"],
        prompt=task
    )
    robot.send_action(action)
```

---

## Hardware

```
robot port:   /dev/cu.usbmodem5B140331511
teleop port:  /dev/cu.usbmodem5B141144971

cameras:
  front:  index 1 — 640x360 @ 30fps
  wrist:  index 0 — 640x360 @ 30fps

training machine: caroline@100.83.46.36
```
