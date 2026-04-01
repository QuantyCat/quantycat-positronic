Robot demos → imitation learning → trained checkpoint.

**The core goal:** take a robot dataset recorded in LeRobot format → train a model → output a checkpoint that a separate inference repo can load and run on the robot.

This repo handles everything up to and including the trained checkpoint. Running the checkpoint on a real robot is out of scope — that lives in a separate inference repo.

## The layers

- **Model folder** (`models/$MODEL_TYPE/`) — everything specific to one model: how to prepare its dataset and how to train it. One folder per model.

Adding a new model = one new folder under `models/`.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — Prepare model dataset                                         │
│  models/$MODEL_TYPE/build_dataset.py                                     │
│                                                                          │
│  Converts the LeRobot input dataset into whatever format this model      │
│  needs. Each model folder owns this step entirely.                       │
│                                                                          │
│  Reads:   $INPUT_DIR                                                     │
│  Writes:  $WORK_DIR/dataset/                                             │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE 2 — Train                                                         │
│  models/$MODEL_TYPE/train.sh                                             │
│                                                                          │
│  Reads:   $WORK_DIR/dataset/                                             │
│  Method:  defined entirely by the model folder                           │
│                                                                          │
│  Writes:  $OUTPUT_DIR/checkpoints/                                       │
│           $OUTPUT_DIR/checkpoints/last/  ← hand this to inference repo  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Input format

The input is a **LeRobot dataset**:

```
$INPUT_DIR/
  meta/info.json     dataset metadata (version, fps, features)
  data/              parquet files with timeseries
  videos/            mp4 files per camera
```

---

## `models/act/`

ACT trains via `lerobot-train`. It uses:
- `observation.state` — current joint positions
- `observation.images.*` — all cameras in the dataset
- `action` — target joint positions to predict

**`build_dataset.py`** — copies the input dataset to `$WORK_DIR/dataset` and upgrades it from v2.1 to v3.0 if needed. Input is never modified.

**`train.sh`** — calls `lerobot-train --policy.type=act`. Reads env vars: `$WORK_DIR`, `$OUTPUT_DIR`, `$TRAIN_STEPS`, `$BATCH_SIZE`, `$CHUNK_SIZE`, `$KL_WEIGHT`, `$SAVE_STEPS`.

**`README.md`** — documents the checkpoint format and `$CHUNK_SIZE` used at training time. The inference repo must use the same chunk size.

---

## Commands

```bash
export INPUT_DIR=       # path to your LeRobot dataset
export WORK_DIR=        # pipeline working directory
export OUTPUT_DIR=      # training output directory (e.g. my_data/training/model)
export MODEL_TYPE=      # folder name under models/ (e.g. act)

# Training (ACT — see models/act/train.sh for defaults)
export TRAIN_STEPS=     # total training steps       (default: 100000)
export BATCH_SIZE=      # batch size                 (default: 8)
export CHUNK_SIZE=      # action chunk size          (default: 100)
export KL_WEIGHT=       # KL loss weight             (default: 10)
export SAVE_STEPS=      # checkpoint save frequency  (default: 10000)

```

```bash
# Stage 1 — prepare dataset
python3 models/$MODEL_TYPE/build_dataset.py \
  --input-dir $INPUT_DIR \
  --work-dir $WORK_DIR

# Stage 2 — train
bash models/$MODEL_TYPE/train.sh
```

---

## Hard constraints

- `$CHUNK_SIZE` at training time must be documented in `models/act/README.md` — the inference repo must use the same value.

---

## Hardware config

```
robot port:    /dev/cu.usbmodem5B140331511
teleop port:   /dev/cu.usbmodem5B141144971

cameras:
  front:   index 1 — 640x360 @ 30fps
  wrist:   index 0 — 640x360 @ 30fps
```

---

## Inference

Copy checkpoint from training machine to local Mac:

```bash
scp -r caroline@100.83.46.36:~/quantycat-positronic/my_data/training/model/checkpoints/last/pretrained_model /Users/carolineshouraboura/Desktop/
```

Run inference:

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/cu.usbmodem5B140331511 \
    --robot.id=so101_follower \
    --robot.cameras="{front: {type: opencv, index_or_path: 1, width: 640, height: 360, fps: 30}, wrist: {type: opencv, index_or_path: 0, width: 640, height: 360, fps: 30}}" \
    --display_data=true \
    --dataset.repo_id=local/eval_screwdriver \
    --dataset.num_episodes=10 \
    --dataset.single_task="Put the screwdriver into the cup" \
    --dataset.push_to_hub=false \
    --policy.path=/Users/carolineshouraboura/Desktop/pretrained_model \
    --policy.device=mps

<!-- TODO: remove these temp notes
# kill inference
pkill -f lerobot-record

# clear eval dataset
rm -rf ~/.cache/huggingface/lerobot/local/eval_screwdriver

# resume claude chat
claude --resume 79feca52-9e8a-46c7-a59f-f04ac90ddc62
-->
```
