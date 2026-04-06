Robot demos → ACT fine-tune → trained checkpoint → inference on SO-101.

ACT (Action Chunking with Transformers) trains via `lerobot-train`. It uses:
- `observation.state` — current joint positions
- `observation.images.*` — all cameras in the dataset
- `action` — target joint positions to predict

---

## Full pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — Prepare dataset                                               │
│  models/act/build_dataset.py                                             │
│                                                                          │
│  Copies the LeRobot input dataset to $WORK_DIR/dataset and upgrades      │
│  from v2.1 to v3.0 if needed. Input is never modified.                  │
│                                                                          │
│  Reads:   $INPUT_DIR                                                     │
│  Writes:  $WORK_DIR/dataset/                                             │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE 2 — Train                                                         │
│  models/act/train.sh                                                     │
│                                                                          │
│  Calls lerobot-train --policy.type=act                                   │
│                                                                          │
│  Reads:   $WORK_DIR/dataset/                                             │
│  Writes:  $OUTPUT_DIR/checkpoints/                                       │
│           $OUTPUT_DIR/checkpoints/last/  ← hand this to inference       │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Commands

```bash
export INPUT_DIR=my_data/input_data
export WORK_DIR=my_data/training_pipeline
export OUTPUT_DIR=my_data/training/model
export MODEL_TYPE=act

# Optional overrides (defaults shown)
export TRAIN_STEPS=100000
export BATCH_SIZE=8
export CHUNK_SIZE=100
export KL_WEIGHT=10
export SAVE_STEPS=10000

# Stage 1 — prepare dataset
python3 models/$MODEL_TYPE/build_dataset.py \
  --input-dir $INPUT_DIR \
  --work-dir $WORK_DIR

# Stage 2 — train
bash models/$MODEL_TYPE/train.sh
```

---

## Hard constraints

- `$CHUNK_SIZE` at training time must match `$CHUNK_SIZE` at inference time. It's documented in `models/act/README.md`.

---

## Inference on SO-101

Copy checkpoint from training machine to Mac:

```bash
scp -r caroline@100.83.46.36:~/quantycat-positronic/my_data/training/model/checkpoints/last/pretrained_model /Users/carolineshouraboura/Desktop/
```

Run:

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
```


