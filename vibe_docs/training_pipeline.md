Robot demos → imitation learning → trained checkpoint.

**The core goal:** take a robot dataset recorded in LeRobot format → train a model → output a checkpoint that a separate inference repo can load and run on the robot.

This repo handles everything up to and including the trained checkpoint. Running the checkpoint on a real robot is out of scope — that lives in a separate inference repo.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — Preprocessing                                                 │
│                                                                          │
│  Convert LeRobot dataset into model-specific training format.            │
│  Each model owns this step entirely.                                     │
│                                                                          │       │
│                                                                          │
│  Reads:   $INPUT_DIR   (LeRobot format — never modified)                 │
│  Writes:  $WORK_DIR/                                                     │
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

## Models

| Model | Folder | Pipeline doc |
|---|---|---|
| ACT | `models/act/` | `vibe_docs/models/act_pipeline.md` |
| RynnVLA-002 | `models/rynnvla-002/` | `vibe_docs/models/rynnvla002_pipeline.md` |

Adding a new model = one new folder under `models/`.

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