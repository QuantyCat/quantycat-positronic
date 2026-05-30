Robot demos → imitation learning → trained checkpoint.

**The core goal:** take a robot dataset recorded in LeRobot format → train a model → output a checkpoint that a separate inference repo can load and run on the robot.

This repo handles preprocessing, training, and offline eval. Running a checkpoint on a real robot lives in `quantycat-iron-fleet`.

---

## Repo Structure

```
quantycat-positronic/
  models/              ← training only: preprocessing, fine-tuning, run scripts
    act/
    lerobot/
    openpi/
    rynnvla-002/
  eval/                ← offline eval: run a checkpoint against saved episode data
    rynnvla_002/
      data_analysis/
    openpi/
      data_analysis/
      gpu_exploration/
  preprocessing_data/  ← shared dataset transforms (trim, pause removal, smoothing)
  vendor/              ← vendored third-party libraries (openpi)
```

---

## Architecture

```
STAGE 1 — Preprocessing
  Convert LeRobot dataset into model-specific training format.
  Each model owns this step entirely.
  Reads:   $INPUT_DIR   (LeRobot format — never modified)
  Writes:  $WORK_DIR/
                |
                v
STAGE 2 — Train
  models/$MODEL_TYPE/run_scripts/training.sh
  Reads:   $WORK_DIR/dataset/
  Method:  defined entirely by the model folder
  Writes:  $OUTPUT_DIR/checkpoints/
           $OUTPUT_DIR/checkpoints/last/  <- hand this to iron-fleet
                |
                v
STAGE 3 — Offline Eval (optional)
  eval/$MODEL_TYPE/episode_batch_eval.py
  Runs checkpoint against saved training episodes.
  Checks sign agreement, MAE, correlation vs ground-truth action chunks.
  Does not require robot hardware.
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

| Model | Training folder | Eval folder | Pipeline doc |
|---|---|---|---|
| ACT | `models/act/` | — | `vibe_docs/models/act_pipeline.md` |
| OpenPI π0.5 | `models/openpi/` | `eval/openpi/` | `vibe_docs/models/openpi_pipeline.md` |
| LeRobot pi05 LoRA | `models/lerobot/` | `eval/openpi/` | `vibe_docs/models/lerobot_pipeline.md` |
| RynnVLA-002 | `models/rynnvla-002/` | `eval/rynnvla_002/` | `vibe_docs/models/rynnvla002_pipeline.md` |

Adding a new model = one new folder under `models/` and one under `eval/`.

---

## Hardware

```
robot port:   /dev/cu.usbmodem5B140331511
teleop port:  /dev/cu.usbmodem5B141144871

cameras:
  front:  index 1 — 640x360 @ 30fps
  wrist:  index 0 — 640x360 @ 30fps

training machine: caroline@100.83.46.36
```
