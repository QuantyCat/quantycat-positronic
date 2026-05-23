# Preprocessing Scripts

General-purpose dataset preprocessing tools for LeRobot v2.1 format datasets. Written for SO-101 arm teleoperation recordings but work on any compatible dataset.

Scripts live at:

```text
preprocessing_data/
```

---

## Scripts

### `trim_dataset.py`

Trims frames from the start of every episode and optionally removes bad episodes entirely.

**Use when:** Your recordings include a countdown/hold phase before the demonstration begins, or you have known-bad episodes (failed grasps, wrong trajectories) you want to drop.

```bash
python preprocessing_data/trim_dataset.py \
    --src my_data/input_data \
    --dst my_data/trimmed \
    --trim-frames 165 \
    --remove-episodes 45 12

# Preview without writing:
python preprocessing_data/trim_dataset.py \
    --src my_data/input_data \
    --dst my_data/trimmed \
    --trim-frames 165 \
    --dry-run
```

| Argument | Default | Description |
|---|---|---|
| `--src` | required | Source dataset root |
| `--dst` | required | Output dataset root |
| `--trim-frames` | 0 | Frames to cut from the start of every episode |
| `--remove-episodes` | none | Episode indices to drop entirely |
| `--dry-run` | off | Show plan without writing |

Episodes are re-indexed contiguously after removal. All meta files (info.json, episodes.jsonl, episodes_stats.jsonl, tasks.jsonl) are regenerated. Videos are trimmed frame-exact and re-encoded as H.264.

---

### `remove_pauses.py`

Removes intra-episode pauses — runs of consecutive frames where the arm is barely moving.

**Use when:** Demonstrations have hesitation mid-motion or post-task hovering before the recording was stopped. Near-zero-motion frames create contradictory supervision: the model sees visually similar frames with different labels, learns to average actions, and produces no movement at inference time.

A "pause" is any run of ≥ `--min-pause-frames` consecutive frames where the L2 norm of the action delta across joints 0–4 stays below `--speed-threshold`.

```bash
python preprocessing_data/remove_pauses.py \
    --src my_data/trimmed \
    --dst my_data/trimmed_nopause

# Preview what would be removed without writing:
python preprocessing_data/remove_pauses.py \
    --src my_data/trimmed \
    --dst my_data/trimmed_nopause \
    --dry-run

# Tune thresholds:
python preprocessing_data/remove_pauses.py \
    --src my_data/trimmed \
    --dst my_data/trimmed_nopause \
    --speed-threshold 0.015 \
    --min-pause-frames 20
```

| Argument | Default | Description |
|---|---|---|
| `--src` | required | Source dataset root |
| `--dst` | required | Output dataset root |
| `--speed-threshold` | 0.01 rad/frame | L2 joint speed below which a frame counts as paused |
| `--min-pause-frames` | 15 (= 0.5s) | Minimum consecutive pause frames before removal |
| `--dry-run` | off | Show pause summary per episode without writing |

Timestamps are reassigned as uniform `1/fps` intervals after removal so parquet and video stay in sync.

---

### `smooth_actions.py`

Smooths action trajectories using a Gaussian filter to reduce teleoperation jitter.

**Use when:** Joint positions have high-frequency noise from the demonstrator's hand tremor or abrupt micro-corrections. Smoothing reduces per-frame delta variance while preserving the overall motion shape. The gripper joint is left untouched by default so open/close events stay sharp.

Only modifies the parquet action column — videos are copied as-is.

```bash
python preprocessing_data/smooth_actions.py \
    --src my_data/trimmed_nopause \
    --dst my_data/smoothed

# Preview before/after delta stats without writing:
python preprocessing_data/smooth_actions.py \
    --src my_data/trimmed_nopause \
    --dst my_data/smoothed \
    --dry-run

# Stronger smoothing:
python preprocessing_data/smooth_actions.py \
    --src my_data/trimmed_nopause \
    --dst my_data/smoothed \
    --sigma 2.0

# Also smooth the gripper joint:
python preprocessing_data/smooth_actions.py \
    --src my_data/trimmed_nopause \
    --dst my_data/smoothed \
    --smooth-all-joints
```

| Argument | Default | Description |
|---|---|---|
| `--src` | required | Source dataset root |
| `--dst` | required | Output dataset root |
| `--sigma` | 1.5 frames | Gaussian filter width — larger = smoother |
| `--smooth-all-joints` | off | Also smooth the gripper (last joint) |
| `--dry-run` | off | Print before/after delta stats without writing |

A sigma of 1.5 gives ~12% reduction in arm delta std with minimal trajectory distortion. Go up to 2.0–3.0 for noisier data. Avoid going above 3.0 — it starts blurring the timing of real motion events.

---

## Typical Pipeline

Use `pipeline.sh` to run all three steps in one command and produce a single clean output dataset. Intermediate files are cleaned up automatically.

```bash
bash preprocessing_data/pipeline.sh \
    --src my_data/input_data \
    --dst my_data/clean_input_data \
    --trim-frames 165 \
    --remove-episodes "45" \
    --sigma 1.5

# Preview all three steps without writing:
bash preprocessing_data/pipeline.sh \
    --src my_data/input_data \
    --dst my_data/clean_input_data \
    --trim-frames 165 \
    --remove-episodes "45" \
    --dry-run
```

| Argument | Default | Description |
|---|---|---|
| `--src` | required | Source dataset root |
| `--dst` | required | Output dataset root |
| `--trim-frames` | 0 | Frames to cut from start of every episode |
| `--remove-episodes` | none | Space-separated episode indices to drop |
| `--speed-threshold` | 0.01 rad/frame | Pause detection threshold |
| `--min-pause-frames` | 15 | Minimum pause length to remove |
| `--sigma` | 1.5 | Gaussian smoothing strength |
| `--dry-run` | off | Preview all three steps without writing |

### Running steps individually

If you need to tune thresholds between steps:

```bash
python preprocessing_data/trim_dataset.py \
    --src my_data/input_data --dst my_data/trimmed \
    --trim-frames 165 --remove-episodes 45

python preprocessing_data/remove_pauses.py \
    --src my_data/trimmed --dst my_data/trimmed_nopause

python preprocessing_data/smooth_actions.py \
    --src my_data/trimmed_nopause --dst my_data/clean_input_data
```

Always run `--dry-run` first to sanity-check thresholds before writing.
