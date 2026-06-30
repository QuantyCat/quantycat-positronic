Robot demos → preprocess → OpenPI π0.5 LoRA fine-tune → trained checkpoint → inference on SO-101.

This pipeline fine-tunes Physical Intelligence `openpi` on local Quantycat SO-101
LeRobot datasets. Unlike the RynnVLA pipeline, the input LeRobot dataset is used
directly: there is no conversion into per-episode image/action folders and no
pretokenization step.

Training is dataset-agnostic: a single `pi05_quantycat_template` config is
registered, and you point it at whichever dataset you want via a CLI override.
See "Training a New Dataset" below.

---

## Full Pipeline

```
<raw recording>                                    (LeRobot v3.0 — never modified)
        ↓  convert_dataset.sh                       (v3.0 -> per-episode v2.1 layout OpenPI expects)
~/quantycat-data/datasets/<dataset>_v21/
        ↓  preprocess.sh                            (writes meta/stats.json quantile norm stats)
~/quantycat-data/datasets/<dataset>_v21/meta/stats.json
        ↓  setup.sh                                 (clones upstream openpi + applies patches + uv sync)
.venvs/openpi                                       (uv-managed openpi environment)
        ↓  train.sh --data.repo-id=<dataset>_v21
~/quantycat-data/checkpoints/openpi/pi05_quantycat_template/<exp-name>/
        ↓  run_inference.sh (in quantycat-iron-fleet)
SO-101 robot
```

Data location is controlled by `QUANTYCAT_DATA_HOME` (default: `~/quantycat-data`).
See `.env.example` for cloud override instructions.

If your raw recording is already in LeRobot v2.1 format, skip `convert_dataset.sh`.

---

## Training a New Dataset

1. Make sure the dataset lives under `$QUANTYCAT_DATA_HOME/datasets/<name>` (LeRobot
   v2.1 layout) and has norm stats in `meta/stats.json` (run `preprocess.sh`, or
   `convert_dataset.sh` already copies stats over if the source dataset has them).
2. Train against the dataset-less template config, overriding the dataset on the CLI:

   ```bash
   bash models/openpi/run_scripts/train.sh --data.repo-id=<name>
   ```

   No file changes needed for a one-off run. If `meta/stats.json` for the norm stats
   lives in a different dataset directory than the training data itself, also pass
   `--data.stats-repo-id=<stats_dataset_name>`.

3. Only once a dataset becomes a stable, recurring recipe you'll train repeatedly,
   give it a permanent name by adding a `_quantycat_train_config(...)` call in
   `models/openpi/vendor_patches/src/quantycat_training_config.py`, then
   `bash models/openpi/vendor_patches/sync.sh` to push it into the live `vendor/openpi`
   checkout. Don't register one-off experiments — that's how `quantycat_training_config.py`
   accumulated a large block of stale Dacha configs that later had to be deleted.

---

## Key Commands

From the repo root:

```bash
cd /home/caroline/quantycat-positronic

# 0. Convert a raw v3.0 recording to the v2.1 layout (skip if already v2.1).
bash models/openpi/run_scripts/convert_dataset.sh \
    --source ~/quantycat-data/datasets/<raw_dataset> \
    --target ~/quantycat-data/datasets/<dataset>_v21

# 1. Clone upstream openpi, apply patches, and install (once per machine).
bash models/openpi/run_scripts/setup.sh

# 2. Compute normalization stats (writes into the dataset's own meta/stats.json).
bash models/openpi/run_scripts/preprocess.sh

# 3. Train.
bash models/openpi/run_scripts/train.sh --data.repo-id=<dataset>_v21
```

Use `tmux` for training so it survives disconnects:

```bash
tmux new -s openpi_train
cd /home/caroline/quantycat-positronic
bash models/openpi/run_scripts/train.sh --data.repo-id=<dataset>_v21
```

---

## Dataset / Robot Schema

This applies to any dataset trained through this pipeline — it's fixed by the
SO-101 robot and `QuantycatInputs`/`QuantycatOutputs`, not per-task:

- Robot: SO-101 follower arm
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

Cleans a raw recording before conversion/training. Runs three steps in order:

1. **Trim**: removes a countdown/hold from every episode start; optionally drops bad episodes.
2. **Remove pauses**: removes runs of consecutive near-zero-motion frames.
3. **Smooth actions**: Gaussian filter on joints 0–4; gripper untouched.

```bash
bash preprocessing_data/pipeline.sh \
    --src <raw_recording_dir> \
    --dst <clean_recording_dir> \
    --trim-frames 165 \
    --remove-episodes "<episode indices to drop>" \
    --sigma 1.5

# Preview without writing:
bash preprocessing_data/pipeline.sh \
    --src <raw_recording_dir> \
    --dst <clean_recording_dir> \
    --trim-frames 165 \
    --dry-run
```

See `preprocessing_data/` for per-script documentation and threshold tuning.

**Video timestamp fix**: `remove_pauses.py` uses `setpts=N/{FPS}/TB` + `-vsync cfr` to produce uniform sequential PTS after dropping frames. Earlier versions using `setpts=PTS-STARTPTS` + `-vsync 0` preserved original PTS, creating gaps that caused LeRobot's timestamp validator to fail during norm stats computation.

---

## What Was Added To OpenPI

`vendor/openpi` is **not** a fork — `setup.sh` clones the official upstream repo
(`Physical-Intelligence/openpi`) and copies the patch files on top of it.
The patch files live in this repo at:

```text
models/openpi/vendor_patches/
  src/openpi/training/config.py             ← registration glue: imports quantycat_training_config
                                               and splices its configs into openpi's _CONFIGS
  src/quantycat_training_config.py          ← LeRobotQuantycatDataConfig, the
                                               _quantycat_train_config() factory, and the
                                               registered pi05_quantycat_template config
  src/openpi/policies/quantycat_policy.py   ← SO-101 policy transform (QuantycatInputs/Outputs)
```

Summary of changes in each file:

- **`config.py`** — Two lines: `import quantycat_training_config`, and
  `*quantycat_training_config.get_quantycat_configs()` spliced into `_CONFIGS`
  (same pattern upstream uses for `roboarena_config`/`polaris_config`). No other
  changes to upstream behavior.

- **`quantycat_training_config.py`** — `LeRobotQuantycatDataConfig` maps a LeRobot
  dataset (resolved via `HF_LEROBOT_HOME`) into OpenPI's data pipeline, loading norm
  stats directly from the dataset's `meta/stats.json`. `_quantycat_train_config(...)`
  is a factory for registering a new dataset as a permanent named config in a few
  lines. The only config registered by default is `pi05_quantycat_template`, which
  has no dataset set — see "Training a New Dataset" above.

- **`quantycat_policy.py`** — SO-101 policy transform, copied into
  `vendor/openpi/src/openpi/policies/quantycat_policy.py` by `setup.sh`. Maps the
  repacked dataset format to OpenPI's image/state/action slots; the wrist camera is
  duplicated into the unused `right_wrist_0_rgb` slot since SO-101 only has two cameras.

---

## Run Scripts

### 1. Setup

```bash
bash models/openpi/run_scripts/setup.sh
```

What it does:

1. Clones `Physical-Intelligence/openpi` (or pulls latest if already cloned) into `vendor/openpi`.
2. Copies the patch files from `models/openpi/vendor_patches/` on top.
3. Runs `GIT_LFS_SKIP_SMUDGE=1 uv sync` to install dependencies into `.venvs/openpi`.

Environment variable overrides:

| Variable | Default | Purpose |
|---|---|---|
| `OPENPI_UPSTREAM` | `https://github.com/Physical-Intelligence/openpi` | Upstream repo URL |
| `OPENPI_REF` | `main` | Branch, tag, or commit to pin |
| `OPENPI_REPO` | `vendor/openpi` | Clone destination (source only — not the venv) |
| `OPENPI_VENV` | `.venvs/openpi` | Where the uv-managed environment lives |

If `uv` is missing:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

Run `setup.sh` again whenever:
- Setting up a new machine.
- Pulling a newer upstream openpi version (`OPENPI_REF=<tag>`).
- The patch files in `vendor_patches/` have changed.
- You want a clean reinstall of `.venvs/openpi`.

For just pushing a patch edit into the live `vendor/openpi` checkout without the
full re-clone + `uv sync` overhead, use `bash models/openpi/vendor_patches/sync.sh`
instead.

### 2. Convert (only if the dataset isn't already LeRobot v2.1)

```bash
bash models/openpi/run_scripts/convert_dataset.sh \
    --source ~/quantycat-data/datasets/<raw_v3_dataset> \
    --target ~/quantycat-data/datasets/<output_v21_dataset>
```

Raw recordings come out of current LeRobot recording tooling in codebase v3.0
(one combined parquet/video file per chunk); OpenPI's pinned `lerobot`/`datasets`
versions expect the older per-episode v2.1 layout. This script does that
conversion. Prints per-episode progress since the video-splitting step (`ffmpeg`
per episode per camera) can take a few minutes with no other output.

### 3. Preprocess / Norm Stats

```bash
bash models/openpi/run_scripts/preprocess.sh
```

This delegates to `models/lerobot/run_scripts/preprocess.sh`, which runs
`augment_dataset_quantile_stats.py` against the dataset and writes q01/q99
quantile stats directly into `<dataset>/meta/stats.json`. Both the LeRobot and
OpenPI pipelines read norm stats from that same file, so this is the only
preprocessing step needed before training.

### 4. Training

```bash
bash models/openpi/run_scripts/train.sh --data.repo-id=<dataset>
```

This runs (via `tyro`'s CLI override mechanism on the `pi05_quantycat_template`
config):

```bash
cd vendor/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_quantycat_template \
  --exp-name=<exp-name> \
  --overwrite \
  --data.repo-id=<dataset>
```

`EXP_NAME` defaults to a timestamp if not set. Extra arguments after `train.sh`
are forwarded straight to `scripts/train.py`, so any `TrainConfig` field can be
overridden the same way (e.g. `--data.stats-repo-id=...`, `--num-train-steps=...`).

Expected checkpoints:

```text
~/quantycat-data/checkpoints/openpi/pi05_quantycat_template/<exp-name>/
```

(Or under a permanent config's name, if you've promoted the dataset via
`_quantycat_train_config(...)` — see "Training a New Dataset" above.)

---

## Live Deployment

> **Inference scripts have moved to `quantycat-iron-fleet`.**
>
> All commands below should be run from `~/quantycat-iron-fleet`.
> See `vibe_docs/inference.md` there for the full guide.

Inference doesn't depend on a config name being registered in OpenPI's `_CONFIGS` —
`quantycat-iron-fleet`'s `openpi_runner.py` builds its own inference-time data
config directly from the deployment JSON's `model` block (checkpoint path,
asset_id, prompt, etc.); it never looks anything up by config name.

### Config File

Inference configs live under `~/quantycat-iron-fleet/inference/openpi/`, one
JSON file per deployed task, e.g.:

```text
~/quantycat-iron-fleet/inference/openpi/quantycat_dacha_inference_config.json
```

Key fields (nested, not flat — see the file itself for the full schema):

| Field | Example | Notes |
|---|---|---|
| `model.checkpoint_path` | `.../checkpoints/openpi/.../9999` | trained checkpoint to load |
| `model.asset_id` | dataset name used for norm stats at train time | must match the checkpoint's training config |
| `model.prompt` | task-specific instruction text | |
| `model.sample_steps` | `10` | diffusion steps at inference |
| `model.action_horizon` | matches the training config's `action_horizon` | |
| `control.max_steps` | `0` | infinite loop (Ctrl+C to stop) |
| `control.execute_steps_per_inference` | action chunk steps before re-inferring | |
| `control.control_period_s` | matches training fps | |
| `calibration.gain_vector` | `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]` | recalibrate after first run |
| `safety.max_delta_per_command_deg` | per-joint safety clip | |

### Check / Dry-Run / Live

`run_inference.sh` starts a policy server (openpi venv) then runs robot control
(lerobot venv) via a websocket, forwarding extra args to `robots/lerobot/run_robot.py`:

```bash
cd ~/quantycat-iron-fleet

# Full policy-load check (no server, no robot).
bash inference/openpi/run_inference.sh --check-only

# Dry-run: connects to robot, runs policy, applies gain/clips, but does NOT send actions.
bash inference/openpi/run_inference.sh --dry-run --max-steps 5

# Live run with step cap.
bash inference/openpi/run_inference.sh --max-steps 60

# Infinite live run.
bash inference/openpi/run_inference.sh
```

Override `CONFIG`/`CHECKPOINT` env vars to point at a different deployment JSON
or checkpoint than the script's defaults.

### Self-Start Workaround

From the default rest pose the model may predict near-hold actions. Move the arm
to the home pose first, then start inference:

```bash
cd ~/quantycat-iron-fleet
bash robots/lerobot/go_home.sh
bash inference/openpi/run_inference.sh
```

### Action Convention

```text
delta = predicted_target - current_state
calibrated_target = current_state + delta * gain_vector
```

After gain, the runner clips per-command deltas and final absolute targets,
converts back to live robot units (degrees), and sends the command dictionary.

### Logs

Each rollout writes to `~/quantycat-data/logs/inference/openpi/` (default; overridden by `QUANTYCAT_DATA_HOME`):

```text
~/quantycat-data/logs/inference/openpi/<timestamp>_<checkpoint>/
  rollout.jsonl            one JSON line per step: state, target, safety info
  deployment_config.json   snapshot of the config used
  rollout_front.mp4        front camera video
  latest_observation.png   front+wrist side-by-side from last step
```

---

## Inference Input Format

Inference callers must send observations with the keys expected by `QuantycatInputs`:

```python
{
    "observation/state": state,          # 6-D joint/gripper positions
    "observation/images/front": front,   # (H, W, 3) uint8
    "observation/images/wrist": wrist,   # (H, W, 3) uint8
    "prompt": "<task prompt>",
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

All outputs write to `$QUANTYCAT_DATA_HOME` (default: `~/quantycat-data`), under
whichever config name you actually trained with (`pi05_quantycat_template` for a
CLI-override run, or a permanent name if promoted):

```text
~/quantycat-data/checkpoints/openpi/<config_name>/<exp-name>/
```

Norm stats live with the dataset itself, not in a separate location:

```text
~/quantycat-data/datasets/<dataset>/meta/stats.json
```

---

## Known Issues / Fixes

### HuggingFace `List` Feature Error

This was previously patched in `vendor_patches/src/openpi/training/data_loader.py`, but
testing showed it's data-dependent — `screwdriver_so101_clean_v2` loads fine without it —
so the patch was dropped. If a future dataset's parquet metadata declares a `List`-typed
column and you hit:

```text
ValueError: Feature type 'List' not found
```

the dataset's HuggingFace `datasets` schema is using a feature type that predates
OpenPI's pinned `datasets==3.6.0`. Fix by aliasing it before loading:

```python
import datasets.features.features as _hf_features
_hf_features._FEATURE_TYPES["List"] = _hf_features.Sequence
```

### Video/Parquet Timestamp Mismatch After Pause Removal

If preprocess fails with:

```text
AssertionError: One or several query timestamps unexpectedly violate the tolerance
```

the fix is in `preprocessing_data/remove_pauses.py`. Uses `setpts=N/{FPS}/TB`
+ `-vsync cfr` + `-r {FPS}` to produce uniform sequential PTS. The original
`setpts=PTS-STARTPTS` + `-vsync 0` preserved gaps where pauses were removed.

---

## Minimal Checklist

```bash
# --- Training (in quantycat-positronic) ---
cd /home/caroline/quantycat-positronic

# 0. Preprocess raw recordings.
bash preprocessing_data/pipeline.sh \
    --src <raw_recording_dir> --dst <clean_recording_dir> \
    --trim-frames 165 --sigma 1.5

# 1. Convert to v2.1 if needed.
bash models/openpi/run_scripts/convert_dataset.sh \
    --source <clean_recording_dir> --target ~/quantycat-data/datasets/<dataset>_v21

# 2. Clone upstream openpi, apply patches, and install.
bash models/openpi/run_scripts/setup.sh

# 3. Compute normalization stats.
bash models/openpi/run_scripts/preprocess.sh

# 4. Train.
bash models/openpi/run_scripts/train.sh --data.repo-id=<dataset>_v21

# --- Inference (in quantycat-iron-fleet) ---
cd /home/caroline/quantycat-iron-fleet

# 5. Check deployment config, then dry-run.
bash inference/openpi/run_inference.sh --check-only
bash inference/openpi/run_inference.sh --dry-run --max-steps 5
```
