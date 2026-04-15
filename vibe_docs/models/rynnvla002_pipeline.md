Robot demos → RynnVLA-002 fine-tune → trained checkpoint → inference on SO-101.

RynnVLA-002 has two components:
- **VLA Model (256×256)** — text + images → actions. This is what we fine-tune.
- **World Model (512×512)** — images + actions → next image frame. Optional, not needed for SO-101 inference.

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

## Set up environment (conda)

Do this once per machine before preprocessing (and before any step that sources `setup.sh`). From the repo root:

```bash
conda create -n rynnvla002 python=3.13 -y
conda activate rynnvla002
source models/rynnvla-002/run_scripts/setup.sh
```

If native extensions or CUDA/driver tooling fail to build, install kernel headers and build tools, then retry `setup.sh`:

```bash
sudo apt install linux-headers-$(uname -r)
sudo apt install build-essential python3-dev
```

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

Run in a tmux session so it keeps going if you disconnect:

```bash
tmux new -s train
source models/rynnvla-002/run_scripts/setup.sh
bash models/rynnvla-002/run_scripts/finetune.sh
```

Detach with `Ctrl+B` then `D`. Reattach later with `tmux attach -t train`.

This calls `models/rynnvla-002/fine_tuning/finetune.py`, which reads all parameters from `config.yaml` and launches torchrun.

Checkpoint output: `$work_dir/fine_tuning/<task_label>_<robot>/`
Training log: `$work_dir/fine_tuning/<task_label>_<robot>/output.log`

---

## Visualization

### Training dashboard

After a fine-tuning run completes, generate a self-contained HTML dashboard:

```bash
source models/rynnvla-002/run_scripts/setup.sh
bash models/rynnvla-002/run_scripts/visualization.sh
```

Output: `<training_output>/<task_label>_<robot>/dashboard.html`

Open in a browser. If on a remote machine, serve it:

```bash
cd <training_output>/<task_label>_<robot>
python3 -m http.server 8080
# then open http://<machine-ip>:8080/dashboard.html
```

**Training Metrics tab** — one chart per metric, x-axis is global step across all epochs. Vertical dotted lines mark epoch boundaries.

| Metric | What it means | What to look for |
|---|---|---|
| **closs** | Main loss — how wrong the action predictions are | Should decrease steadily. If it plateaus early, try more epochs or a higher lr. If it diverges or goes NaN, reduce lr. |
| **loss_ct** | Contrastive loss — quality of visual/state representations | Should decrease alongside closs, usually faster |
| **z_loss** | Entropy regularization — prevents codebook collapse | Should stay small and flat. A spike means the token distribution is collapsing — rare but bad. |
| **grad_norm** | Magnitude of gradients before clipping (clip at 4.0) | Should be noisy but bounded. Sustained values at the clip limit (4.0) means the model is struggling. Early training will be high. |
| **lr** | Learning rate over time | Flat after warmup for this config (warmup_epochs=0.01) |
| **dataload_s** | Time spent loading data per step | Should be low and flat. A spike means a disk or CPU bottleneck — increase `num_workers` in config. |
| **update_s** | Time spent on the forward/backward pass per step | Should be flat. A growing trend means memory pressure. |
| **samples/sec** | Training throughput | Higher is better. Dips early (GPU warming up) then stabilizes. |

**Timeline tab** — training progress vs wall time. Shows how fast steps accumulated and where checkpoints were saved/deleted.

**Raw Logs tab** — filterable log viewer showing everything except the per-step lines (those are in the charts). Useful for checking warnings, checkpoint events, and epoch summaries.

---

### Resource monitor

Captures CPU%, RAM, GPU compute%, and GPU memory at 5-second intervals during training. Produces a CSV for later analysis (resource page coming to the dashboard).

**Workflow — run alongside training:**

```bash
tmux new -s train
source models/rynnvla-002/run_scripts/setup.sh

bash models/rynnvla-002/run_scripts/resource_monitor.sh &
MONITOR_PID=$!

bash models/rynnvla-002/run_scripts/finetune.sh

kill $MONITOR_PID
```

`$!` captures the PID of the background monitor so it can be stopped cleanly when training ends.

Output: `<training_output>/<task_label>_<robot>/resources.csv`

**What it captures:**

| Column | What it means | What to look for |
|---|---|---|
| `cpu_percent` | Overall CPU utilization | Should be moderate during data loading. Very low means workers are starved; very high means a CPU bottleneck. |
| `ram_used_gb` | System RAM in use | Should be stable. Creeping upward over time is a memory leak. |
| `gpu_util_percent` | GPU compute utilization | Should be close to 100% during the update step. Low GPU utilization means the GPU is waiting on data loading — increase `num_workers`. |
| `gpu_mem_used_mb` | GPU VRAM in use | Should be flat after the first few steps. Gradual increase is a VRAM leak. |

---

## Push checkpoint to S3

```bash
aws s3 cp quantycat-positronic/my_data/training_pipeline/fine_tuning/screwdriver_so101/<epoch> s3://quantycat-positronic/screwdriver_so101/<date_epoch> --recursive
```

Verify:

```bash
aws s3 ls s3://quantycat-positronic/screwdriver_so101/
```

---

## Inference on SO-101

1. Activate the environment and set PYTHONPATH (from repo root):

```bash
source models/rynnvla-002/run_scripts/setup.sh
export PYTHONPATH="$HOME/Desktop/RynnVLA-002/rynnvla-002:$PYTHONPATH"
```

2. Set the checkpoint in `models/rynnvla-002/config.yaml`:

```yaml
checkpoint: /home/caroline/Desktop/fine_tuning/screwdriver_so101/<your-checkpoint>
```

3. Run inference:

```bash
python3 models/rynnvla-002/inference/inference.py
```

### Forked RynnVLA-002 changes to remember

The local fork at `~/RynnVLA-002/` is not a small patch on top of official RynnVLA-002. The most important differences for SO-101 debugging are:

- single-GPU training/inference fixes:
  - the original code assumed multi-GPU FSDP
  - the fork adds `SingleGPUWrapper`, removes bad dispatch-hook behavior, and fixes several CPU/GPU placement bugs
- custom checkpoint loading for inference:
  - SO-101 checkpoints were saved with `module.` prefixes
  - the fork strips `module.`, registers LoRA parameters before `load_state_dict`, and then moves the model to GPU once
- action-head inference path:
  - the fork does a direct backbone forward in `generate_action_head(...)` instead of relying on `generate()`
  - this is what fixed the all-zero / broken hidden-state path
- LeRobot-specific normalization:
  - action/state min-max values in the fork are radian-scale and task-specific, not the original repo defaults
- pretokenized training contract:
  - this checkpoint trained with `preprocess=true`, so actual `.pkl` training samples matter more than flags like `with_wrist=false`
  - the training samples for this run contained 4 image blocks, 1 state block, and 5 action blocks

Agent debugging notes:
- treat this checkpoint as trained on `his=2` with `front t-1`, `front t`, `wrist t-1`, `wrist t`, plus state
- use `2h_1a_img_both_wrist_state` for inference alignment
- the action-head input must end with the `10004` trigger token
- the wrist camera was present during training, but appears much weaker than the front camera for this task
- use `--deterministic-crop` only for debugging reproducibility; default behavior intentionally keeps randomized crop selection

### Inference alignment notes

The screwdriver checkpoint should be treated as trained on this observation contract:
- `his = 2`
- front + wrist + state
- image order: `front t-1`, `front t`, `wrist t-1`, `wrist t`

That is why the forked inference path uses `2h_1a_img_both_wrist_state` instead of the older `1h_1a_img_only_state`.

### What recent offline debugging established

Using the offline inference runner on saved observations:

- the action-head path is now structurally working:
  - the final token is the `10004` trigger
  - hidden states are finite
  - action chunks are produced with shape `(5, 6)`
- front-camera sensitivity is strong
- state sensitivity is present
- wrist sensitivity exists but is much weaker than front

This matches manual inspection of the dataset:
- the front camera usually carries the global task geometry
- the wrist camera often only shows the handle, cup wall, or a tight local view

So weak wrist influence in this checkpoint is currently expected and does **not** imply that wrist packing is broken.

### Deterministic crop flag for debugging

The item processor uses randomized crop selection by default, which is useful as augmentation but bad for debugging because the same image can tokenize differently on different runs.

For offline debugging there is now a deterministic center-crop option:

```bash
python3 models/rynnvla-002/inference/offline_inference.py \
  --deterministic-crop ...
```

This is intentionally behind a flag so the default code path stays unchanged unless reproducibility is needed.

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
