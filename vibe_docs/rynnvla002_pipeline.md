Robot demos → RynnVLA-002 fine-tune → trained checkpoint → inference on SO-101.

## Run training pipeline

```bash
source models/rynnvla-002/run_scripts/setup.sh
source models/rynnvla-002/run_scripts/preprocess.sh
```

## Full pipeline

```
LeRobot dataset  (parquet + mp4)
    ↓  Step 1 — data_lerobot/lerobot_to_hdf5.py
HDF5 files  (one per episode)
    ↓  Step 2 — data_lerobot/extract_all_data.py  (needs a JSON config)
per-episode folders  (images / state / action chunks on disk)
    ↓  Step 3 — lerobot_util/action_model_conv_generation_w_2_abs_state_all_data.py
conversations JSON
    ↓  Step 4 — data_lerobot/pre_tokenize_action_state.py
pretokenized pkl files
    ↓  Step 5 — evals_lerobot/7B_ts_his_1_wrist_img_state_ck_20_all_grab_blocks_256_abs_awm_action_head.sh
trained checkpoint
    ↓  Step 6 — eval_solver_lerobot_action_head_state.py
inference on SO-101
```

---

### Step 1 — LeRobot → HDF5

```bash
python data_lerobot/lerobot_to_hdf5.py \
  --lerobot_input_dir path/to/lerobot_dataset \
  --hdf5_output_dir path/to/hdf5_output
```

Reads your LeRobot dataset via `LeRobotDataset` and writes one HDF5 file per episode.
Each HDF5 contains: `obs/front_image`, `obs/wrist_image`, `obs/state`, `action`, `timestamp`,
plus `language_instruction` and `task_index` as metadata attributes.

---

### Step 2 — HDF5 → per-episode folders

```bash
python data_lerobot/extract_all_data.py \
  --json_path path/to/task_config.json \
  --output_dir path/to/extracted_data \
  --num_processes 4
```

The `task_config.json` tells the script what task instruction to use and where the HDF5 files are:

```json
{
  "task_data": {
    "put_screwdriver_into_cup": {
      "instructions": ["Put the screwdriver into the cup"],
      "data_path": [
        "path/to/hdf5_output/episode_000000.hdf5",
        "path/to/hdf5_output/episode_000001.hdf5"
      ]
    }
  }
}
```

Output folder structure per episode:

```
output_dir/
  put_screwdriver_into_cup/
    episode_000000/
      front_image/image_0.png, image_1.png ...
      wrist_image/image_0.png, image_1.png ...
      state/state_0.npy, state_1.npy ...       ← shape (6,)
      abs_action/
        action_0/0.npy ... 19.npy              ← 20 sub-actions per timestep
        action_1/0.npy ... 19.npy
        ...
      rel_action/  ← also written, not used by conv generation
```

Note: the script filters out timesteps where the full 20-step action window is all zeros (no-op).

---

### Step 3 — Generate conversation JSON

```bash
python lerobot_util/action_model_conv_generation_w_2_abs_state_all_data.py \
  --input_dir path/to/extracted_data \
  --output_dir path/to/convs_output \
  --his 1 \
  --len_action 20 \
  --task_name screwdriver \
  --resolution 256
```

Reads the extracted folders and produces a single training JSON:
`libero_screwdriver_his_1_train_img_state_abs_ck_1_256.json`

---

### Step 4 — Pretokenize

```bash
python data_lerobot/pre_tokenize_action_state.py \
  --in_filename path/to/convs_output/libero_screwdriver_his_1_train_img_state_abs_ck_1_256.json \
  --out_dir path/to/tokens_output \
  --tokenizer path/to/ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768/snapshots/... \
  --splits 8 \
  --rank 0 \
  --target_size 256
```

Run once per `--rank` (0 through splits-1) to parallelize. Writes pkl files to `tokens_output/files/`.
Also writes `0-of-8-record.jsonl` (index of all pkl paths) used by the training data config.

---

### Step 5 — Fine-tune

Update the data config YAML to point at your record jsonl:

```
# configs/lerobot/his_1_third_view_wrist_w_state_20_256_pretokenize.yaml
META:
  - path: 'path/to/tokens_output/0-of-8-record.jsonl'
```

Then run training:

```bash
bash evals_lerobot/7B_ts_his_1_wrist_img_state_ck_20_all_grab_blocks_256_abs_awm_action_head.sh
```

Key training flags (already set in that script):
- `--action_dim 6` — matches SO-101 (6 joints)
- `--time_horizon 20` — 20 sub-actions per chunk
- `--init_from ../ckpts/starting_point` — pretrained Chameleon 7B starting point

---

### Step 6 — Inference on SO-101

```python
# eval_solver_lerobot_action_head_state.py
action = solver.get_action_wrist_action_head_state(
    front_image=front_camera_image,   # numpy array (360, 640, 3) uint8
    wrist_image=wrist_camera_image,   # numpy array (360, 640, 3) uint8
    state=joint_state,                # numpy array (6,) from SO-101
    prompt="Put the screwdriver into the cup"
)
# returns 6D action → send directly to SO-101
```

Full inference loop:

```python
import numpy as np
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera

robot = make_robot(
    robot_type="so101_follower",
    port="/dev/cu.usbmodem5B140331511",
    cameras={
        "front": OpenCVCamera(index_or_path=1, width=640, height=360, fps=30),
        "wrist": OpenCVCamera(index_or_path=0, width=640, height=360, fps=30),
    }
)
robot.connect()

from rynnvla002 import Solver
solver = Solver(resume_path="path/to/your/checkpoint")

task = "Put the screwdriver into the cup"
try:
    while True:
        obs = robot.get_observation()
        action = solver.get_action_wrist_action_head_state(
            front_image=obs["observation.images.front"],
            wrist_image=obs["observation.images.wrist"],
            state=obs["observation.state"],
            prompt=task
        )
        robot.send_action(action)
except KeyboardInterrupt:
    robot.disconnect()
```

**Before running inference:**
- `unnorm_min_max()` in `eval_solver_lerobot_action_head_state.py` has hardcoded min/max values from a different dataset. Replace them with values from your training data's `stats.json`.
- Confirm `make_robot` API matches your installed LeRobot version.

---

## What exists vs what you need to do

| Piece | Status |
|---|---|
| Model weights | ✅ HuggingFace: `Alibaba-DAMO-Academy/RynnVLA-002` |
| Step 1 — LeRobot → HDF5 | ✅ `data_lerobot/lerobot_to_hdf5.py` |
| Step 2 — HDF5 → folders | ✅ `data_lerobot/extract_all_data.py` |
| Step 2 — task_config.json | ❌ needs to be written for your dataset |
| Step 3 — conversation generation | ✅ `lerobot_util/action_model_conv_generation_w_2_abs_state_all_data.py` |
| Step 4 — pretokenize | ✅ `data_lerobot/pre_tokenize_action_state.py` |
| Step 5 — training script | ✅ `evals_lerobot/7B_ts_...action_head.sh` (update data config YAML path) |
| Step 6 — inference | ✅ `eval_solver_lerobot_action_head_state.py` (fix unnorm_min_max stats) |

---

## Hardware

```
robot port:   /dev/cu.usbmodem5B140331511
teleop port:  /dev/cu.usbmodem5B141144971

cameras:
  front:  index 1 — 640x360 @ 30fps
  wrist:  index 0 — 640x360 @ 30fps
```

Training machine: `caroline@100.83.46.36`
