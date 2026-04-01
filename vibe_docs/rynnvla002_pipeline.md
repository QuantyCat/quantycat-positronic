Robot demos → RynnVLA-002 → trained checkpoint → inference on SO-101.

## Full pipeline

### Step 1 — Convert LeRobot demos to RynnVLA-002 format

Your LeRobot data (parquet + mp4) needs to be converted into per-episode folders:

```
episode_0/
  imgs_third_view/image_0.png, image_1.png...   ← front camera frames
  imgs_wrist/image_0.png, image_1.png...         ← wrist camera frames
  action/action_0.npy, action_1.npy...           ← joint commands
  eef_gripper_state/state_0.npy...               ← joint positions
episode_1/
  ...
```

This converter does not exist yet — you need to write it. It is straightforward:
extract mp4 frames → PNG, extract parquet rows → npy files.

---

### Step 2 — Generate conversation files

```bash
python data/action_state_model_conv_generation.py
```

Produces JSON files pairing images + states + actions into the conversational format
the model expects for training.

---

### Step 3 — Pretokenize

```bash
python data/pre_tokenize_action_state.py
```

Converts images and actions into Chameleon tokens before training (faster than
tokenizing on the fly during training).

---

### Step 4 — Fine-tune

```bash
bash exps_pretokenize/your_config.sh
```

Runs training on your processed data. Copy one of the existing LIBERO config scripts
and update the data paths to point to your converted dataset.

---

### Step 5 — Inference on your robot

```python
# eval_solver_lerobot_action_head_state.py
action = solver.get_action_wrist_action_head_state(
    front_image=front_camera_image,   # numpy array (360, 640, 3) from front camera
    wrist_image=wrist_camera_image,   # numpy array (360, 640, 3) from wrist camera
    state=joint_state,                # numpy array (6,) from SO-101
    prompt="Put the screwdriver into the cup"
)
# returns 7D action → send directly to SO-101
```

Full inference loop wired to SO-101:

```python
import numpy as np
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera

# Connect to SO-101
robot = make_robot(
    robot_type="so101_follower",
    port="/dev/cu.usbmodem5B140331511",
    cameras={
        "front": OpenCVCamera(index_or_path=1, width=640, height=360, fps=30),
        "wrist": OpenCVCamera(index_or_path=0, width=640, height=360, fps=30),
    }
)
robot.connect()

# Load trained checkpoint
from rynnvla002 import Solver
solver = Solver(resume_path="path/to/your/checkpoint")

# Run
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

**Things to verify before this runs:**
- The model outputs 7D — your SO-101 uses 6D joints + gripper. Check ordering.
- Action normalization stats must match your training data (`my_data/training/dataset/meta/stats.json`).
- Confirm `make_robot` API matches your installed LeRobot version.

---

## What exists vs what you need to build

| Piece | Status |
|---|---|
| Model weights | ✅ HuggingFace: `Alibaba-DAMO-Academy/RynnVLA-002` |
| Conversation generation | ✅ `rynnvla-002/data/action_state_model_conv_generation.py` |
| Pretokenize scripts | ✅ `rynnvla-002/data/pre_tokenize_action_state.py` |
| Training scripts | ✅ `rynnvla-002/exps_pretokenize/` |
| Inference loop | ✅ `rynnvla-002/eval_solver_lerobot_action_head_state.py` |
| LeRobot → training format converter | ❌ needs to be written (~50 lines Python) |
| SO-101 inference integration | ❌ needs to be wired up (~30 lines Python) |

---

## Hardware

```
robot port:   /dev/cu.usbmodem5B140331511
teleop port:  /dev/cu.usbmodem5B141144971

cameras:
  front:  index 1 — 640x360 @ 30fps
  wrist:  index 0 — 640x360 @ 30fps
```

Training runs on: `caroline@100.83.46.36`
