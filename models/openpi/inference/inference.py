"""
openpi closed-loop inference on SO-101 via LeRobot.

Loads the trained pi0 policy directly (no websocket server required for
single-machine deployment), then runs a receding-horizon control loop
identical in structure to models/rynnvla-002/inference/inference.py.

Requires:
  - openpi conda/venv env active: source models/openpi/run_scripts/setup.sh
  - PYTHONPATH includes ~/openpi/src and models/openpi/training_config/

Run from repo root:
    python3 models/openpi/inference/inference.py \\
        --checkpoint my_data/training_pipeline/fine_tuning/<run>/29999 \\
        --prompt "Put the screwdriver into the cup"
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
from typing import Any

import numpy as np
from PIL import Image as PILImage
import torch
import yaml

_MOTOR_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

_DEFAULT_CONFIG = "models/openpi/config.yaml"


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3]


def _setup_paths(repo_root: pathlib.Path) -> None:
    openpi_repo = pathlib.Path.home() / "openpi"
    training_config_dir = repo_root / "models" / "openpi" / "training_config"
    for p in (str(openpi_repo / "src"), str(training_config_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _lerobot_state_to_model_state(state_raw: np.ndarray) -> np.ndarray:
    """Convert live LeRobot state (degrees) to model state (also degrees).

    openpi normalises values internally; the model state convention matches
    the dataset convention (degrees), so no conversion is needed here.
    The state is passed through as-is, matching what the LeRobot dataset
    stored during data collection.
    """
    return np.asarray(state_raw, dtype=np.float32)


def _obs_to_policy_input(
    obs: dict[str, Any],
    front_key: str,
    wrist_key: str,
    prompt: str,
) -> dict:
    """Build the dict that the openpi Policy.infer() call expects."""
    front = np.asarray(obs[front_key])
    wrist = np.asarray(obs[wrist_key])
    state = np.array([float(obs[f"{m}.pos"]) for m in _MOTOR_NAMES], dtype=np.float32)
    state_model = _lerobot_state_to_model_state(state)
    return {
        "observation/image": front,
        "observation/wrist_image": wrist,
        "observation/state": state_model,
        "prompt": prompt,
    }, state_model


def _absolute_action_from_delta(
    delta: np.ndarray,
    current_state: np.ndarray,
) -> dict[str, float]:
    """Convert delta action chunk step to absolute motor positions.

    Training applies DeltaActions(mask=[True]*5+[False]) which makes
    joints 0-4 relative to current state and keeps gripper (joint 5)
    absolute. This inverts that transform.
    """
    delta = np.asarray(delta, dtype=np.float64).reshape(-1)
    state = np.asarray(current_state, dtype=np.float64).reshape(-1)
    absolute = state + delta
    absolute[-1] = delta[-1]  # gripper stays absolute
    return {f"{m}.pos": float(absolute[i]) for i, m in enumerate(_MOTOR_NAMES)}


def _make_robot(args: argparse.Namespace):
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.robots.utils import make_robot_from_config

    cameras = {
        args.front_camera_key: OpenCVCameraConfig(
            index_or_path=args.front_camera_index,
            fps=args.camera_fps,
            width=args.camera_width,
            height=args.camera_height,
        ),
        args.wrist_camera_key: OpenCVCameraConfig(
            index_or_path=args.wrist_camera_index,
            fps=args.camera_fps,
            width=args.camera_width,
            height=args.camera_height,
        ),
    }
    cfg = SO101FollowerConfig(port=args.robot_port, id=args.robot_id, cameras=cameras)
    return make_robot_from_config(cfg)


def main():
    root = _repo_root()
    os.chdir(root)
    _setup_paths(root)

    _cam_type = lambda x: int(x) if str(x).lstrip("-").isdigit() else x

    parser = argparse.ArgumentParser(description="openpi inference on SO-101.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--positronic-config", type=str, default=_DEFAULT_CONFIG)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--robot-port", type=str, default=None)
    parser.add_argument("--robot-id", type=str, default=None)
    parser.add_argument("--front-camera-index", type=_cam_type, default=None)
    parser.add_argument("--wrist-camera-index", type=_cam_type, default=None)
    parser.add_argument("--front-camera-key", type=str, default="front")
    parser.add_argument("--wrist-camera-key", type=str, default="wrist")
    parser.add_argument("--camera-width", type=int, default=None)
    parser.add_argument("--camera-height", type=int, default=None)
    parser.add_argument("--camera-fps", type=int, default=None)
    parser.add_argument("--control-period-s", type=float, default=None)
    parser.add_argument("--execute-steps-per-inference", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()

    cfg_path = pathlib.Path(args.positronic_config)
    if not cfg_path.is_absolute():
        cfg_path = (root / cfg_path).resolve()
    with open(cfg_path) as f:
        yaml_cfg = yaml.safe_load(f)

    def _cfg(key, required=True):
        v = yaml_cfg.get(key)
        if required and v is None:
            raise SystemExit(f"ERROR: '{key}' must be set in config.yaml")
        return v

    if args.checkpoint is None:
        args.checkpoint = _cfg("checkpoint")
    if args.prompt is None:
        args.prompt = _cfg("prompt")
    if args.robot_port is None:
        args.robot_port = _cfg("robot_port")
    if args.robot_id is None:
        args.robot_id = _cfg("robot_id")
    if args.front_camera_index is None:
        args.front_camera_index = _cam_type(_cfg("front_camera_index"))
    if args.wrist_camera_index is None:
        args.wrist_camera_index = _cam_type(_cfg("wrist_camera_index"))
    if args.camera_width is None:
        args.camera_width = _cfg("camera_width")
    if args.camera_height is None:
        args.camera_height = _cfg("camera_height")
    if args.camera_fps is None:
        args.camera_fps = _cfg("camera_fps")
    if args.control_period_s is None:
        args.control_period_s = _cfg("control_period_s")
    if args.execute_steps_per_inference is None:
        args.execute_steps_per_inference = yaml_cfg.get("execute_steps_per_inference", 2)
    if args.max_steps is None:
        args.max_steps = yaml_cfg.get("max_steps", 0)
    if args.gpu is None:
        args.gpu = yaml_cfg.get("gpu", 0)

    ckpt_path = pathlib.Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.is_dir():
        raise SystemExit(f"ERROR: checkpoint directory not found: {ckpt_path}")

    # Build TrainConfig and load policy
    from so101_train_config import build_train_config
    import openpi.policies.policy_config as _policy_cfg

    train_cfg = build_train_config(yaml_cfg)
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    print(f"Loading policy from {ckpt_path} …")
    policy = _policy_cfg.create_trained_policy(
        train_cfg,
        ckpt_path,
        default_prompt=args.prompt,
        pytorch_device=device_str,
    )

    robot = _make_robot(args)
    img_out = ckpt_path / "inference_logs" / "latest_obs"
    img_out.mkdir(parents=True, exist_ok=True)

    step = 0
    try:
        try:
            robot.connect(calibrate=True)
        except ConnectionError as e:
            if "Lock" in str(e) or "enable_torque" in str(e).lower() or "status packet" in str(e).lower():
                print(f"WARNING: torque re-enable failed during connect ({e}). Retrying…")
                robot.bus.enable_torque(num_retry=5)
            else:
                raise

        print("Robot connected. Running control loop (Ctrl+C to stop).")

        while True:
            obs = robot.get_observation()
            policy_input, state_model = _obs_to_policy_input(
                obs, args.front_camera_key, args.wrist_camera_key, args.prompt
            )

            # Save latest images for debugging
            PILImage.fromarray(np.asarray(obs[args.front_camera_key])).save(img_out / "front.jpg")
            PILImage.fromarray(np.asarray(obs[args.wrist_camera_key])).save(img_out / "wrist.jpg")

            # Get predicted action chunk from policy
            result = policy.infer(policy_input)
            # result["actions"] has shape (action_horizon, action_dim)
            deltas = np.asarray(result["actions"])
            if deltas.ndim == 1:
                deltas = deltas[np.newaxis, :]

            if not np.isfinite(deltas).all():
                print("[safety] Non-finite action chunk. Aborting.")
                break

            print(f"[chunk] shape={deltas.shape}  deltas[0]={np.round(deltas[0], 4)}")

            # Receding-horizon execution
            execute_steps = max(1, min(int(args.execute_steps_per_inference), int(deltas.shape[0])))
            for delta in deltas[:execute_steps]:
                obs = robot.get_observation()
                _, state_model = _obs_to_policy_input(
                    obs, args.front_camera_key, args.wrist_camera_key, args.prompt
                )
                cmd = _absolute_action_from_delta(delta, state_model)
                if not all(np.isfinite(v) for v in cmd.values()):
                    print("[safety] Non-finite command. Aborting.")
                    break
                print(f"  [step {step}] delta={np.round(delta, 4)}  cmd_deg={[round(v, 2) for v in cmd.values()]}")
                robot.send_action(cmd)
                step += 1
                if args.max_steps and step >= args.max_steps:
                    break
                if args.control_period_s > 0:
                    time.sleep(args.control_period_s)

            if args.max_steps and step >= args.max_steps:
                break

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
