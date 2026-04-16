#!/usr/bin/env python3
"""
RynnVLA-002 closed-loop inference on SO-101 via LeRobot.

Requires:
  - This repo's conda env (lerobot, torch, etc.): source models/rynnvla-002/run_scripts/setup.sh
  - QuantyCat fork of RynnVLA-002 on PYTHONPATH or installed, e.g.:
      export PYTHONPATH="$HOME/RynnVLA-002/rynnvla-002:$PYTHONPATH"

Run from repository root:

  python3 models/rynnvla-002/inference/inference.py \\
    --checkpoint /path/to/fine_tuning/run_or_ckpt \\
    --prompt "Put the screwdriver into the cup" \\
    --robot-port /dev/ttyUSB0 \\
    --robot-id so101_follower

Default checkpoint: if you pass --from-config-checkpoint, resolves
  <work_dir>/fine_tuning/<task_label>_<robot>/
from models/rynnvla-002/config.yaml (use the directory that contains your saved solver ckpt).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage
import yaml

# Joint order must match LeRobot dataset `observation.state` for SO-101 follower demos.
_MOTOR_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

_DEFAULT_CONFIG = "models/rynnvla-002/config.yaml"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_positronic_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def _resolve_checkpoint(args: argparse.Namespace, cfg: dict[str, Any], root: Path) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint).expanduser().resolve()
    if args.from_config_checkpoint:
        work_dir = Path(cfg["work_dir"])
        if not work_dir.is_absolute():
            work_dir = (root / work_dir).resolve()
        sub = f"{cfg['task_label']}_{cfg['robot']}"
        p = (work_dir / "fine_tuning" / sub).resolve()
        if not p.is_dir():
            raise FileNotFoundError(
                f"Expected fine-tuning output directory at {p} — train first or pass --checkpoint"
            )
        return p
    raise ValueError("Pass --checkpoint PATH or --from-config-checkpoint")


def _ensure_rynnvla_on_path(extra: str | None) -> None:
    if extra:
        p = Path(extra).expanduser().resolve()
        if p.is_dir():
            sys.path.insert(0, str(p))


def _obs_to_solver_inputs(
    obs: dict[str, Any],
    front_key: str,
    wrist_key: str,
) -> tuple[Any, Any, np.ndarray]:
    if front_key not in obs or wrist_key not in obs:
        raise KeyError(
            f"Observation missing camera keys {front_key!r} / {wrist_key!r}; "
            f"got: {sorted(obs.keys())}"
        )
    front = np.asarray(obs[front_key])
    wrist = np.asarray(obs[wrist_key])
    state = np.array([float(obs[f"{m}.pos"]) for m in _MOTOR_NAMES], dtype=np.float32)
    return front, wrist, state


def _lerobot_state_to_model_state(state_raw: np.ndarray) -> np.ndarray:
    """Convert live LeRobot state values into the model-state convention.

    For this SO-101 pipeline, the dataset convention used by the model is:
    - joints 0..4 stored in radians
    - gripper stored as the same deg2rad-transformed scalar convention used
      during dataset creation

    The live robot observation is converted at this boundary so the solver
    always sees the same convention as the training data.
    """
    return np.deg2rad(np.asarray(state_raw, dtype=np.float32))


def _action_vector_to_robot_dict(vec: np.ndarray) -> dict[str, float]:
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    if v.size != len(_MOTOR_NAMES):
        raise ValueError(
            f"Expected action length {len(_MOTOR_NAMES)}, got {v.size} "
            f"(shape {getattr(vec, 'shape', None)})"
        )
    return {f"{m}.pos": float(v[i]) for i, m in enumerate(_MOTOR_NAMES)}


def _delta_to_absolute_action(delta: Any, current_state: np.ndarray) -> dict[str, float]:
    """Convert relative delta action to absolute motor positions.

    Training stores actions as (action - current_state) for joints 0-4,
    and absolute gripper position for joint 5 (see step1_convert_lerobot.py).
    send_action expects absolute positions, so we add current state back.
    """
    arr = np.asarray(delta, dtype=np.float64).reshape(-1)
    if arr.size != len(_MOTOR_NAMES):
        raise ValueError(f"Expected action length {len(_MOTOR_NAMES)}, got {arr.size}")
    state = np.asarray(current_state, dtype=np.float64).reshape(-1)
    absolute = state + arr
    absolute[-1] = arr[-1]  # gripper is already absolute
    return _action_vector_to_robot_dict(absolute)


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
    cfg = SO101FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        cameras=cameras,
    )
    return make_robot_from_config(cfg)


def main() -> None:
    root = _repo_root()
    os.chdir(root)

    parser = argparse.ArgumentParser(description="RynnVLA-002 inference on SO-101 (LeRobot).")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to finetuned checkpoint directory (or resume_path your Solver expects).",
    )
    parser.add_argument(
        "--from-config-checkpoint",
        action="store_true",
        help=f"Use work_dir/fine_tuning/<task_label>_<robot> from {_DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--positronic-config",
        type=str,
        default=_DEFAULT_CONFIG,
        help="Path to quantycat positronic models/rynnvla-002/config.yaml",
    )
    parser.add_argument(
        "--rynnvla-repo",
        type=str,
        default=os.environ.get("RYNNVLA_REPO", ""),
        help="If set, prepend this directory to sys.path (e.g. ~/RynnVLA-002/rynnvla-002)",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Language instruction for the policy.")
    parser.add_argument("--robot-port", type=str, default=None, help="Serial port for SO-101 follower.")
    parser.add_argument("--robot-id", type=str, default=None, help="Calibration id on disk.")
    parser.add_argument("--front-camera-index", type=lambda x: int(x) if x.lstrip('-').isdigit() else x, default=None)
    parser.add_argument("--wrist-camera-index", type=lambda x: int(x) if x.lstrip('-').isdigit() else x, default=None)
    parser.add_argument("--front-camera-key", type=str, default="front")
    parser.add_argument("--wrist-camera-key", type=str, default="wrist")
    parser.add_argument("--camera-width", type=int, default=None)
    parser.add_argument("--camera-height", type=int, default=None)
    parser.add_argument("--camera-fps", type=int, default=None)
    parser.add_argument(
        "--control-period-s",
        type=float,
        default=None,
        help="Sleep between steps (0 = as fast as possible).",
    )
    parser.add_argument(
        "--execute-steps-per-inference",
        type=int,
        default=None,
        help="How many predicted chunk steps to execute before re-reading cameras and re-planning.",
    )
    parser.add_argument(
        "--deterministic-crop",
        action="store_true",
        help="Use deterministic center crop in the solver item processor for reproducible debugging.",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after N steps (0 = forever).")
    parser.add_argument("--gpu", type=int, default=None, help="CUDA device index.")
    args = parser.parse_args()

    cfg_path = Path(args.positronic_config)
    if not cfg_path.is_absolute():
        cfg_path = (root / cfg_path).resolve()
    positronic_cfg = _load_positronic_config(cfg_path)

    # Fill inference params from config.yaml when not provided on CLI
    _cam_type = lambda x: int(x) if str(x).lstrip('-').isdigit() else x

    def _cfg(key, required=False):
        val = positronic_cfg.get(key)
        if required and val is None:
            raise SystemExit(f"ERROR: '{key}' is required in config.yaml")
        return val

    work_dir = Path(_cfg("work_dir", required=True))
    if not work_dir.is_absolute():
        work_dir = (root / work_dir).resolve()
    os.environ["RYNNVLA_ACTION_STATS_FILE"] = str((work_dir / "min_max_action.txt").resolve())
    os.environ["RYNNVLA_STATE_STATS_FILE"] = str((work_dir / "min_max_state.txt").resolve())

    if args.prompt is None:
        args.prompt = _cfg("prompt", required=True)
    if args.robot_port is None:
        args.robot_port = _cfg("robot_port", required=True)
    if args.robot_id is None:
        args.robot_id = _cfg("robot_id", required=True)
    if args.front_camera_index is None:
        args.front_camera_index = _cam_type(_cfg("front_camera_index", required=True))
    if args.wrist_camera_index is None:
        args.wrist_camera_index = _cam_type(_cfg("wrist_camera_index", required=True))
    if args.camera_width is None:
        args.camera_width = _cfg("camera_width", required=True)
    if args.camera_height is None:
        args.camera_height = _cfg("camera_height", required=True)
    if args.camera_fps is None:
        args.camera_fps = _cfg("camera_fps", required=True)
    if args.control_period_s is None:
        args.control_period_s = _cfg("control_period_s", required=True)
    if args.execute_steps_per_inference is None:
        args.execute_steps_per_inference = positronic_cfg.get("execute_steps_per_inference", 1)
    if args.max_steps is None:
        args.max_steps = _cfg("max_steps", required=True)
    if args.gpu is None:
        args.gpu = _cfg("gpu", required=True)
    if args.checkpoint is None:
        args.checkpoint = _cfg("checkpoint", required=True)

    ckpt_path = _resolve_checkpoint(args, positronic_cfg, root)

    _ensure_rynnvla_on_path(args.rynnvla_repo.strip() or None)

    try:
        from eval_solver_lerobot_action_head_state import Solver
    except ImportError as e:
        raise SystemExit(
            "Import failed: eval_solver_lerobot_action_head_state.Solver\n"
            "Add the QuantyCat RynnVLA-002 python root to PYTHONPATH, e.g.:\n"
            "  export PYTHONPATH=\"$HOME/RynnVLA-002/rynnvla-002:$PYTHONPATH\"\n"
            "Or pass --rynnvla-repo pointing at that directory.\n"
            f"Original error: {e}"
        ) from e

    import argparse as _argparse
    solver_args = _argparse.Namespace(
        resume_path=str(ckpt_path),
        output_dir=str(ckpt_path / "inference_logs"),
        device=args.gpu,
        action_dim=_cfg("action_dim", required=True),
        time_horizon=_cfg("chunk_size", required=True),
        max_seq_len=_cfg("max_seq_len", required=True),
        mask_image_logits=_cfg("mask_image_logits", required=True),
        dropout=_cfg("dropout", required=True),
        z_loss_weight=_cfg("inference_z_loss_weight", required=True),
        his=_cfg("his_mode", required=True),
        action_steps=_cfg("action_steps", required=True),
        deterministic_crop=bool(
            args.deterministic_crop or positronic_cfg.get("deterministic_crop", False)
        ),
    )

    Path(solver_args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Loading Solver from {ckpt_path} …")
    solver = Solver(solver_args)
    robot = _make_robot(args)

    step = 0
    try:
        try:
            robot.connect(calibrate=True)
        except ConnectionError as e:
            if "Lock" in str(e) or "enable_torque" in str(e).lower() or "status packet" in str(e).lower():
                print(f"WARNING: torque re-enable failed during configure ({e}). Retrying torque enable…")
                robot.bus.enable_torque(num_retry=5)
            else:
                raise
        print("Robot connected. Running control loop (Ctrl+C to stop).")
        img_out = Path(solver_args.output_dir) / "latest_obs"
        img_out.mkdir(parents=True, exist_ok=True)

        while True:
            obs = robot.get_observation()
            front, wrist, state = _obs_to_solver_inputs(
                obs, args.front_camera_key, args.wrist_camera_key
            )
            PILImage.fromarray(front).save(img_out / "front.jpg")
            PILImage.fromarray(wrist).save(img_out / "wrist.jpg")
            state_model = _lerobot_state_to_model_state(state)
            action_chunk = solver.get_action_wrist_action_head_state(
                front_image=front,
                wrist_image=wrist,
                state=state_model,
                prompt=args.prompt,
            )
            deltas = np.asarray(action_chunk)
            if deltas.ndim == 1:
                deltas = deltas[np.newaxis, :]  # ensure (T, 6)

            print(f"[chunk] shape={deltas.shape}  deltas[0]={np.round(deltas[0], 4)}  state_model={np.round(state_model, 4)}")

            # Receding-horizon control: predict a chunk, execute only the configured
            # prefix, then re-read cameras and re-plan.
            execute_steps = max(1, min(int(args.execute_steps_per_inference), int(deltas.shape[0])))
            for i_step, delta in enumerate(deltas[:execute_steps]):
                obs = robot.get_observation()
                _, _, state = _obs_to_solver_inputs(
                    obs, args.front_camera_key, args.wrist_camera_key
                )
                state_model = _lerobot_state_to_model_state(state)
                cmd_rad = _delta_to_absolute_action(delta, state_model)
                cmd = {k: float(np.rad2deg(v)) for k, v in cmd_rad.items()}
                print(f"  [step {step}] delta={np.round(delta, 4)}  cmd_deg={[round(v,2) for v in cmd.values()]}")
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
