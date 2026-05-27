#!/usr/bin/env python3
"""Run a Quantycat OpenPI policy on a live SO-101 follower through LeRobot."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


_MOTOR_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
_DEFAULT_CONFIG = "models/openpi/inference/inference_config.json"
_NEW_LEROBOT_CALIBRATION_DIR = Path("/home/caroline/.cache/huggingface/lerobot/calibration/robots/so_follower")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _checkpoint_name(cfg: dict[str, Any]) -> str:
    checkpoint = Path(cfg["model"]["checkpoint_path"])
    name = checkpoint.name
    if name.isdigit():
        name = checkpoint.parent.name
    return name


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _as_camera_index(value: Any) -> int | str:
    text = str(value)
    if text.lstrip("-").isdigit():
        return int(text)
    return text


def _image_to_uint8_hwc(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if np.issubdtype(arr.dtype, np.floating):
        max_value = float(np.nanmax(arr)) if arr.size else 1.0
        if max_value <= 1.0:
            arr = arr * 255.0
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3:
        raise ValueError(f"Expected image with 2 or 3 dimensions, got shape {arr.shape}")
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    return arr


def _save_latest_observation_png(session_dir: Path, front: np.ndarray, wrist: np.ndarray) -> None:
    from PIL import Image

    front_img = Image.fromarray(_image_to_uint8_hwc(front), mode="RGB")
    wrist_img = Image.fromarray(_image_to_uint8_hwc(wrist), mode="RGB")
    if wrist_img.height != front_img.height:
        width = max(1, round(wrist_img.width * front_img.height / wrist_img.height))
        wrist_img = wrist_img.resize((width, front_img.height))
    combined = Image.new("RGB", (front_img.width + wrist_img.width, front_img.height))
    combined.paste(front_img, (0, 0))
    combined.paste(wrist_img, (front_img.width, 0))
    combined.save(session_dir / "latest_observation.png")


def _vector(value: Any, *, name: str, length: int = 6) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != length:
        raise ValueError(f"{name} must have length {length}; got {arr.size}: {value}")
    return arr


def _state_to_model_units(state_raw: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    live_units = cfg["robot"].get("live_state_units", "deg")
    model_units = cfg["robot"].get("model_state_units", "rad")
    state = np.asarray(state_raw, dtype=np.float64)
    if live_units == model_units:
        return state.astype(np.float32)
    if live_units == "deg" and model_units == "rad":
        return np.deg2rad(state).astype(np.float32)
    if live_units == "rad" and model_units == "deg":
        return np.rad2deg(state).astype(np.float32)
    raise ValueError(f"Unsupported unit conversion: {live_units} -> {model_units}")


def _model_to_robot_units(target_model: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    live_units = cfg["robot"].get("live_state_units", "deg")
    model_units = cfg["robot"].get("model_state_units", "rad")
    target = np.asarray(target_model, dtype=np.float64)
    if live_units == model_units:
        return target
    if model_units == "rad" and live_units == "deg":
        return np.rad2deg(target)
    if model_units == "deg" and live_units == "rad":
        return np.deg2rad(target)
    raise ValueError(f"Unsupported unit conversion: {model_units} -> {live_units}")


def _obs_to_policy_input(obs: dict[str, Any], cfg: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    robot_cfg = cfg["robot"]
    front_key = robot_cfg["front_camera_key"]
    wrist_key = robot_cfg["wrist_camera_key"]
    front_obs_key = front_key if front_key in obs else f"observation.images.{front_key}"
    wrist_obs_key = wrist_key if wrist_key in obs else f"observation.images.{wrist_key}"
    if front_obs_key not in obs or wrist_obs_key not in obs:
        raise KeyError(
            f"Observation missing camera keys {front_key!r}/{wrist_key!r}; got {sorted(obs)}"
        )

    front = np.asarray(obs[front_obs_key])
    wrist = np.asarray(obs[wrist_obs_key])
    if "observation.state" in obs:
        state_raw = np.asarray(obs["observation.state"], dtype=np.float64).reshape(-1)
    else:
        state_raw = np.array([float(obs[f"{motor}.pos"]) for motor in _MOTOR_NAMES], dtype=np.float64)
    state_model = _state_to_model_units(state_raw, cfg)
    policy_obs = {
        "observation/images/front": front,
        "observation/images/wrist": wrist,
        "observation/state": state_model,
        "prompt": cfg["model"]["prompt"],
    }
    return policy_obs, state_raw, state_model


def _action_vector_to_robot_dict(vec_live_units: np.ndarray) -> dict[str, float]:
    vec = np.asarray(vec_live_units, dtype=np.float64).reshape(-1)
    if vec.size != len(_MOTOR_NAMES):
        raise ValueError(f"Expected {len(_MOTOR_NAMES)} action values, got {vec.size}")
    return {f"{motor}.pos": float(vec[idx]) for idx, motor in enumerate(_MOTOR_NAMES)}


def _calibrate_and_clip(
    pred_abs_model: np.ndarray,
    current_state_model: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    pred = np.asarray(pred_abs_model, dtype=np.float64).reshape(-1)[:6]
    state = np.asarray(current_state_model, dtype=np.float64).reshape(-1)[:6]
    gains = _vector(cfg["calibration"]["gain_vector"], name="calibration.gain_vector")

    delta = pred - state
    gained = state + delta * gains

    safety = cfg["safety"]
    max_delta_live = _vector(safety["max_delta_per_command_deg"], name="safety.max_delta_per_command_deg")
    max_delta_model = np.deg2rad(max_delta_live) if cfg["robot"].get("model_state_units", "rad") == "rad" else max_delta_live
    limited_delta = np.clip(gained - state, -max_delta_model, max_delta_model)
    limited = state + limited_delta

    limits = safety.get("target_limits_deg")
    if limits is None:
        clipped = limited
    else:
        lower_live = _vector(limits["min"], name="safety.target_limits_deg.min")
        upper_live = _vector(limits["max"], name="safety.target_limits_deg.max")
        if cfg["robot"].get("model_state_units", "rad") == "rad":
            lower_model = np.deg2rad(lower_live)
            upper_model = np.deg2rad(upper_live)
        else:
            lower_model = lower_live
            upper_model = upper_live
        clipped = np.clip(limited, lower_model, upper_model)

    info = {
        "pred_abs_model": pred.tolist(),
        "delta_model": delta.tolist(),
        "gained_abs_model": gained.tolist(),
        "delta_limited": bool(not np.allclose(gained, limited)),
        "target_limited": bool(not np.allclose(limited, clipped)),
    }
    return clipped.astype(np.float32), info


def _make_robot(cfg: dict[str, Any]):
    robot_cfg = cfg["robot"]
    try:
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
        from lerobot.robots.utils import make_robot_from_config

        cameras = {
            robot_cfg["front_camera_key"]: OpenCVCameraConfig(
                index_or_path=_as_camera_index(robot_cfg["front_camera_index"]),
                fps=int(robot_cfg["camera_fps"]),
                width=int(robot_cfg["camera_width"]),
                height=int(robot_cfg["camera_height"]),
            ),
            robot_cfg["wrist_camera_key"]: OpenCVCameraConfig(
                index_or_path=_as_camera_index(robot_cfg["wrist_camera_index"]),
                fps=int(robot_cfg["camera_fps"]),
                width=int(robot_cfg["camera_width"]),
                height=int(robot_cfg["camera_height"]),
            ),
        }
        robot_config = SO101FollowerConfig(
            port=robot_cfg["robot_port"],
            id=robot_cfg["robot_id"],
            cameras=cameras,
        )
        return make_robot_from_config(robot_config)
    except ModuleNotFoundError:
        from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
        from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
        from lerobot.common.robot_devices.robots.configs import So101RobotConfig
        from lerobot.common.robot_devices.robots.utils import make_robot_from_config

        cameras = {
            robot_cfg["front_camera_key"]: OpenCVCameraConfig(
                camera_index=_as_camera_index(robot_cfg["front_camera_index"]),
                fps=int(robot_cfg["camera_fps"]),
                width=int(robot_cfg["camera_width"]),
                height=int(robot_cfg["camera_height"]),
            ),
            robot_cfg["wrist_camera_key"]: OpenCVCameraConfig(
                camera_index=_as_camera_index(robot_cfg["wrist_camera_index"]),
                fps=int(robot_cfg["camera_fps"]),
                width=int(robot_cfg["camera_width"]),
                height=int(robot_cfg["camera_height"]),
            ),
        }
        follower_bus = FeetechMotorsBusConfig(
            port=robot_cfg["robot_port"],
            motors={
                "shoulder_pan": [1, "sts3215"],
                "shoulder_lift": [2, "sts3215"],
                "elbow_flex": [3, "sts3215"],
                "wrist_flex": [4, "sts3215"],
                "wrist_roll": [5, "sts3215"],
                "gripper": [6, "sts3215"],
            },
        )
        compat_calibration_dir = _prepare_compat_calibration(robot_cfg["robot_id"])
        robot_config = So101RobotConfig(
            calibration_dir=str(compat_calibration_dir),
            leader_arms={},
            follower_arms={"so101": follower_bus},
            cameras=cameras,
        )
        return make_robot_from_config(robot_config)


def _connect_robot(robot: Any, control: dict[str, Any]) -> None:
    def _try_connect() -> None:
        try:
            robot.connect(calibrate=bool(control.get("connect_calibrate", True)))
        except TypeError:
            robot.connect()

    def _cleanup_partial_connect() -> None:
        for attr in ("cameras", "leader_arms", "follower_arms"):
            devices = getattr(robot, attr, {})
            if isinstance(devices, dict):
                for device in devices.values():
                    try:
                        if getattr(device, "is_connected", False):
                            device.disconnect()
                    except Exception:
                        pass
        if hasattr(robot, "is_connected"):
            robot.is_connected = False

    try:
        _try_connect()
        return
    except ConnectionError as e:
        message = str(e).lower()
        if "no status packet" not in message and "communication error" not in message:
            raise
        print(f"WARNING: initial robot connect failed ({e}). Retrying once after cleanup...")
        _cleanup_partial_connect()
        time.sleep(0.5)
        _try_connect()


def _read_observation(robot: Any) -> dict[str, Any]:
    if hasattr(robot, "get_observation"):
        return robot.get_observation()
    if hasattr(robot, "capture_observation"):
        return robot.capture_observation()
    raise AttributeError("Robot object does not provide get_observation() or capture_observation().")


def _send_robot_action(robot: Any, target_live: np.ndarray) -> Any:
    if hasattr(robot, "get_observation"):
        return robot.send_action(_action_vector_to_robot_dict(target_live))

    import torch

    return robot.send_action(torch.as_tensor(target_live, dtype=torch.float32))


def _load_policy(root: Path, cfg: dict[str, Any]):
    openpi_repo = _resolve_path(root, cfg["model"]["openpi_repo"])
    if openpi_repo is None or not openpi_repo.is_dir():
        raise FileNotFoundError(f"OpenPI repo not found: {openpi_repo}")
    src = openpi_repo / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"OpenPI src directory not found: {src}")
    sys.path.insert(0, str(src))
    os.chdir(openpi_repo)

    from openpi.policies import policy_config
    from openpi.training import config as openpi_config

    import dataclasses

    train_config = openpi_config.get_config(cfg["model"]["config_name"])
    right_wrist_source = cfg["model"].get("right_wrist_source", "wrist")
    if right_wrist_source != "wrist":
        variant_data = dataclasses.replace(train_config.data, right_wrist_source=right_wrist_source)
        train_config = dataclasses.replace(train_config, data=variant_data)

    checkpoint = _resolve_path(root, cfg["model"]["checkpoint_path"])
    if checkpoint is None or not (checkpoint / "params").is_dir():
        raise FileNotFoundError(f"OpenPI checkpoint params not found: {checkpoint / 'params'}")
    if not (checkpoint / "assets").is_dir():
        raise FileNotFoundError(f"OpenPI checkpoint assets not found: {checkpoint / 'assets'}")

    sample_steps = int(cfg["model"].get("sample_steps", 10))
    return policy_config.create_trained_policy(
        train_config,
        checkpoint,
        sample_kwargs={"num_steps": sample_steps},
        default_prompt=cfg["model"]["prompt"],
        pytorch_device=cfg["model"].get("pytorch_device"),
    )


def _load_state_stats(root: Path, cfg: dict[str, Any]) -> dict[str, Any] | None:
    checkpoint = _resolve_path(root, cfg["model"]["checkpoint_path"])
    if checkpoint is None:
        return None
    stats_path = checkpoint / "assets" / "openpi" / "norm_stats.json"
    payload = _maybe_load_json(stats_path)
    if payload is None:
        return None

    state_stats = payload.get("norm_stats", {}).get("state")
    if not isinstance(state_stats, dict) or "q01" not in state_stats or "q99" not in state_stats:
        return None

    q01 = _vector(state_stats["q01"], name="norm_stats.state.q01")
    q99 = _vector(state_stats["q99"], name="norm_stats.state.q99")
    return {"q01": q01, "q99": q99, "path": str(stats_path)}


def _state_guard_issue(
    state_model: np.ndarray,
    cfg: dict[str, Any],
    state_stats: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not cfg["safety"].get("abort_on_ood_state", True):
        return None

    state = np.asarray(state_model, dtype=np.float64).reshape(-1)[:6]
    issues: list[dict[str, Any]] = []

    if state_stats is not None:
        q01 = np.asarray(state_stats["q01"], dtype=np.float64)
        q99 = np.asarray(state_stats["q99"], dtype=np.float64)
        margin = np.deg2rad(float(cfg["safety"].get("state_stats_margin_deg", 5.0)))
        below = np.where(state < (q01 - margin))[0]
        above = np.where(state > (q99 + margin))[0]
        if below.size or above.size:
            issues.append(
                {
                    "kind": "state_outside_training_stats",
                    "stats_path": state_stats["path"],
                    "margin_deg": float(cfg["safety"].get("state_stats_margin_deg", 5.0)),
                    "below_q01": [int(i) for i in below],
                    "above_q99": [int(i) for i in above],
                    "q01": q01.tolist(),
                    "q99": q99.tolist(),
                }
            )

    limits = cfg["safety"].get("target_limits_deg")
    if limits is not None:
        lower_live = _vector(limits["min"], name="safety.target_limits_deg.min")
        upper_live = _vector(limits["max"], name="safety.target_limits_deg.max")
        if cfg["robot"].get("model_state_units", "rad") == "rad":
            lower = np.deg2rad(lower_live)
            upper = np.deg2rad(upper_live)
        else:
            lower = lower_live
            upper = upper_live
        margin = np.deg2rad(float(cfg["safety"].get("state_limit_margin_deg", 5.0)))
        below = np.where(state < (lower - margin))[0]
        above = np.where(state > (upper + margin))[0]
        if below.size or above.size:
            issues.append(
                {
                    "kind": "state_outside_target_limits",
                    "margin_deg": float(cfg["safety"].get("state_limit_margin_deg", 5.0)),
                    "below_min": [int(i) for i in below],
                    "above_max": [int(i) for i in above],
                    "target_min_model": lower.tolist(),
                    "target_max_model": upper.tolist(),
                }
            )

    if not issues:
        return None

    return {
        "state_model": state.tolist(),
        "issues": issues,
    }


def _validate_config(root: Path, cfg: dict[str, Any]) -> None:
    _vector(cfg["calibration"]["gain_vector"], name="calibration.gain_vector")
    _vector(cfg["safety"]["max_delta_per_command_deg"], name="safety.max_delta_per_command_deg")
    if cfg["safety"].get("target_limits_deg") is not None:
        _vector(cfg["safety"]["target_limits_deg"]["min"], name="safety.target_limits_deg.min")
        _vector(cfg["safety"]["target_limits_deg"]["max"], name="safety.target_limits_deg.max")
    openpi_repo = _resolve_path(root, cfg["model"]["openpi_repo"])
    checkpoint = _resolve_path(root, cfg["model"]["checkpoint_path"])
    print(f"config: {cfg['name']}")
    print(f"openpi_repo: {openpi_repo}")
    print(f"checkpoint: {checkpoint}")
    print(f"config_name: {cfg['model']['config_name']}")
    print(f"right_wrist_source: {cfg['model'].get('right_wrist_source', 'wrist')}")
    print(f"prompt: {cfg['model']['prompt']}")
    print(f"gain_vector: {cfg['calibration']['gain_vector']}")
    print(f"max_delta_per_command_deg: {cfg['safety']['max_delta_per_command_deg']}")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _compat_calibration_dir() -> Path:
    return _repo_root() / "run_logs" / "openpi" / "_compat_calibration"


def _old_calibration_path(robot_id: str) -> Path:
    return _NEW_LEROBOT_CALIBRATION_DIR / f"{robot_id}.json"


def _looks_like_new_calibration(payload: dict[str, Any]) -> bool:
    return "motor_names" in payload and "calib_mode" in payload


def _convert_old_calibration(payload: dict[str, Any]) -> dict[str, Any]:
    motor_names = [name for name in _MOTOR_NAMES if name in payload]
    if len(motor_names) != len(_MOTOR_NAMES):
        missing = [name for name in _MOTOR_NAMES if name not in payload]
        raise ValueError(f"Old calibration file is missing joints: {missing}")
    calib_mode: list[str] = []
    drive_mode: list[int] = []
    homing_offset: list[int] = []
    start_pos: list[int] = []
    end_pos: list[int] = []

    for name in motor_names:
        joint = payload[name]
        range_min = int(joint["range_min"])
        range_max = int(joint["range_max"])
        start_pos.append(range_min)
        end_pos.append(range_max)
        drive_mode.append(int(joint.get("drive_mode", 0)))

        if name == "gripper":
            # Old SOFollower used RANGE_0_100 normalization for gripper.
            calib_mode.append("LINEAR")
            homing_offset.append(0)
        else:
            # Old SOFollower ignored homing_offset at runtime and instead normalized
            # body joints around the midpoint of [range_min, range_max]:
            #   degrees = (raw - mid) * 360 / 4095
            # New LeRobot DEGREE mode expects:
            #   degrees = (raw + homing_offset) * 360 / 4096
            # so choose homing_offset ~= -mid to preserve the old convention.
            mid = int(round((range_min + range_max) / 2.0))
            calib_mode.append("DEGREE")
            homing_offset.append(-mid)

    return {
        "motor_names": motor_names,
        "calib_mode": calib_mode,
        "drive_mode": drive_mode,
        "homing_offset": homing_offset,
        "start_pos": start_pos,
        "end_pos": end_pos,
    }


def _prepare_compat_calibration(robot_id: str) -> Path:
    source = _old_calibration_path(robot_id)
    if not source.is_file():
        raise FileNotFoundError(f"Expected existing SO-101 calibration at {source}")

    payload = _load_json(source)
    if _looks_like_new_calibration(payload):
        converted = payload
    else:
        converted = _convert_old_calibration(payload)
    out_dir = _compat_calibration_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "so101_follower.json"
    out_path.write_text(json.dumps(converted, indent=2), encoding="utf-8")
    return out_dir


def _open_front_video_writer(session_dir: Path, cfg: dict[str, Any]):
    import cv2
    w = int(cfg["robot"]["camera_width"])
    h = int(cfg["robot"]["camera_height"])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(session_dir / "rollout_front.mp4"), fourcc, 30.0, (w, h))


def _write_front_frame(writer, front_arr: np.ndarray) -> None:
    import cv2
    writer.write(cv2.cvtColor(_image_to_uint8_hwc(front_arr), cv2.COLOR_RGB2BGR))


def main() -> int:
    root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy-config", default=_DEFAULT_CONFIG)
    parser.add_argument("--openpi-repo", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--robot-port", default=None)
    parser.add_argument("--front-camera-index", default=None)
    parser.add_argument("--wrist-camera-index", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Run policy and safety logic but do not send robot commands.")
    parser.add_argument("--check-only", action="store_true", help="Validate config, OpenPI import, and checkpoint loading, then exit.")
    parser.add_argument("--skip-policy-load", action="store_true", help="For config checks only; do not import OpenPI or load weights.")
    args = parser.parse_args()

    config_path = _resolve_path(root, args.deploy_config)
    if config_path is None:
        raise ValueError("--deploy-config cannot be empty")
    cfg = _load_json(config_path)
    if args.openpi_repo:
        cfg["model"]["openpi_repo"] = args.openpi_repo
    if args.checkpoint:
        cfg["model"]["checkpoint_path"] = args.checkpoint
    if args.robot_port:
        cfg["robot"]["robot_port"] = args.robot_port
    if args.front_camera_index:
        cfg["robot"]["front_camera_index"] = args.front_camera_index
    if args.wrist_camera_index:
        cfg["robot"]["wrist_camera_index"] = args.wrist_camera_index
    if args.max_steps is not None:
        cfg["control"]["max_steps"] = args.max_steps

    _validate_config(root, cfg)
    policy = None
    state_stats = _load_state_stats(root, cfg)
    if not args.skip_policy_load:
        print("loading OpenPI policy...")
        policy = _load_policy(root, cfg)
        print("OpenPI policy loaded")
    if state_stats is not None:
        print(f"state_stats: {state_stats['path']}")
    if args.check_only:
        return 0
    if policy is None:
        raise ValueError("--skip-policy-load is only valid with --check-only")

    control = cfg["control"]
    log_dir = _resolve_path(root, control.get("log_dir", "run_logs/openpi"))
    if log_dir is None:
        raise ValueError("control.log_dir cannot be empty")
    session_dir = log_dir / f"{_stamp()}_{_checkpoint_name(cfg)}"
    session_dir.mkdir(parents=True, exist_ok=True)
    log_path = session_dir / "rollout.jsonl"
    (session_dir / "deployment_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    video_writer = _open_front_video_writer(session_dir, cfg)
    robot = _make_robot(cfg)
    step = 0
    try:
        try:
            _connect_robot(robot, control)
        except ConnectionError as e:
            message = str(e).lower()
            if (
                ("lock" in message or "enable_torque" in message or "status packet" in message)
                and hasattr(robot, "bus")
            ):
                print(f"WARNING: robot connect hit torque/status issue ({e}). Retrying torque enable...")
                robot.bus.enable_torque(num_retry=5)
            else:
                raise
        print(f"Robot connected. dry_run={args.dry_run}. Logs: {session_dir}")
        while True:
            obs = _read_observation(robot)
            policy_obs, state_raw, state_model = _obs_to_policy_input(obs, cfg)
            _write_front_frame(video_writer, policy_obs["observation/images/front"])
            state_issue = _state_guard_issue(state_model, cfg, state_stats)
            if control.get("save_latest_observation", True):
                np.save(session_dir / "latest_front.npy", policy_obs["observation/images/front"])
                np.save(session_dir / "latest_wrist.npy", policy_obs["observation/images/wrist"])
                np.save(session_dir / "latest_state_model.npy", state_model)
                _save_latest_observation_png(
                    session_dir,
                    policy_obs["observation/images/front"],
                    policy_obs["observation/images/wrist"],
                )
            if state_issue is not None:
                _append_jsonl(
                    log_path,
                    {
                        "step": step,
                        "dry_run": args.dry_run,
                        "state_raw": state_raw.tolist(),
                        "state_model": state_model.tolist(),
                        "guard": state_issue,
                    },
                )
                raise RuntimeError(
                    "Observed live state is outside the checkpoint training/safety envelope; "
                    "aborting before policy inference. See rollout.jsonl guard entry."
                )
            pred_abs = np.asarray(policy.infer(policy_obs)["actions"], dtype=np.float32)
            if pred_abs.ndim != 2 or pred_abs.shape[1] < 6:
                raise ValueError(f"Expected policy actions shape (T, >=6), got {pred_abs.shape}")
            if not np.isfinite(pred_abs).all() and cfg["safety"].get("abort_on_nonfinite", True):
                raise ValueError("Non-finite policy action detected")

            execute_steps = max(1, min(int(control["execute_steps_per_inference"]), pred_abs.shape[0]))
            for horizon_idx in range(execute_steps):
                if horizon_idx > 0:
                    obs = _read_observation(robot)
                    inner_obs, state_raw, state_model = _obs_to_policy_input(obs, cfg)
                    _write_front_frame(video_writer, inner_obs["observation/images/front"])
                    state_issue = _state_guard_issue(state_model, cfg, state_stats)
                    if state_issue is not None:
                        _append_jsonl(
                            log_path,
                            {
                                "step": step,
                                "horizon_idx": horizon_idx,
                                "dry_run": args.dry_run,
                                "state_raw": state_raw.tolist(),
                                "state_model": state_model.tolist(),
                                "guard": state_issue,
                            },
                        )
                        raise RuntimeError(
                            "Observed live state is outside the checkpoint training/safety envelope; "
                            "aborting before sending command. See rollout.jsonl guard entry."
                        )
                target_model, safety_info = _calibrate_and_clip(pred_abs[horizon_idx], state_model, cfg)
                target_live = _model_to_robot_units(target_model, cfg)
                pred_live_raw = _model_to_robot_units(np.asarray(safety_info["pred_abs_model"], dtype=np.float64), cfg)
                raw_delta_live = pred_live_raw - state_raw
                command_delta_live = target_live - state_raw
                if not np.isfinite(target_live).all() and cfg["safety"].get("abort_on_nonfinite", True):
                    raise ValueError("Non-finite clipped target detected")

                row = {
                    "step": step,
                    "horizon_idx": horizon_idx,
                    "dry_run": args.dry_run,
                    "state_raw": state_raw.tolist(),
                    "state_model": state_model.tolist(),
                    "target_model": target_model.tolist(),
                    "target_live": target_live.tolist(),
                    "pred_live_raw": pred_live_raw.tolist(),
                    "raw_delta_live": raw_delta_live.tolist(),
                    "command_delta_live": command_delta_live.tolist(),
                    "command": _action_vector_to_robot_dict(target_live),
                    "safety": safety_info,
                }
                _append_jsonl(log_path, row)
                print(
                    f"[step {step}] h={horizon_idx} "
                    f"state={np.round(state_raw, 2).tolist()} "
                    f"target={np.round(target_live, 2).tolist()} "
                    f"raw_delta={np.round(raw_delta_live, 2).tolist()} "
                    f"cmd_delta={np.round(command_delta_live, 2).tolist()} "
                    f"limited={safety_info['delta_limited'] or safety_info['target_limited']}"
                )
                if not args.dry_run:
                    _send_robot_action(robot, target_live)
                step += 1
                if int(control["max_steps"]) > 0 and step >= int(control["max_steps"]):
                    return 0
                period = float(control.get("control_period_s", 0.0))
                if period > 0:
                    time.sleep(period)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        video_writer.release()
        if getattr(robot, "is_connected", False):
            robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
