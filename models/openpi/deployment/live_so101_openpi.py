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
_DEFAULT_CONFIG = "models/openpi/deployment/pi05_lora_step9999_so101.json"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
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
    if front_key not in obs or wrist_key not in obs:
        raise KeyError(f"Observation missing camera keys {front_key!r}/{wrist_key!r}; got {sorted(obs)}")

    front = np.asarray(obs[front_key])
    wrist = np.asarray(obs[wrist_key])
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

    limits = safety["target_limits_deg"]
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
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.robots.utils import make_robot_from_config

    robot_cfg = cfg["robot"]
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

    train_config = openpi_config.get_config(cfg["model"]["config_name"])
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


def _validate_config(root: Path, cfg: dict[str, Any]) -> None:
    _vector(cfg["calibration"]["gain_vector"], name="calibration.gain_vector")
    _vector(cfg["safety"]["max_delta_per_command_deg"], name="safety.max_delta_per_command_deg")
    _vector(cfg["safety"]["target_limits_deg"]["min"], name="safety.target_limits_deg.min")
    _vector(cfg["safety"]["target_limits_deg"]["max"], name="safety.target_limits_deg.max")
    openpi_repo = _resolve_path(root, cfg["model"]["openpi_repo"])
    checkpoint = _resolve_path(root, cfg["model"]["checkpoint_path"])
    print(f"config: {cfg['name']}")
    print(f"openpi_repo: {openpi_repo}")
    print(f"checkpoint: {checkpoint}")
    print(f"config_name: {cfg['model']['config_name']}")
    print(f"prompt: {cfg['model']['prompt']}")
    print(f"gain_vector: {cfg['calibration']['gain_vector']}")
    print(f"max_delta_per_command_deg: {cfg['safety']['max_delta_per_command_deg']}")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


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
    if not args.skip_policy_load:
        print("loading OpenPI policy...")
        policy = _load_policy(root, cfg)
        print("OpenPI policy loaded")
    if args.check_only:
        return 0
    if policy is None:
        raise ValueError("--skip-policy-load is only valid with --check-only")

    control = cfg["control"]
    log_dir = _resolve_path(root, control.get("log_dir", "run_logs/openpi_live_so101"))
    if log_dir is None:
        raise ValueError("control.log_dir cannot be empty")
    session_dir = log_dir / _stamp()
    session_dir.mkdir(parents=True, exist_ok=True)
    log_path = session_dir / "rollout.jsonl"
    (session_dir / "deployment_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    robot = _make_robot(cfg)
    step = 0
    try:
        try:
            robot.connect(calibrate=bool(control.get("connect_calibrate", True)))
        except ConnectionError as e:
            message = str(e).lower()
            if "lock" in message or "enable_torque" in message or "status packet" in message:
                print(f"WARNING: robot connect hit torque/status issue ({e}). Retrying torque enable...")
                robot.bus.enable_torque(num_retry=5)
            else:
                raise
        print(f"Robot connected. dry_run={args.dry_run}. Logs: {session_dir}")
        while True:
            obs = robot.get_observation()
            policy_obs, state_raw, state_model = _obs_to_policy_input(obs, cfg)
            if control.get("save_latest_observation", True):
                np.save(session_dir / "latest_front.npy", policy_obs["observation/images/front"])
                np.save(session_dir / "latest_wrist.npy", policy_obs["observation/images/wrist"])
                np.save(session_dir / "latest_state_model.npy", state_model)
            pred_abs = np.asarray(policy.infer(policy_obs)["actions"], dtype=np.float32)
            if pred_abs.ndim != 2 or pred_abs.shape[1] < 6:
                raise ValueError(f"Expected policy actions shape (T, >=6), got {pred_abs.shape}")
            if not np.isfinite(pred_abs).all() and cfg["safety"].get("abort_on_nonfinite", True):
                raise ValueError("Non-finite policy action detected")

            execute_steps = max(1, min(int(control["execute_steps_per_inference"]), pred_abs.shape[0]))
            for horizon_idx in range(execute_steps):
                if horizon_idx > 0:
                    obs = robot.get_observation()
                    _, state_raw, state_model = _obs_to_policy_input(obs, cfg)
                target_model, safety_info = _calibrate_and_clip(pred_abs[horizon_idx], state_model, cfg)
                target_live = _model_to_robot_units(target_model, cfg)
                if not np.isfinite(target_live).all() and cfg["safety"].get("abort_on_nonfinite", True):
                    raise ValueError("Non-finite clipped target detected")

                command = _action_vector_to_robot_dict(target_live)
                row = {
                    "step": step,
                    "horizon_idx": horizon_idx,
                    "dry_run": args.dry_run,
                    "state_raw": state_raw.tolist(),
                    "state_model": state_model.tolist(),
                    "target_model": target_model.tolist(),
                    "target_live": target_live.tolist(),
                    "command": command,
                    "safety": safety_info,
                }
                _append_jsonl(log_path, row)
                print(
                    f"[step {step}] h={horizon_idx} "
                    f"state={np.round(state_raw, 2).tolist()} "
                    f"target={np.round(target_live, 2).tolist()} "
                    f"limited={safety_info['delta_limited'] or safety_info['target_limited']}"
                )
                if not args.dry_run:
                    robot.send_action(command)
                step += 1
                if int(control["max_steps"]) > 0 and step >= int(control["max_steps"]):
                    return 0
                period = float(control.get("control_period_s", 0.0))
                if period > 0:
                    time.sleep(period)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if getattr(robot, "is_connected", False):
            robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
