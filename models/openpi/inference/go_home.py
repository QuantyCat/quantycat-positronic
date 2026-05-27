#!/usr/bin/env python3
"""Move the SO-101 back to the home/rest position.

Interpolates from the current pose to the target in small steps so the
arm moves smoothly rather than jumping.

Run from the quantycat-positronic repo root:
  python models/openpi/inference/go_home.py

Options:
  --port     Serial port (default: /dev/ttyACM0)
  --steps    Number of interpolation steps (default: 40)
  --period   Seconds between steps (default: 0.05)
  --home     6 joint values in degrees to use instead of the default pose
             order: shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_roll gripper
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


_MOTOR_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

_HOME_POSE_DEG: dict[str, float] = {
    "shoulder_pan":  5.0,
    "shoulder_lift": -100.0,
    "elbow_flex":    95.0,
    "wrist_flex":    71.0,
    "wrist_roll":    3.0,
    "gripper":       0.5,
}

_NEW_LEROBOT_CALIBRATION_DIR = Path("/home/caroline/.cache/huggingface/lerobot/calibration/robots/so_follower")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _compat_calibration_dir() -> Path:
    return _repo_root() / "run_logs" / "openpi" / "_compat_calibration"


def _looks_like_new_calibration(payload: dict) -> bool:
    return "motor_names" in payload and "calib_mode" in payload


def _convert_old_calibration(payload: dict) -> dict:
    motor_names = [name for name in _MOTOR_NAMES if name in payload]
    if len(motor_names) != len(_MOTOR_NAMES):
        missing = [name for name in _MOTOR_NAMES if name not in payload]
        raise ValueError(f"Old calibration file is missing joints: {missing}")
    calib_mode, drive_mode, homing_offset, start_pos, end_pos = [], [], [], [], []
    for name in motor_names:
        joint = payload[name]
        rmin, rmax = int(joint["range_min"]), int(joint["range_max"])
        start_pos.append(rmin)
        end_pos.append(rmax)
        drive_mode.append(int(joint.get("drive_mode", 0)))
        if name == "gripper":
            calib_mode.append("LINEAR")
            homing_offset.append(0)
        else:
            mid = int(round((rmin + rmax) / 2.0))
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
    source = _NEW_LEROBOT_CALIBRATION_DIR / f"{robot_id}.json"
    if not source.is_file():
        raise FileNotFoundError(f"Expected existing SO-101 calibration at {source}")
    payload = _load_json(source)
    converted = payload if _looks_like_new_calibration(payload) else _convert_old_calibration(payload)
    out_dir = _compat_calibration_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "so101_follower.json").write_text(json.dumps(converted, indent=2), encoding="utf-8")
    return out_dir


def _make_robot(port: str):
    try:
        from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
        from lerobot.robots.utils import make_robot_from_config

        config = SO101FollowerConfig(port=port, id="so101_follower", cameras={})
        return make_robot_from_config(config)
    except ModuleNotFoundError:
        from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
        from lerobot.common.robot_devices.robots.configs import So101RobotConfig
        from lerobot.common.robot_devices.robots.utils import make_robot_from_config

        follower_bus = FeetechMotorsBusConfig(
            port=port,
            motors={name: [idx, "sts3215"] for idx, name in enumerate(_MOTOR_NAMES, start=1)},
        )
        config = So101RobotConfig(
            calibration_dir=str(_prepare_compat_calibration("so101_follower")),
            leader_arms={},
            follower_arms={"so101": follower_bus},
            cameras={},
        )
        return make_robot_from_config(config)


def _read_observation(robot) -> dict:
    if hasattr(robot, "get_observation"):
        return robot.get_observation()
    if hasattr(robot, "capture_observation"):
        return robot.capture_observation()
    raise AttributeError("Robot does not provide get_observation() or capture_observation().")


def _get_state_deg(obs: dict) -> dict[str, float]:
    if "observation.state" in obs:
        vec = list(obs["observation.state"])
        return {name: float(vec[i]) for i, name in enumerate(_MOTOR_NAMES)}
    return {name: float(obs[f"{name}.pos"]) for name in _MOTOR_NAMES}


def _send_pose(robot, obs: dict, targets: dict[str, float]) -> None:
    if hasattr(robot, "get_observation"):
        robot.send_action({f"{name}.pos": float(targets[name]) for name in _MOTOR_NAMES})
        return
    import torch
    robot.send_action(torch.as_tensor([float(targets[name]) for name in _MOTOR_NAMES], dtype=torch.float32))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--steps", type=int, default=40, help="Interpolation steps")
    parser.add_argument("--period", type=float, default=0.05, help="Seconds between steps")
    parser.add_argument(
        "--home",
        nargs=6,
        type=float,
        metavar="DEG",
        help="Custom home pose: shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_roll gripper",
    )
    args = parser.parse_args()

    target = _HOME_POSE_DEG.copy()
    if args.home:
        target = {name: float(args.home[i]) for i, name in enumerate(_MOTOR_NAMES)}

    print("Connecting to robot (no cameras) ...")
    robot = _make_robot(args.port)
    try:
        robot.connect(calibrate=True)
    except TypeError:
        robot.connect()
    print("Connected.\n")

    obs = _read_observation(robot)
    current = _get_state_deg(obs)

    print("Current pose (deg):")
    for name in _MOTOR_NAMES:
        print(f"  {name:14s} = {current[name]:7.2f}")
    print("\nTarget home pose (deg):")
    for name in _MOTOR_NAMES:
        print(f"  {name:14s} = {target[name]:7.2f}")

    duration = args.steps * args.period
    print(f"\nMoving to home in {args.steps} steps over ~{duration:.1f}s ...")

    try:
        for i in range(1, args.steps + 1):
            alpha = i / args.steps
            interp = {name: current[name] + alpha * (target[name] - current[name]) for name in _MOTOR_NAMES}
            _send_pose(robot, obs, interp)
            time.sleep(args.period)
            sys.stdout.write(f"\r  step {i}/{args.steps}")
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nInterrupted — stopping at current position.")
        robot.disconnect()
        return 1

    print("\nDone. Reading final position ...")
    obs2 = _read_observation(robot)
    final = _get_state_deg(obs2)
    print("\nFinal pose (deg):")
    for name in _MOTOR_NAMES:
        err = final[name] - target[name]
        print(f"  {name:14s} = {final[name]:7.2f}  (err {err:+.2f})")

    robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
