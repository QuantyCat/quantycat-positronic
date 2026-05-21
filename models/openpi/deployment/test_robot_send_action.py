#!/usr/bin/env python3
"""Diagnostic: verify send_action reaches the servos.

Connects to the SO-101, reads an observation, prints all observation keys,
then commands wrist_flex +10 deg, waits, reads again, and reports actual movement.

Run from the quantycat-positronic repo root:
  python models/openpi/deployment/test_robot_send_action.py

Options:
  --port        Serial port (default: /dev/ttyACM0)
  --delta       Degrees to move wrist_flex (default: 10.0)
  --joint       Joint name to test (default: wrist_flex)
  --wait        Seconds to wait after sending command (default: 1.0)
  --no-cameras  Skip camera initialisation (faster, motor-only test)
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
_START_POSE_DEG = {
    "shoulder_pan": 5.0,
    "shoulder_lift": -100.0,
    "elbow_flex": 95.0,
    "wrist_flex": 71.0,
    "wrist_roll": 3.0,
    "gripper": 0.5,
}
_NEW_LEROBOT_CALIBRATION_DIR = Path("/home/caroline/.cache/huggingface/lerobot/calibration/robots/so_follower")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _bootstrap_vendor_imports() -> None:
    vendor = _repo_root() / "_vendor"
    if vendor.is_dir():
        sys.path.insert(0, str(vendor))


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _compat_calibration_dir() -> Path:
    return _repo_root() / "run_logs" / "openpi_live_so101" / "_compat_calibration"


def _old_calibration_path(robot_id: str) -> Path:
    return _NEW_LEROBOT_CALIBRATION_DIR / f"{robot_id}.json"


def _looks_like_new_calibration(payload: dict) -> bool:
    return "motor_names" in payload and "calib_mode" in payload


def _convert_old_calibration(payload: dict) -> dict:
    motor_names = [name for name in _MOTOR_NAMES if name in payload]
    if len(motor_names) != len(_MOTOR_NAMES):
        missing = [name for name in _MOTOR_NAMES if name not in payload]
        raise ValueError(f"Old calibration file is missing joints: {missing}")

    calib_mode = []
    drive_mode = []
    homing_offset = []
    start_pos = []
    end_pos = []
    for name in motor_names:
        joint = payload[name]
        range_min = int(joint["range_min"])
        range_max = int(joint["range_max"])
        start_pos.append(range_min)
        end_pos.append(range_max)
        drive_mode.append(int(joint.get("drive_mode", 0)))

        if name == "gripper":
            calib_mode.append("LINEAR")
            homing_offset.append(0)
        else:
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
    converted = payload if _looks_like_new_calibration(payload) else _convert_old_calibration(payload)
    out_dir = _compat_calibration_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "so101_follower.json").write_text(json.dumps(converted, indent=2), encoding="utf-8")
    return out_dir


def _read_observation(robot):
    if hasattr(robot, "get_observation"):
        return robot.get_observation()
    if hasattr(robot, "capture_observation"):
        return robot.capture_observation()
    raise AttributeError("Robot object does not provide get_observation() or capture_observation().")


def _extract_state_vector(obs: dict) -> list[float]:
    if "observation.state" in obs:
        return [float(value) for value in obs["observation.state"]]
    return [float(obs[f"{motor}.pos"]) for motor in _MOTOR_NAMES]


def _find_joint_position(obs: dict, joint: str) -> tuple[str, float]:
    candidates = [key for key in obs if joint in key]
    if candidates:
        joint_key = candidates[0]
        return joint_key, float(obs[joint_key])

    if "observation.state" in obs and joint in _MOTOR_NAMES:
        joint_idx = _MOTOR_NAMES.index(joint)
        return f"observation.state[{joint_idx}]", float(obs["observation.state"][joint_idx])

    raise KeyError(joint)


def _send_joint_action(robot, obs: dict, joint: str, joint_key: str, target_pos: float):
    if hasattr(robot, "get_observation"):
        return robot.send_action({joint_key: target_pos})

    import torch

    state = _extract_state_vector(obs)
    joint_idx = _MOTOR_NAMES.index(joint)
    state[joint_idx] = float(target_pos)
    return robot.send_action(torch.as_tensor(state, dtype=torch.float32))


def _joint_positions(obs: dict) -> dict[str, float]:
    return {name: _find_joint_position(obs, name)[1] for name in _MOTOR_NAMES}


def _print_joint_positions(title: str, positions: dict[str, float]) -> None:
    print(f"\n{title}:")
    for name in _MOTOR_NAMES:
        value = positions.get(name)
        if value is None:
            print(f"  {name:14s} = <unavailable>")
        else:
            print(f"  {name:14s} = {value:.3f} deg")


def _send_pose_action(robot, obs: dict, targets_deg: dict[str, float]):
    if hasattr(robot, "get_observation"):
        return robot.send_action({f"{name}.pos": float(targets_deg[name]) for name in _MOTOR_NAMES})

    import torch

    target = [float(targets_deg[name]) for name in _MOTOR_NAMES]
    return robot.send_action(torch.as_tensor(target, dtype=torch.float32))


def _bus_read_compat(bus, register: str, joint: str):
    try:
        return bus.read(register, joint, normalize=False)
    except TypeError:
        value = bus.read(register, joint)
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, list):
            return value[0]
        if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
            try:
                return value[0]
            except Exception:
                pass
        return value


def _bus_write_compat(bus, register: str, joint: str, value: int):
    try:
        bus.write(register, joint, value, normalize=False)
    except TypeError:
        bus.write(register, value, joint)


def _bus_register_names(bus) -> set[str]:
    model_ctrl_table = getattr(bus, "model_ctrl_table", None)
    motors = getattr(bus, "motors", None)
    if not isinstance(model_ctrl_table, dict) or not isinstance(motors, dict) or not motors:
        return set()

    register_names: set[str] = set()
    for _, model in motors.values():
        ctrl_table = model_ctrl_table.get(model, {})
        if isinstance(ctrl_table, dict):
            register_names.update(ctrl_table)
    return register_names


def _pick_first_supported(bus, *registers: str) -> str | None:
    supported = _bus_register_names(bus)
    if not supported:
        return registers[0] if registers else None
    for register in registers:
        if register in supported:
            return register
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--delta", type=float, default=10.0,
                        help="Degrees to move the test joint")
    parser.add_argument("--joint", default="wrist_flex",
                        help="Joint to move")
    parser.add_argument("--wait", type=float, default=1.0,
                        help="Seconds to wait after sending command")
    parser.add_argument("--no-cameras", action="store_true",
                        help="Skip cameras (motor-only test)")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="Connect without calibrate=True (test if calibration sets speed limit)")
    parser.add_argument(
        "--start-pose",
        nargs="*",
        type=float,
        metavar="DEG",
        help=(
            "Move all joints to a target pose and exit. "
            "Pass 6 values in degrees, or pass no values to use the standard OpenPI start pose."
        ),
    )
    args = parser.parse_args()
    if args.start_pose is not None and len(args.start_pose) not in (0, 6):
        parser.error("--start-pose expects either 0 values or exactly 6 values")
    return args


def _make_robot(args: argparse.Namespace):
    cameras = {}
    try:
        from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
        from lerobot.robots.utils import make_robot_from_config

        if not args.no_cameras:
            try:
                from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

                cameras = {
                    "front": OpenCVCameraConfig(index_or_path=2, fps=30, width=640, height=360),
                    "wrist": OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=360),
                }
            except ModuleNotFoundError:
                from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig

                cameras = {
                    "front": OpenCVCameraConfig(camera_index=2, fps=30, width=640, height=360),
                    "wrist": OpenCVCameraConfig(camera_index=0, fps=30, width=640, height=360),
                }
        config = SO101FollowerConfig(
            port=args.port,
            id="so101_follower",
            cameras=cameras,
        )
        return make_robot_from_config(config)
    except ModuleNotFoundError:
        from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
        from lerobot.common.robot_devices.robots.configs import So101RobotConfig
        from lerobot.common.robot_devices.robots.utils import make_robot_from_config

        try:
            from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig

            if not args.no_cameras:
                cameras = {
                    "front": OpenCVCameraConfig(camera_index=2, fps=30, width=640, height=360),
                    "wrist": OpenCVCameraConfig(camera_index=0, fps=30, width=640, height=360),
                }
        except ModuleNotFoundError:
            cameras = {}

        follower_bus = FeetechMotorsBusConfig(
            port=args.port,
            motors={name: [idx, "sts3215"] for idx, name in enumerate(_MOTOR_NAMES, start=1)},
        )
        config = So101RobotConfig(
            calibration_dir=str(_prepare_compat_calibration("so101_follower")),
            leader_arms={},
            follower_arms={"so101": follower_bus},
            cameras=cameras,
        )
        return make_robot_from_config(config)


def main() -> int:
    args = _parse_args()
    _bootstrap_vendor_imports()

    print("Connecting to robot ...")
    robot = _make_robot(args)
    try:
        robot.connect(calibrate=not args.no_calibrate)
    except TypeError:
        robot.connect()
    print("Connected.\n")

    # ------------------------------------------------------------------ #
    # 1. Read observation and show all keys                               #
    # ------------------------------------------------------------------ #
    obs = _read_observation(robot)
    print("Observation keys:")
    for k, v in obs.items():
        try:
            shape = getattr(v, "shape", None)
            print(f"  {k!r:40s} = {v if shape is None else f'array{shape}'}")
        except Exception:
            print(f"  {k!r}")

    current_positions = _joint_positions(obs)

    if args.start_pose is not None:
        target_pose = _START_POSE_DEG.copy()
        if len(args.start_pose) == 6:
            target_pose = {name: float(args.start_pose[idx]) for idx, name in enumerate(_MOTOR_NAMES)}
        _print_joint_positions("Current joint positions", current_positions)
        print("\nSending start pose:")
        for name in _MOTOR_NAMES:
            print(f"  {name:14s} -> {target_pose[name]:.3f} deg")
        result = _send_pose_action(robot, obs, target_pose)
        print(f"send_action returned: {result}")
        print(f"Waiting {args.wait}s for robot to settle ...")
        time.sleep(args.wait)
        obs2 = _read_observation(robot)
        actual_positions = _joint_positions(obs2)
        _print_joint_positions("Joint positions after move", actual_positions)
        robot.disconnect()
        return 0

    if abs(args.delta) < 1e-9:
        _print_joint_positions("Current joint positions", current_positions)
        print("\nRead-only check requested (delta=0); skipping send_action.")
        robot.disconnect()
        return 0

    # ------------------------------------------------------------------ #
    # 2. Find the joint key in the observation                            #
    # ------------------------------------------------------------------ #
    joint = args.joint
    try:
        joint_key, current_pos = _find_joint_position(obs, joint)
    except KeyError:
        print(f"\nERROR: no key containing {joint!r} found in observation.")
        print("Available keys:", [k for k in obs if "pos" in k or "state" in k])
        robot.disconnect()
        return 1
    target_pos = current_pos + args.delta
    print(f"\nJoint key: {joint_key!r}")
    print(f"Current position: {current_pos:.3f} deg")
    print(f"Target  position: {target_pos:.3f} deg  (delta={args.delta:+.1f} deg)")

    # ------------------------------------------------------------------ #
    # 3. Send command and report                                          #
    # ------------------------------------------------------------------ #
    print(f"\nSending command {{'{joint_key}': {target_pos:.3f}}} ...")
    result = _send_joint_action(robot, obs, joint, joint_key, target_pos)
    print(f"send_action returned: {result}")

    print(f"Waiting {args.wait}s for servo to move ...")
    time.sleep(args.wait)

    obs2 = _read_observation(robot)
    _, actual_pos = _find_joint_position(obs2, joint)
    moved = actual_pos - current_pos
    print(f"\nAfter {args.wait}s:")
    print(f"  Position: {actual_pos:.3f} deg  (moved {moved:+.3f} deg, "
          f"target was {args.delta:+.1f} deg, tracking = {100*moved/args.delta:.0f}%)")

    if abs(moved) < 0.2:
        print("\n[!] Servo did NOT move. Possible causes:")
        print("    - send_action key format mismatch")
        print("    - torque not enabled")
        print("    - motion-profile registers are limiting the move")
    elif abs(moved) < abs(args.delta) * 0.5:
        print("\n[!] Servo moved but only partially. Speed/period may be limiting.")
    else:
        print("\n[OK] Servo responded correctly.")

    # ------------------------------------------------------------------ #
    # 4. Read and optionally reset the STS3215 motion-profile registers  #
    # ------------------------------------------------------------------ #
    try:
        bus = getattr(robot, "bus", None) or getattr(robot, "motors", None)
        if bus is None:
            follower_arms = getattr(robot, "follower_arms", None)
            if isinstance(follower_arms, dict) and follower_arms:
                bus = next(iter(follower_arms.values()))
        if bus is not None:
            goal_velocity_register = _pick_first_supported(bus, "Goal_Velocity", "Goal_Speed")
            present_velocity_register = _pick_first_supported(bus, "Present_Velocity", "Present_Speed")

            goal_velocity = None
            if goal_velocity_register is not None:
                goal_velocity = _bus_read_compat(bus, goal_velocity_register, joint)
            goal_time = _bus_read_compat(bus, "Goal_Time", joint)
            present_velocity = None
            if present_velocity_register is not None:
                present_velocity = _bus_read_compat(bus, present_velocity_register, joint)
            moving = _bus_read_compat(bus, "Moving", joint)
            acceleration = _bus_read_compat(bus, "Acceleration", joint)
            max_acceleration = _bus_read_compat(bus, "Maximum_Acceleration", joint)

            print(f"\nSTS3215 motion-profile registers for {joint}:")
            if goal_velocity_register is not None:
                print(f"  {goal_velocity_register:18s} = {goal_velocity}")
            else:
                print("  Goal_Velocity/Speed = (not exposed by this Feetech backend)")
            print(f"  Goal_Time          = {goal_time}")
            if present_velocity_register is not None:
                print(f"  {present_velocity_register:18s} = {present_velocity}")
            else:
                print("  Present_Velocity/Speed = (not exposed by this Feetech backend)")
            print(f"  Moving             = {moving}")
            print(f"  Acceleration       = {acceleration}")
            print(f"  Maximum_Acceleration = {max_acceleration}")
            if goal_velocity_register == "Goal_Speed":
                print("  (On this backend, Goal_Speed=0 and Goal_Time=0 mean 'unconstrained/default')")
            else:
                print("  (For STS3215, Goal_Velocity=0 and Goal_Time=0 mean 'unconstrained/default')")

            if (goal_velocity is not None and goal_velocity != 0) or goal_time != 0:
                velocity_label = goal_velocity_register or "Goal_Velocity"
                print(f"  [!] Motion profile is LIMITED. Resetting {velocity_label} and Goal_Time to 0 and re-testing ...")
                if goal_velocity_register is not None:
                    _bus_write_compat(bus, goal_velocity_register, joint, 0)
                _bus_write_compat(bus, "Goal_Time", joint, 0)
                time.sleep(0.2)
                obs3 = _read_observation(robot)
                _, pos_before_reset = _find_joint_position(obs3, joint)
                result2 = _send_joint_action(robot, obs3, joint, joint_key, pos_before_reset + args.delta)
                time.sleep(args.wait)
                obs4 = _read_observation(robot)
                _, pos_after_reset = _find_joint_position(obs4, joint)
                moved2 = pos_after_reset - pos_before_reset
                print(f"  After reset to max speed: moved {moved2:+.3f} deg "
                      f"(tracking = {100*moved2/args.delta:.0f}%)")
            else:
                print("  Motion-profile registers are already at defaults. Speed limit is elsewhere.")
        else:
            print("\n(Could not access motor bus to read STS3215 motion-profile registers)")
    except Exception as e:
        print(f"\n(STS3215 motion-profile read failed: {e})")

    robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
