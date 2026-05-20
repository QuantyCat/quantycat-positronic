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
import time


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
    return parser.parse_args()


def _make_robot(args: argparse.Namespace):
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.robots.utils import make_robot_from_config

    cameras = {} if args.no_cameras else {
        "front": OpenCVCameraConfig(index_or_path=2, fps=30, width=640, height=360),
        "wrist": OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=360),
    }
    config = SO101FollowerConfig(
        port=args.port,
        id="so101_follower",
        cameras=cameras,
    )
    return make_robot_from_config(config)


def main() -> int:
    args = _parse_args()

    print("Connecting to robot ...")
    robot = _make_robot(args)
    robot.connect(calibrate=not args.no_calibrate)
    print("Connected.\n")

    # ------------------------------------------------------------------ #
    # 1. Read observation and show all keys                               #
    # ------------------------------------------------------------------ #
    obs = robot.get_observation()
    print("Observation keys:")
    for k, v in obs.items():
        try:
            shape = getattr(v, "shape", None)
            print(f"  {k!r:40s} = {v if shape is None else f'array{shape}'}")
        except Exception:
            print(f"  {k!r}")

    # ------------------------------------------------------------------ #
    # 2. Find the joint key in the observation                            #
    # ------------------------------------------------------------------ #
    joint = args.joint
    candidates = [k for k in obs if joint in k]
    if not candidates:
        print(f"\nERROR: no key containing {joint!r} found in observation.")
        print("Available keys:", [k for k in obs if "pos" in k or "state" in k])
        robot.disconnect()
        return 1

    joint_key = candidates[0]
    current_pos = float(obs[joint_key])
    target_pos = current_pos + args.delta
    print(f"\nJoint key: {joint_key!r}")
    print(f"Current position: {current_pos:.3f} deg")
    print(f"Target  position: {target_pos:.3f} deg  (delta={args.delta:+.1f} deg)")

    # ------------------------------------------------------------------ #
    # 3. Send command and report                                          #
    # ------------------------------------------------------------------ #
    print(f"\nSending command {{'{joint_key}': {target_pos:.3f}}} ...")
    result = robot.send_action({joint_key: target_pos})
    print(f"send_action returned: {result}")

    print(f"Waiting {args.wait}s for servo to move ...")
    time.sleep(args.wait)

    obs2 = robot.get_observation()
    actual_pos = float(obs2[joint_key])
    moved = actual_pos - current_pos
    print(f"\nAfter {args.wait}s:")
    print(f"  Position: {actual_pos:.3f} deg  (moved {moved:+.3f} deg, "
          f"target was {args.delta:+.1f} deg, tracking = {100*moved/args.delta:.0f}%)")

    if abs(moved) < 0.2:
        print("\n[!] Servo did NOT move. Possible causes:")
        print("    - send_action key format mismatch")
        print("    - torque not enabled")
        print("    - servo speed register set to 0 or very low")
    elif abs(moved) < abs(args.delta) * 0.5:
        print("\n[!] Servo moved but only partially. Speed/period may be limiting.")
    else:
        print("\n[OK] Servo responded correctly.")

    # ------------------------------------------------------------------ #
    # 4. Read and optionally reset the Moving_Speed register             #
    # ------------------------------------------------------------------ #
    try:
        bus = getattr(robot, "bus", None) or getattr(robot, "motors", None)
        if bus is not None:
            speed_val = bus.read("Moving_Speed", joint)
            print(f"\nMoving_Speed register for {joint}: {speed_val}")
            print("  (0 = max speed, higher value = slower; default should be 0)")
            if speed_val != 0:
                print(f"  [!] Speed is LIMITED. Setting to 0 (max) and re-testing ...")
                bus.write("Moving_Speed", 0, joint)
                time.sleep(0.2)
                obs3 = robot.get_observation()
                pos_before_reset = float(obs3[joint_key])
                result2 = robot.send_action({joint_key: pos_before_reset + args.delta})
                time.sleep(args.wait)
                obs4 = robot.get_observation()
                moved2 = float(obs4[joint_key]) - pos_before_reset
                print(f"  After reset to max speed: moved {moved2:+.3f} deg "
                      f"(tracking = {100*moved2/args.delta:.0f}%)")
            else:
                print("  Speed register is already at max (0). Speed limit is elsewhere.")
        else:
            print("\n(Could not access motor bus to read Moving_Speed register)")
    except Exception as e:
        print(f"\n(Moving_Speed read failed: {e})")

    robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
