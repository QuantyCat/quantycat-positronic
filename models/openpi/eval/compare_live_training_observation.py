#!/usr/bin/env python3
"""Compare OpenPI predictions for matched training and live observations.

This is a diagnostic only. It does not connect to the robot and does not modify
calibration. It answers one question: when the policy predicts tiny live motion,
is that caused by the numeric state, the camera images, or both?
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path("/home/caroline/quantycat-positronic")
OPENPI_REPO = Path("/home/caroline/openpi")
DATASET_ROOT = REPO / "my_data/input_data"
CHECKPOINT = (
    REPO
    / "my_data/training_pipeline/openpi/checkpoints/pi05_quantycat_lora/"
    "screwdriver_so101_pi05_h20_lora_20260516_pdt/9999"
)
PROMPT = "Put the screwdriver into the cup"
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-repo", type=Path, default=OPENPI_REPO)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--config-name", default="pi05_quantycat_lora")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--frame", type=int, default=27)
    parser.add_argument("--live-run", type=Path, default=None, help="Directory containing latest_front.npy/latest_wrist.npy/latest_state_model.npy")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _load_policy(args: argparse.Namespace):
    src = args.openpi_repo / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"OpenPI src directory not found: {src}")
    if not (args.checkpoint / "params").is_dir():
        raise FileNotFoundError(f"Checkpoint params not found: {args.checkpoint / 'params'}")
    sys.path.insert(0, str(src))
    os.chdir(args.openpi_repo)

    from openpi.policies import policy_config
    from openpi.training import config as openpi_config

    train_config = openpi_config.get_config(args.config_name)
    return policy_config.create_trained_policy(
        train_config,
        args.checkpoint,
        sample_kwargs={"num_steps": args.sample_steps},
        default_prompt=args.prompt,
    )


def _ffprobe_size(video: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(video),
    ]
    payload = json.loads(subprocess.check_output(cmd, text=True))
    stream = payload["streams"][0]
    return int(stream["width"]), int(stream["height"])


def _decode_video_frame(video: Path, frame: int) -> np.ndarray:
    width, height = _ffprobe_size(video)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video),
        "-vf",
        f"select=eq(n\\,{frame})",
        "-vframes",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    raw = subprocess.check_output(cmd)
    expected = width * height * 3
    if len(raw) != expected:
        raise RuntimeError(f"Decoded {len(raw)} bytes from {video}, expected {expected}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)


def _as_vector(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.dtype == object:
        arr = np.asarray(list(value), dtype=np.float32)
    return arr.reshape(-1)


def _load_training_sample(dataset_root: Path, episode: int, frame: int) -> dict[str, Any]:
    episode_name = f"episode_{episode:06d}"
    parquet = dataset_root / "data/chunk-000" / f"{episode_name}.parquet"
    front_video = dataset_root / "videos/chunk-000/observation.images.front" / f"{episode_name}.mp4"
    wrist_video = dataset_root / "videos/chunk-000/observation.images.wrist" / f"{episode_name}.mp4"
    if not parquet.is_file():
        raise FileNotFoundError(parquet)
    if not front_video.is_file():
        raise FileNotFoundError(front_video)
    if not wrist_video.is_file():
        raise FileNotFoundError(wrist_video)

    df = pd.read_parquet(parquet)
    if frame < 0 or frame >= len(df):
        raise IndexError(f"{episode_name} has {len(df)} frames; requested {frame}")
    row = df.iloc[frame]
    return {
        "episode": episode_name,
        "frame": frame,
        "front": _decode_video_frame(front_video, frame),
        "wrist": _decode_video_frame(wrist_video, frame),
        "state": _as_vector(row["observation.state"]),
        "action": _as_vector(row["action"]) if "action" in row else None,
    }


def _load_live_sample(live_run: Path) -> dict[str, Any]:
    return {
        "front": np.load(live_run / "latest_front.npy"),
        "wrist": np.load(live_run / "latest_wrist.npy"),
        "state": np.load(live_run / "latest_state_model.npy").astype(np.float32),
    }


def _infer(policy, *, front: np.ndarray, wrist: np.ndarray, state: np.ndarray, prompt: str) -> np.ndarray:
    obs = {
        "observation/images/front": front,
        "observation/images/wrist": wrist,
        "observation/state": state.astype(np.float32),
        "prompt": prompt,
    }
    actions = np.asarray(policy.infer(obs)["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] < 6:
        raise ValueError(f"Expected policy actions shape (T, >=6), got {actions.shape}")
    return actions[:, :6]


def _summary(name: str, actions_abs: np.ndarray, state: np.ndarray) -> dict[str, Any]:
    deltas = actions_abs.copy()
    deltas[:, :5] -= state[:5].reshape(1, 5)
    payload = {
        "name": name,
        "state_deg": np.rad2deg(state[:6]).tolist(),
        "h0_abs_deg": np.rad2deg(actions_abs[0, :6]).tolist(),
        "h0_delta_deg": np.rad2deg(deltas[0, :6]).tolist(),
        "h9_delta_deg": np.rad2deg(deltas[min(9, len(deltas) - 1), :6]).tolist(),
        "chunk_delta_min_deg": np.rad2deg(deltas[:, :6].min(axis=0)).tolist(),
        "chunk_delta_max_deg": np.rad2deg(deltas[:, :6].max(axis=0)).tolist(),
    }
    return payload


def _print_summary(item: dict[str, Any]) -> None:
    print(f"\n{item['name']}")
    print("  state_deg:       " + _fmt(item["state_deg"]))
    print("  h0_delta_deg:    " + _fmt(item["h0_delta_deg"]))
    print("  h9_delta_deg:    " + _fmt(item["h9_delta_deg"]))
    print("  chunk_min_delta: " + _fmt(item["chunk_delta_min_deg"]))
    print("  chunk_max_delta: " + _fmt(item["chunk_delta_max_deg"]))


def _fmt(values: list[float]) -> str:
    return "[" + ", ".join(f"{v:7.2f}" for v in values) + "]"


def main() -> int:
    args = _parse_args()
    train = _load_training_sample(args.dataset_root, args.episode, args.frame)
    policy = _load_policy(args)

    print(f"training sample: {train['episode']} frame {train['frame']}")
    print(f"training state deg: {_fmt(np.rad2deg(train['state'][:6]).tolist())}")
    if train["action"] is not None:
        demo_delta = train["action"].copy()
        demo_delta[:5] -= train["state"][:5]
        print(f"dataset action delta deg: {_fmt(np.rad2deg(demo_delta[:6]).tolist())}")

    cases: list[dict[str, Any]] = []
    train_train = _infer(policy, front=train["front"], wrist=train["wrist"], state=train["state"], prompt=args.prompt)
    cases.append(_summary("A training images + training state", train_train, train["state"]))

    if args.live_run is not None:
        live = _load_live_sample(args.live_run)
        print(f"live run: {args.live_run}")
        print(f"live state deg:     {_fmt(np.rad2deg(live['state'][:6]).tolist())}")

        cases.append(
            _summary(
                "B training images + live state",
                _infer(policy, front=train["front"], wrist=train["wrist"], state=live["state"], prompt=args.prompt),
                live["state"],
            )
        )
        cases.append(
            _summary(
                "C live images + training state",
                _infer(policy, front=live["front"], wrist=live["wrist"], state=train["state"], prompt=args.prompt),
                train["state"],
            )
        )
        cases.append(
            _summary(
                "D live images + live state",
                _infer(policy, front=live["front"], wrist=live["wrist"], state=live["state"], prompt=args.prompt),
                live["state"],
            )
        )

    for item in cases:
        _print_summary(item)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({"cases": cases}, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output_json}")

    print("\nInterpretation:")
    print("  If A and B are similar, state is not the main cause.")
    print("  If A and C differ, live images/camera preprocessing are the main cause.")
    print("  If C and D differ, live state is contributing on top of the images.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
