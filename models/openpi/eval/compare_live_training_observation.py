#!/usr/bin/env python3
"""Compare OpenPI predictions for matched training and live observations.

This is a diagnostic only. It does not connect to the robot and does not modify
calibration. It answers one question: when the policy predicts tiny live motion,
is that caused by the numeric state, the camera images, or both?

It also tests three right_wrist_0_rgb slot-mapping variants:
  wrist  — duplicate wrist into both wrist slots (current behaviour)
  front  — put the front/base view in the right-wrist slot (Illia-style)
  zeros  — black image + mask=False (UR5-style: slot absent)
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
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
    parser.add_argument(
        "--live-run",
        type=Path,
        default=None,
        help="Directory containing latest_front.npy/latest_wrist.npy/latest_state_model.npy",
    )
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--skip-slot-variants",
        action="store_true",
        help="Skip the right-wrist slot-mapping ablation (faster).",
    )
    return parser.parse_args()


def _bootstrap_openpi(args: argparse.Namespace) -> None:
    src = args.openpi_repo / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"OpenPI src directory not found: {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    os.chdir(args.openpi_repo)


def _load_policy(args: argparse.Namespace, right_wrist_source: str = "wrist"):
    from openpi.policies import policy_config
    from openpi.training import config as openpi_config
    from openpi.training.config import LeRobotQuantycatDataConfig

    train_config = openpi_config.get_config(args.config_name)

    if right_wrist_source != "wrist":
        # Build a variant config with a different slot mapping.
        variant_data = dataclasses.replace(
            train_config.data, right_wrist_source=right_wrist_source
        )
        train_config = dataclasses.replace(train_config, data=variant_data)

    if not (args.checkpoint / "params").is_dir():
        raise FileNotFoundError(f"Checkpoint params not found: {args.checkpoint / 'params'}")

    return policy_config.create_trained_policy(
        train_config,
        args.checkpoint,
        sample_kwargs={"num_steps": args.sample_steps},
        default_prompt=args.prompt,
    )


def _ffprobe_size(video: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", str(video),
    ]
    payload = json.loads(subprocess.check_output(cmd, text=True))
    stream = payload["streams"][0]
    return int(stream["width"]), int(stream["height"])


def _decode_video_frame(video: Path, frame: int) -> np.ndarray:
    width, height = _ffprobe_size(video)
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(video),
        "-vf", f"select=eq(n\\,{frame})",
        "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
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
    for p in (parquet, front_video, wrist_video):
        if not p.is_file():
            raise FileNotFoundError(p)

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
    h_idx = min(9, len(deltas) - 1)
    chunk_max = np.rad2deg(deltas[:, :6]).max(axis=0).tolist()
    chunk_min = np.rad2deg(deltas[:, :6]).min(axis=0).tolist()
    # Signed chunk extremum (largest absolute delta, preserving sign)
    chunk_max_abs_idx = np.argmax(np.abs(deltas[:, :6]), axis=0)
    chunk_extreme_signed = [
        float(np.rad2deg(deltas[chunk_max_abs_idx[j], j]))
        for j in range(6)
    ]
    return {
        "name": name,
        "state_deg": np.rad2deg(state[:6]).tolist(),
        "h0_abs_deg": np.rad2deg(actions_abs[0, :6]).tolist(),
        "h0_delta_deg": np.rad2deg(deltas[0, :6]).tolist(),
        "h9_delta_deg": np.rad2deg(deltas[h_idx, :6]).tolist(),
        "chunk_delta_min_deg": chunk_min,
        "chunk_delta_max_deg": chunk_max,
        "chunk_extreme_signed_deg": chunk_extreme_signed,
    }


def _print_summary(item: dict[str, Any]) -> None:
    print(f"\n{item['name']}")
    print("  state_deg:            " + _fmt(item["state_deg"]))
    print("  h0_delta_deg:         " + _fmt(item["h0_delta_deg"]))
    print("  h9_delta_deg:         " + _fmt(item["h9_delta_deg"]))
    print("  chunk_min_delta:      " + _fmt(item["chunk_delta_min_deg"]))
    print("  chunk_max_delta:      " + _fmt(item["chunk_delta_max_deg"]))
    print("  chunk_extreme_signed: " + _fmt(item["chunk_extreme_signed_deg"]))


def _fmt(values: list[float]) -> str:
    return "[" + ", ".join(f"{v:7.2f}" for v in values) + "]"


def _print_slot_comparison(cases: list[dict[str, Any]], live_state: np.ndarray) -> None:
    slot_cases = [c for c in cases if c["name"].startswith("SLOT")]
    if not slot_cases:
        return
    print("\n" + "=" * 70)
    print("SLOT MAPPING COMPARISON  (live images + live state)")
    print("joint:           " + "  ".join(f"{n[:9]:>9}" for n in JOINT_NAMES))
    print("-" * 70)
    for c in slot_cases:
        label = c["name"].split("|", 1)[-1].strip()
        vals = c["chunk_extreme_signed_deg"]
        print(f"{label:<18} " + "  ".join(f"{v:9.2f}" for v in vals))
    print("=" * 70)
    print("Interpretation: larger |values| on task-relevant joints (wrist_roll)")
    print("means that slot mapping produces stronger motion commands from live images.")


def main() -> int:
    args = _parse_args()

    # Resolve relative paths now, before _bootstrap_openpi does os.chdir.
    cwd = Path.cwd()
    for attr in ("live_run", "dataset_root", "checkpoint", "output_json"):
        val = getattr(args, attr)
        if val is not None and not val.is_absolute():
            setattr(args, attr, cwd / val)

    _bootstrap_openpi(args)

    train = _load_training_sample(args.dataset_root, args.episode, args.frame)
    live = _load_live_sample(args.live_run) if args.live_run else None

    print(f"training sample: {train['episode']} frame {train['frame']}")
    print(f"training state deg: {_fmt(np.rad2deg(train['state'][:6]).tolist())}")
    if train["action"] is not None:
        demo_delta = train["action"].copy()
        demo_delta[:5] -= train["state"][:5]
        print(f"dataset action delta deg: {_fmt(np.rad2deg(demo_delta[:6]).tolist())}")
    if live is not None:
        print(f"live run: {args.live_run}")
        print(f"live state deg:     {_fmt(np.rad2deg(live['state'][:6]).tolist())}")

    # ------------------------------------------------------------------ #
    # Phase 1: standard A/B/C/D/E/F cases with the default policy        #
    # ------------------------------------------------------------------ #
    print("\nLoading policy (right_wrist_source=wrist) ...")
    policy = _load_policy(args, right_wrist_source="wrist")
    print("Policy loaded.")

    cases: list[dict[str, Any]] = []

    cases.append(_summary(
        "A training images + training state",
        _infer(policy, front=train["front"], wrist=train["wrist"], state=train["state"], prompt=args.prompt),
        train["state"],
    ))

    if live is not None:
        cases.append(_summary(
            "B training images + live state",
            _infer(policy, front=train["front"], wrist=train["wrist"], state=live["state"], prompt=args.prompt),
            live["state"],
        ))
        cases.append(_summary(
            "C live images + training state",
            _infer(policy, front=live["front"], wrist=live["wrist"], state=train["state"], prompt=args.prompt),
            train["state"],
        ))
        cases.append(_summary(
            "D live images + live state",
            _infer(policy, front=live["front"], wrist=live["wrist"], state=live["state"], prompt=args.prompt),
            live["state"],
        ))
        cases.append(_summary(
            "E live front + training wrist + live state",
            _infer(policy, front=live["front"], wrist=train["wrist"], state=live["state"], prompt=args.prompt),
            live["state"],
        ))
        cases.append(_summary(
            "F training front + live wrist + live state",
            _infer(policy, front=train["front"], wrist=live["wrist"], state=live["state"], prompt=args.prompt),
            live["state"],
        ))
        # Slot mapping baseline with current policy (wrist duplicated)
        cases.append(_summary(
            "SLOT | wrist (current: right_wrist=wrist duplicate)",
            _infer(policy, front=live["front"], wrist=live["wrist"], state=live["state"], prompt=args.prompt),
            live["state"],
        ))

    for item in cases:
        _print_summary(item)

    # Free policy before loading next variant (JAX arrays stay alive until GC)
    del policy
    gc.collect()

    # ------------------------------------------------------------------ #
    # Phase 2: slot-mapping ablation (loads policy twice more)            #
    # ------------------------------------------------------------------ #
    if live is not None and not args.skip_slot_variants:
        for source, label in [
            ("front", "SLOT | front (Illia-style: right_wrist=front)"),
            ("zeros", "SLOT | zeros (UR5-style: right_wrist=zeros+mask=False)"),
        ]:
            print(f"\nLoading policy (right_wrist_source={source}) ...")
            p = _load_policy(args, right_wrist_source=source)
            print("Policy loaded.")
            cases.append(_summary(
                label,
                _infer(p, front=live["front"], wrist=live["wrist"], state=live["state"], prompt=args.prompt),
                live["state"],
            ))
            _print_summary(cases[-1])
            del p
            gc.collect()

        _print_slot_comparison(cases, live["state"])

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({"cases": cases}, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output_json}")

    if live is None:
        print("\nNote: pass --live-run <dir> to include live-image comparisons and slot tests.")
    else:
        print("\nInterpretation:")
        print("  A vs B: if similar, state is not the main cause.")
        print("  A vs C: if different, live images / camera preprocessing are the main cause.")
        print("  C vs D: if different, live state contributes on top of images.")
        print("  E:      replaces live wrist with training wrist — isolates wrist contribution.")
        print("  F:      replaces live front with training front — isolates front contribution.")
        print("  SLOT:   compares right-wrist slot mapping variants on live images.")
        print("          The winner is the mapping that produces largest |task-relevant| deltas.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
