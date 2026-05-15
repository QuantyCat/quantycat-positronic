#!/usr/bin/env python3
"""
Evaluate one training episode timestep against its saved ground-truth action chunk.

This script is designed for direct debugging of one observation from the training
data. It:
  - loads front/wrist image history and current state from one episode directory
  - runs the same solver used by inference
  - compares the predicted action chunk to the saved ground-truth chunk

Example:

  conda run -n rynnvla002 python models/rynnvla-002/eval/model_eval/episode_step_eval.py \
    --episode my_data/training_pipeline/training_data/Put_the_screwdriver_into_the_cup/episode_000025 \
    --step 120 \
    --checkpoint /home/caroline/old_checkpoints/04142026_epoch3 \
    --rynnvla-repo /home/caroline/RynnVLA-002/rynnvla-002
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import yaml

_DEFAULT_CONFIG = "models/rynnvla-002/config.yaml"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_positronic_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def _resolve_checkpoint(args: argparse.Namespace, cfg: dict[str, Any]) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint).expanduser().resolve()
    ckpt = cfg.get("checkpoint")
    if not ckpt:
        raise ValueError("Pass --checkpoint PATH or set checkpoint in config.yaml")
    return Path(ckpt).expanduser().resolve()


def _ensure_rynnvla_on_path(extra: str | None) -> None:
    if extra:
        p = Path(extra).expanduser().resolve()
        if p.is_dir():
            sys.path.insert(0, str(p))


def _numeric_suffix(path: Path, prefix: str) -> int:
    name = path.stem if path.is_file() else path.name
    if not name.startswith(prefix):
        raise ValueError(f"unexpected name {name!r}, expected prefix {prefix!r}")
    return int(name[len(prefix):])


def _sorted_frame_paths(dir_path: Path, prefix: str, suffix: str) -> list[Path]:
    paths = [p for p in dir_path.iterdir() if p.is_file() and p.name.startswith(prefix) and p.name.endswith(suffix)]
    return sorted(paths, key=lambda p: _numeric_suffix(p, prefix))


def _sorted_action_dirs(dir_path: Path) -> list[Path]:
    paths = [p for p in dir_path.iterdir() if p.is_dir() and p.name.startswith("action_")]
    return sorted(paths, key=lambda p: _numeric_suffix(p, "action_"))


def _load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_state(path: Path) -> np.ndarray:
    return np.asarray(np.load(path), dtype=np.float32)


def _load_action_chunk(action_dir: Path) -> np.ndarray:
    """Load saved training targets.

    Despite the directory name `abs_action`, preprocessing has already converted
    joints 0-4 to relative deltas. Do not subtract the current state here.
    The gripper channel remains an absolute target.
    """
    files = sorted(
        [p for p in action_dir.iterdir() if p.is_file() and p.suffix == ".npy"],
        key=lambda p: int(p.stem),
    )
    if not files:
        raise ValueError(f"no action files found in {action_dir}")
    return np.asarray([np.load(p).astype(np.float32) for p in files], dtype=np.float32)


def _extract_task_from_episode(episode_dir: Path) -> str:
    return episode_dir.parent.name.replace("_", " ")


def _build_sample(episode_dir: Path, step: int, his: int) -> dict[str, Any]:
    front_dir = episode_dir / "front_image"
    wrist_dir = episode_dir / "wrist_image"
    state_dir = episode_dir / "state"
    action_root = episode_dir / "abs_action"

    for p in [front_dir, wrist_dir, state_dir, action_root]:
        if not p.is_dir():
            raise FileNotFoundError(f"missing directory: {p}")

    front_paths = _sorted_frame_paths(front_dir, "image_", ".png")
    wrist_paths = _sorted_frame_paths(wrist_dir, "image_", ".png")
    state_paths = _sorted_frame_paths(state_dir, "state_", ".npy")
    action_dirs = _sorted_action_dirs(action_root)

    if step < 0:
        raise ValueError(f"--step must be >= 0, got {step}")
    if step >= len(front_paths) or step >= len(wrist_paths) or step >= len(state_paths):
        raise ValueError(
            f"step {step} out of range for episode {episode_dir.name}: "
            f"front={len(front_paths)} wrist={len(wrist_paths)} state={len(state_paths)}"
        )
    if step >= len(action_dirs):
        raise ValueError(
            f"step {step} has no saved action chunk in {action_root}. "
            f"Available action chunks: 0..{len(action_dirs) - 1}"
        )

    hist_start = max(0, step - his + 1)
    front_hist_paths = list(front_paths[hist_start : step + 1])
    wrist_hist_paths = list(wrist_paths[hist_start : step + 1])

    while len(front_hist_paths) < his:
        front_hist_paths.insert(0, front_hist_paths[0])
        wrist_hist_paths.insert(0, wrist_hist_paths[0])

    front_imgs = [_load_image(p) for p in front_hist_paths]
    wrist_imgs = [_load_image(p) for p in wrist_hist_paths]
    state = _load_state(state_paths[step])
    gt_action = _load_action_chunk(action_dirs[step])

    return {
        "episode_dir": episode_dir,
        "step": step,
        "task": _extract_task_from_episode(episode_dir),
        "front_history_paths": front_hist_paths[:-1],
        "wrist_history_paths": wrist_hist_paths[:-1],
        "front_current_path": front_hist_paths[-1],
        "wrist_current_path": wrist_hist_paths[-1],
        "state_path": state_paths[step],
        "action_dir": action_dirs[step],
        "front_history": front_imgs[:-1],
        "wrist_history": wrist_imgs[:-1],
        "front_current": front_imgs[-1],
        "wrist_current": wrist_imgs[-1],
        "state": state,
        "gt_action": gt_action,
    }


def _metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    diff = np.asarray(pred, dtype=np.float32) - np.asarray(gt, dtype=np.float32)
    return {
        "mean_abs": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "l2": float(np.linalg.norm(diff)),
        "first_step_mean_abs": float(np.mean(np.abs(diff[0]))),
        "gripper_mean_abs": float(np.mean(np.abs(diff[:, -1]))),
        "joint_mean_abs": float(np.mean(np.abs(diff[:, :-1]))),
    }


def _reset_solver_history(solver, sample: dict[str, Any]) -> None:
    solver.his_img = list(sample["front_history"])
    solver.his_wrist_img = list(sample["wrist_history"])


def main() -> None:
    root = _repo_root()
    os.chdir(root)

    parser = argparse.ArgumentParser(description="Evaluate one episode timestep against saved ground-truth actions.")
    parser.add_argument("--episode", type=str, required=True, help="Path to one episode directory.")
    parser.add_argument("--step", type=int, required=True, help="Timestep index within the episode.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--positronic-config", type=str, default=_DEFAULT_CONFIG)
    parser.add_argument("--prompt", type=str, default=None, help="Override task prompt; default uses episode task name.")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--deterministic-crop", action="store_true")
    parser.add_argument(
        "--rynnvla-repo",
        type=str,
        default=os.environ.get("RYNNVLA_REPO", ""),
        help="If set, prepend this directory to sys.path (e.g. ~/RynnVLA-002/rynnvla-002)",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Write a JSON report under <checkpoint>/episode_step_eval_logs/",
    )
    args = parser.parse_args()

    cfg_path = Path(args.positronic_config)
    if not cfg_path.is_absolute():
        cfg_path = (root / cfg_path).resolve()
    cfg = _load_positronic_config(cfg_path)

    work_dir = Path(cfg["work_dir"])
    if not work_dir.is_absolute():
        work_dir = (root / work_dir).resolve()
    os.environ["RYNNVLA_ACTION_STATS_FILE"] = str((work_dir / "min_max_action.txt").resolve())
    os.environ["RYNNVLA_STATE_STATS_FILE"] = str((work_dir / "min_max_state.txt").resolve())
    if cfg.get("action_norm_joint_scales"):
        os.environ["RYNNVLA_ACTION_NORM_SCALES"] = str(cfg["action_norm_joint_scales"])

    ckpt_path = _resolve_checkpoint(args, cfg)
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
        output_dir=str((ckpt_path / "episode_step_eval_logs").resolve()),
        device=args.gpu if args.gpu is not None else cfg["gpu"],
        action_dim=cfg["action_dim"],
        time_horizon=cfg["chunk_size"],
        max_seq_len=cfg["max_seq_len"],
        mask_image_logits=cfg["mask_image_logits"],
        dropout=cfg["dropout"],
        z_loss_weight=cfg["inference_z_loss_weight"],
        his=cfg["his_mode"],
        action_steps=cfg["action_steps"],
        deterministic_crop=bool(args.deterministic_crop or cfg.get("deterministic_crop", False)),
    )

    episode_dir = Path(args.episode).expanduser().resolve()
    his = int(cfg["his"])
    sample = _build_sample(episode_dir, step=args.step, his=his)
    prompt = args.prompt if args.prompt is not None else sample["task"]

    Path(solver_args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Loading Solver from {ckpt_path} ...")
    solver = Solver(solver_args)

    _reset_solver_history(solver, sample)
    pred = solver.get_action_wrist_action_head_state(
        front_image=sample["front_current"],
        wrist_image=sample["wrist_current"],
        state=sample["state"],
        prompt=prompt,
    )
    pred = np.asarray(pred, dtype=np.float32)
    gt = sample["gt_action"]

    if pred.shape != gt.shape:
        raise ValueError(f"pred shape {pred.shape} != gt shape {gt.shape}")

    metrics = _metrics(pred, gt)
    diff = pred - gt

    print(f"episode={episode_dir}")
    print(f"step={args.step}")
    print(f"prompt={prompt}")
    print(f"front_current={sample['front_current_path']}")
    print(f"wrist_current={sample['wrist_current_path']}")
    print(f"state_path={sample['state_path']}")
    print(f"action_dir={sample['action_dir']}")
    print("action_convention=saved target deltas for joints 0-4; gripper absolute; no eval-time state subtraction")
    print(f"front_history={[str(p) for p in sample['front_history_paths']]}")
    print(f"wrist_history={[str(p) for p in sample['wrist_history_paths']]}")
    print(f"state={np.round(sample['state'], 6).tolist()}")

    print("\nMetrics")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")

    print("\nFirst step")
    print(f"  gt:   {np.round(gt[0], 6).tolist()}")
    print(f"  pred: {np.round(pred[0], 6).tolist()}")
    print(f"  diff: {np.round(diff[0], 6).tolist()}")

    print("\nFull chunk")
    print(f"  gt:   {np.round(gt, 6).tolist()}")
    print(f"  pred: {np.round(pred, 6).tolist()}")
    print(f"  diff: {np.round(diff, 6).tolist()}")

    if args.save_json:
        out_path = Path(solver_args.output_dir) / f"{episode_dir.name}_step_{args.step:06d}.json"
        payload = {
            "episode": str(episode_dir),
            "step": args.step,
            "prompt": prompt,
            "front_current": str(sample["front_current_path"]),
            "wrist_current": str(sample["wrist_current_path"]),
            "front_history": [str(p) for p in sample["front_history_paths"]],
            "wrist_history": [str(p) for p in sample["wrist_history_paths"]],
            "state_path": str(sample["state_path"]),
            "action_dir": str(sample["action_dir"]),
            "action_convention": "saved target deltas for joints 0-4; gripper absolute; no eval-time state subtraction",
            "state": np.round(sample["state"], 6).tolist(),
            "metrics": metrics,
            "gt_action": np.round(gt, 6).tolist(),
            "pred_action": np.round(pred, 6).tolist(),
            "diff_action": np.round(diff, 6).tolist(),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved JSON report to {out_path}")


if __name__ == "__main__":
    main()
