#!/usr/bin/env python3
"""
Offline RynnVLA-002 inference test.

Runs the same solver used by the live robot loop, but uses images from disk and a
manually provided robot state. This is useful for validating:
  - checkpoint/model loading
  - front+wrist image packing
  - history handling
  - action-head outputs

Example:

  python3 models/rynnvla-002/inference/offline_inference.py \
    --front-image /home/caroline/Desktop/fine_tuning/screwdriver_so101/04142026_epoch3/inference_logs/latest_obs/front.jpg \
    --wrist-image /home/caroline/Desktop/fine_tuning/screwdriver_so101/04142026_epoch3/inference_logs/latest_obs/wrist.jpg \
    --state-rad="-0.1741,-1.8036,1.6939,1.3195,-2.8792,0.1549" \
    --iterations 2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from PIL import ImageOps
import yaml

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
    ckpt = cfg.get("checkpoint")
    if not ckpt:
        raise ValueError("Pass --checkpoint PATH or set checkpoint in config.yaml")
    return Path(ckpt).expanduser().resolve()


def _ensure_rynnvla_on_path(extra: str | None) -> None:
    if extra:
        p = Path(extra).expanduser().resolve()
        if p.is_dir():
            sys.path.insert(0, str(p))


def _parse_state_rad(raw: str | None) -> np.ndarray:
    if not raw:
        return np.zeros(6, dtype=np.float32)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 6:
        raise ValueError(f"--state-rad must have 6 comma-separated values, got {len(parts)}")
    return np.array([float(x) for x in parts], dtype=np.float32)


def _load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _run_one_inference(solver, front: np.ndarray, wrist: np.ndarray, state_rad: np.ndarray, prompt: str, label: str):
    print(f"\n[offline] case={label}")
    action_chunk = solver.get_action_wrist_action_head_state(
        front_image=front,
        wrist_image=wrist,
        state=state_rad,
        prompt=prompt,
    )
    action_chunk = np.asarray(action_chunk, dtype=np.float32)
    print(f"[offline] action_chunk shape={action_chunk.shape}")
    print(f"[offline] action_chunk={np.round(action_chunk, 4).tolist()}")
    return action_chunk


def _print_delta(name: str, baseline: np.ndarray, candidate: np.ndarray):
    diff = np.asarray(candidate, dtype=np.float32) - np.asarray(baseline, dtype=np.float32)
    l2 = float(np.linalg.norm(diff))
    linf = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    print(
        f"[sensitivity] {name}: "
        f"l2={l2:.6f}  linf={linf:.6f}  mean_abs={mean_abs:.6f}"
    )


def _reset_solver_history(solver) -> None:
    solver.his_img = []
    solver.his_wrist_img = []


def main() -> None:
    root = _repo_root()
    os.chdir(root)

    parser = argparse.ArgumentParser(description="Offline RynnVLA-002 inference test.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--positronic-config", type=str, default=_DEFAULT_CONFIG)
    parser.add_argument(
        "--rynnvla-repo",
        type=str,
        default=os.environ.get("RYNNVLA_REPO", ""),
        help="If set, prepend this directory to sys.path (e.g. ~/RynnVLA-002/rynnvla-002)",
    )
    parser.add_argument("--front-image", type=str, required=True)
    parser.add_argument("--wrist-image", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument(
        "--state-rad",
        type=str,
        default=None,
        help="6 comma-separated joint values in radians. Default: all zeros.",
    )
    parser.add_argument("--iterations", type=int, default=2, help="Number of repeated inference calls.")
    parser.add_argument(
        "--sensitivity-check",
        action="store_true",
        help="Run baseline plus simple state/front/wrist perturbation tests.",
    )
    parser.add_argument(
        "--deterministic-crop",
        action="store_true",
        help="Use deterministic center crop for debugging instead of random inference crop.",
    )
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()

    cfg_path = Path(args.positronic_config)
    if not cfg_path.is_absolute():
        cfg_path = (root / cfg_path).resolve()
    positronic_cfg = _load_positronic_config(cfg_path)

    def _cfg(key, required=False):
        val = positronic_cfg.get(key)
        if required and val is None:
            raise SystemExit(f"ERROR: '{key}' is required in config.yaml")
        return val

    if args.prompt is None:
        args.prompt = _cfg("prompt", required=True)
    if args.gpu is None:
        args.gpu = _cfg("gpu", required=True)

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
        output_dir=str(ckpt_path / "offline_inference_logs"),
        device=args.gpu,
        action_dim=_cfg("action_dim", required=True),
        time_horizon=_cfg("chunk_size", required=True),
        max_seq_len=_cfg("max_seq_len", required=True),
        mask_image_logits=_cfg("mask_image_logits", required=True),
        dropout=_cfg("dropout", required=True),
        z_loss_weight=_cfg("z_loss_weight", required=True),
        his=_cfg("his_mode", required=True),
        action_steps=_cfg("action_steps", required=True),
        deterministic_crop=args.deterministic_crop,
    )

    Path(solver_args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Loading Solver from {ckpt_path} …")
    solver = Solver(solver_args)

    front = _load_image(args.front_image)
    wrist = _load_image(args.wrist_image)
    state_rad = _parse_state_rad(args.state_rad)

    print(f"front_image={args.front_image}")
    print(f"wrist_image={args.wrist_image}")
    print(f"prompt={args.prompt}")
    print(f"state_rad={np.round(state_rad, 4).tolist()}")
    print(f"iterations={args.iterations}")

    if not args.sensitivity_check:
        for i in range(args.iterations):
            _run_one_inference(
                solver=solver,
                front=front,
                wrist=wrist,
                state_rad=state_rad,
                prompt=args.prompt,
                label=f"iteration={i}",
            )
        return

    _reset_solver_history(solver)
    baseline = _run_one_inference(
        solver=solver,
        front=front,
        wrist=wrist,
        state_rad=state_rad,
        prompt=args.prompt,
        label="baseline",
    )

    _reset_solver_history(solver)
    state_shift = np.array([0.15, -0.15, 0.15, -0.15, 0.15, 0.05], dtype=np.float32)
    state_variant = _run_one_inference(
        solver=solver,
        front=front,
        wrist=wrist,
        state_rad=state_rad + state_shift,
        prompt=args.prompt,
        label="state_shift",
    )
    _print_delta("state_shift_vs_baseline", baseline, state_variant)

    _reset_solver_history(solver)
    front_variant_img = np.asarray(ImageOps.mirror(Image.fromarray(front)), dtype=np.uint8)
    front_variant = _run_one_inference(
        solver=solver,
        front=front_variant_img,
        wrist=wrist,
        state_rad=state_rad,
        prompt=args.prompt,
        label="front_mirror",
    )
    _print_delta("front_mirror_vs_baseline", baseline, front_variant)

    _reset_solver_history(solver)
    wrist_variant_img = np.asarray(ImageOps.mirror(Image.fromarray(wrist)), dtype=np.uint8)
    wrist_variant = _run_one_inference(
        solver=solver,
        front=front,
        wrist=wrist_variant_img,
        state_rad=state_rad,
        prompt=args.prompt,
        label="wrist_mirror",
    )
    _print_delta("wrist_mirror_vs_baseline", baseline, wrist_variant)

    _reset_solver_history(solver)
    front_black = np.zeros_like(front, dtype=np.uint8)
    front_black_variant = _run_one_inference(
        solver=solver,
        front=front_black,
        wrist=wrist,
        state_rad=state_rad,
        prompt=args.prompt,
        label="front_black",
    )
    _print_delta("front_black_vs_baseline", baseline, front_black_variant)

    _reset_solver_history(solver)
    wrist_black = np.zeros_like(wrist, dtype=np.uint8)
    wrist_black_variant = _run_one_inference(
        solver=solver,
        front=front,
        wrist=wrist_black,
        state_rad=state_rad,
        prompt=args.prompt,
        label="wrist_black",
    )
    _print_delta("wrist_black_vs_baseline", baseline, wrist_black_variant)

    _reset_solver_history(solver)
    wrist_equals_front_variant = _run_one_inference(
        solver=solver,
        front=front,
        wrist=front.copy(),
        state_rad=state_rad,
        prompt=args.prompt,
        label="wrist_equals_front",
    )
    _print_delta("wrist_equals_front_vs_baseline", baseline, wrist_equals_front_variant)


if __name__ == "__main__":
    main()
