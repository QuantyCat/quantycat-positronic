#!/usr/bin/env python3
"""Benchmark inference latency for the Quantycat OpenPI policy.

Loads the policy from inference_config.json and runs N inferences with a
synthetic observation, reporting JIT warmup and steady-state timing.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np


_DEFAULT_CONFIG = "models/openpi/inference/inference_config.json"
_N_WARMUP = 1   # calls before we start recording (JIT compile)
_N_BENCH  = 20  # calls to record for steady-state stats


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve_path(root: Path, value: str) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (root / p).resolve()


def _make_synthetic_obs(cfg: dict) -> dict:
    robot = cfg["robot"]
    h, w = int(robot["camera_height"]), int(robot["camera_width"])
    rng = np.random.default_rng(42)
    return {
        "observation/images/front": rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
        "observation/images/wrist": rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
        "observation/state":        rng.uniform(-1.0, 1.0, 6).astype(np.float32),
        "prompt":                   cfg["model"]["prompt"],
    }


def main() -> int:
    root = _repo_root()
    cfg = json.loads((root / _DEFAULT_CONFIG).read_text())

    openpi_repo = _resolve_path(root, cfg["model"]["openpi_repo"])
    sys.path.insert(0, str(openpi_repo / "src"))
    os.chdir(openpi_repo)

    from openpi.policies import policy_config
    from openpi.training import config as openpi_config

    train_config = openpi_config.get_config(cfg["model"]["config_name"])
    checkpoint   = _resolve_path(root, cfg["model"]["checkpoint_path"])
    sample_steps = int(cfg["model"].get("sample_steps", 10))

    print(f"config_name:   {cfg['model']['config_name']}")
    print(f"checkpoint:    {checkpoint}")
    print(f"sample_steps:  {sample_steps}  (diffusion denoising iters per call)")
    print(f"action_horizon: 20  (steps returned per call)")
    print(f"warmup calls:  {_N_WARMUP}")
    print(f"bench calls:   {_N_BENCH}")
    print()

    print("Loading policy...")
    t0 = time.monotonic()
    policy = policy_config.create_trained_policy(
        train_config,
        checkpoint,
        sample_kwargs={"num_steps": sample_steps},
        default_prompt=cfg["model"]["prompt"],
    )
    print(f"Policy loaded in {(time.monotonic() - t0)*1000:.0f} ms\n")

    obs = _make_synthetic_obs(cfg)

    wall_times_ms  = []
    model_times_ms = []

    for i in range(_N_WARMUP + _N_BENCH):
        t_wall = time.monotonic()
        result = policy.infer(obs)
        wall_ms  = (time.monotonic() - t_wall) * 1000
        model_ms = result["policy_timing"]["infer_ms"]

        if i < _N_WARMUP:
            label = f"[warmup {i+1}/{_N_WARMUP}]"
        else:
            wall_times_ms.append(wall_ms)
            model_times_ms.append(model_ms)
            label = f"[bench  {i - _N_WARMUP + 1:2d}/{_N_BENCH}]"

        actions = result["actions"]
        print(
            f"{label}  wall={wall_ms:7.1f} ms  model={model_ms:7.1f} ms  "
            f"actions shape={actions.shape}"
        )

    print()
    print("=== Steady-state results ===")
    for label, times in [("wall (total)", wall_times_ms), ("model only  ", model_times_ms)]:
        arr = np.array(times)
        print(
            f"  {label}:  "
            f"mean={arr.mean():.1f}  "
            f"median={np.median(arr):.1f}  "
            f"p95={np.percentile(arr, 95):.1f}  "
            f"min={arr.min():.1f}  "
            f"max={arr.max():.1f}  ms"
        )

    chunk_budget_ms = 20 * cfg["control"]["control_period_s"] * 1000
    print()
    print(f"  Chunk execution budget:  {chunk_budget_ms:.0f} ms  "
          f"(execute_steps={20} × control_period={cfg['control']['control_period_s']*1000:.0f} ms)")
    headroom = chunk_budget_ms - np.median(wall_times_ms)
    print(f"  Headroom vs median wall: {headroom:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
