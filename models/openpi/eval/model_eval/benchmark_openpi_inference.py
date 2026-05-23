#!/usr/bin/env python3
"""Benchmark OpenPI policy inference latency for a deployment checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy-config", type=Path, default=_repo_root() / "models/openpi/deployment/pi05_lora_step9999_so101.json")
    parser.add_argument("--live-run", type=Path, default=None, help="Directory with latest_front.npy/latest_wrist.npy/latest_state_model.npy")
    parser.add_argument(
        "--sample-steps",
        type=int,
        nargs="+",
        default=None,
        help="One or more rollout horizons to benchmark. Defaults to model.sample_steps from the deploy config.",
    )
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--timed-runs", type=int, default=20)
    parser.add_argument("--prompt", default=None, help="Override the prompt from the deploy config.")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


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


def _load_live_observation(live_run: Path, prompt: str) -> dict[str, Any]:
    return {
        "observation/images/front": np.load(live_run / "latest_front.npy"),
        "observation/images/wrist": np.load(live_run / "latest_wrist.npy"),
        "observation/state": np.load(live_run / "latest_state_model.npy").astype(np.float32),
        "prompt": prompt,
    }


def _synthetic_observation(cfg: dict[str, Any], prompt: str) -> dict[str, Any]:
    robot_cfg = cfg["robot"]
    height = int(robot_cfg["camera_height"])
    width = int(robot_cfg["camera_width"])
    return {
        "observation/images/front": np.zeros((height, width, 3), dtype=np.uint8),
        "observation/images/wrist": np.zeros((height, width, 3), dtype=np.uint8),
        "observation/state": np.zeros((6,), dtype=np.float32),
        "prompt": prompt,
    }


def _load_policy(openpi_repo: Path, checkpoint: Path, config_name: str, prompt: str, sample_steps: int, pytorch_device: Any):
    src = openpi_repo / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"OpenPI src directory not found: {src}")
    if not (checkpoint / "params").is_dir():
        raise FileNotFoundError(f"Checkpoint params not found: {checkpoint / 'params'}")
    if not (checkpoint / "assets").is_dir():
        raise FileNotFoundError(f"Checkpoint assets not found: {checkpoint / 'assets'}")

    sys.path.insert(0, str(src))
    os.chdir(openpi_repo)

    from openpi.policies import policy_config
    from openpi.training import config as openpi_config

    train_config = openpi_config.get_config(config_name)
    return policy_config.create_trained_policy(
        train_config,
        checkpoint,
        sample_kwargs={"num_steps": sample_steps},
        default_prompt=prompt,
        pytorch_device=pytorch_device,
    )


def _infer(policy: Any, obs: dict[str, Any]) -> np.ndarray:
    actions = np.asarray(policy.infer(obs)["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] < 6:
        raise ValueError(f"Expected policy actions shape (T, >=6), got {actions.shape}")
    return actions


def _device_info() -> dict[str, Any]:
    payload: dict[str, Any] = {}
    try:
        import jax

        payload["jax_default_backend"] = jax.default_backend()
        payload["jax_devices"] = [str(device) for device in jax.devices()]
    except Exception as exc:
        payload["jax_error"] = repr(exc)
    return payload


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("No values provided")
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * q
    lo = int(np.floor(idx))
    hi = int(np.ceil(idx))
    if lo == hi:
        return sorted_values[lo]
    frac = idx - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _benchmark_policy(policy: Any, obs: dict[str, Any], warmup_runs: int, timed_runs: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    first_actions = _infer(policy, obs)
    first_ms = (time.perf_counter() - t0) * 1000.0

    for _ in range(max(0, warmup_runs)):
        _infer(policy, obs)

    timings_ms: list[float] = []
    last_actions = first_actions
    for _ in range(max(1, timed_runs)):
        t1 = time.perf_counter()
        last_actions = _infer(policy, obs)
        timings_ms.append((time.perf_counter() - t1) * 1000.0)

    sorted_ms = sorted(timings_ms)
    mean_ms = float(sum(timings_ms) / len(timings_ms))
    return {
        "first_infer_ms": first_ms,
        "warm_runs": len(timings_ms),
        "steady_mean_ms": mean_ms,
        "steady_min_ms": float(sorted_ms[0]),
        "steady_p50_ms": float(_percentile(sorted_ms, 0.50)),
        "steady_p90_ms": float(_percentile(sorted_ms, 0.90)),
        "steady_p95_ms": float(_percentile(sorted_ms, 0.95)),
        "steady_max_ms": float(sorted_ms[-1]),
        "steady_hz_mean": float(1000.0 / mean_ms),
        "actions_shape": list(last_actions.shape),
        "actions_per_infer": int(last_actions.shape[0]),
        "actions_per_second_mean": float(last_actions.shape[0] * 1000.0 / mean_ms),
    }


def main() -> int:
    args = _parse_args()
    cfg = _load_json(args.deploy_config)
    root = _repo_root()
    model_cfg = cfg["model"]
    openpi_repo = _resolve_path(root, model_cfg["openpi_repo"])
    checkpoint = _resolve_path(root, model_cfg["checkpoint_path"])
    if openpi_repo is None or checkpoint is None:
        raise ValueError("deploy config must define model.openpi_repo and model.checkpoint_path")

    prompt = args.prompt or model_cfg["prompt"]
    obs = (
        _load_live_observation(args.live_run, prompt)
        if args.live_run is not None
        else _synthetic_observation(cfg, prompt)
    )
    sample_steps_list = args.sample_steps or [int(model_cfg.get("sample_steps", 10))]

    payload: dict[str, Any] = {
        "deploy_config": str(args.deploy_config),
        "openpi_repo": str(openpi_repo),
        "checkpoint": str(checkpoint),
        "prompt": prompt,
        "observation_source": str(args.live_run) if args.live_run is not None else "synthetic_zeros",
        "warmup_runs": args.warmup_runs,
        "timed_runs": args.timed_runs,
        "device": _device_info(),
        "results": [],
    }

    print(f"openpi_repo: {openpi_repo}")
    print(f"checkpoint: {checkpoint}")
    print(f"observation_source: {payload['observation_source']}")
    print(f"device: {json.dumps(payload['device'])}")

    for sample_steps in sample_steps_list:
        t0 = time.perf_counter()
        policy = _load_policy(
            openpi_repo=openpi_repo,
            checkpoint=checkpoint,
            config_name=model_cfg["config_name"],
            prompt=prompt,
            sample_steps=int(sample_steps),
            pytorch_device=model_cfg.get("pytorch_device"),
        )
        load_ms = (time.perf_counter() - t0) * 1000.0
        bench = _benchmark_policy(policy, obs, args.warmup_runs, args.timed_runs)
        row = {
            "sample_steps": int(sample_steps),
            "load_ms": load_ms,
            **bench,
        }
        payload["results"].append(row)
        print()
        print(f"sample_steps={row['sample_steps']}")
        print(f"  load_ms:               {row['load_ms']:.2f}")
        print(f"  first_infer_ms:        {row['first_infer_ms']:.2f}")
        print(f"  steady_mean_ms:        {row['steady_mean_ms']:.2f}")
        print(f"  steady_p50_ms:         {row['steady_p50_ms']:.2f}")
        print(f"  steady_p90_ms:         {row['steady_p90_ms']:.2f}")
        print(f"  steady_p95_ms:         {row['steady_p95_ms']:.2f}")
        print(f"  steady_max_ms:         {row['steady_max_ms']:.2f}")
        print(f"  steady_hz_mean:        {row['steady_hz_mean']:.2f}")
        print(f"  actions_per_second:    {row['actions_per_second_mean']:.2f}")
        print(f"  actions_shape:         {tuple(row['actions_shape'])}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print()
        print(f"wrote {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
