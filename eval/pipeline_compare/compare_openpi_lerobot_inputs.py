#!/usr/bin/env python3
"""Compare Quantycat SO-101 training inputs for OpenPI and LeRobot pi05.

This is a read-only inspection tool. It does not train or write checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _array(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _summary(x: Any) -> dict[str, Any]:
    a = _array(x)
    out: dict[str, Any] = {"shape": list(a.shape), "dtype": str(a.dtype)}
    if np.issubdtype(a.dtype, np.number) or a.dtype == np.bool_:
        out.update(
            min=float(np.min(a)),
            max=float(np.max(a)),
            mean=float(np.mean(a)),
            std=float(np.std(a)),
        )
    return out


def _values(x: Any, decimals: int | None = None) -> list[Any]:
    a = _array(x)
    if decimals is not None and np.issubdtype(a.dtype, np.floating):
        a = np.round(a.astype(np.float64), decimals)
    return a.tolist()


def _diff_summary(a: Any, b: Any) -> dict[str, Any]:
    aa = _array(a).astype(np.float32)
    bb = _array(b).astype(np.float32)
    diff = aa - bb
    return {
        "shape_a": list(aa.shape),
        "shape_b": list(bb.shape),
        "same_shape": list(aa.shape) == list(bb.shape),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "allclose_atol_1e_6": bool(np.allclose(aa, bb, atol=1e-6, rtol=0.0)),
    }


def _read_v3_sample(dataset_root: Path, index: int) -> dict[str, Any]:
    files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No v3 parquet files under {dataset_root / 'data'}")
    df = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    row = df.iloc[index]
    return {
        "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
        "action": np.asarray(row["action"], dtype=np.float32),
        "episode_index": int(row["episode_index"]),
        "frame_index": int(row["frame_index"]),
        "index": int(row["index"]),
        "task": "Put the screwdriver into the cup",
    }


def _openpi_sample(config_name: str, index: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import openpi.training.config as openpi_config
    import openpi.training.data_loader as data_loader

    cfg = openpi_config.get_config(config_name)
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    raw_dataset = data_loader.create_torch_dataset(data_cfg, cfg.model.action_horizon, cfg.model)
    raw = raw_dataset[index]

    repacked = data_loader.TransformedDataset(raw_dataset, data_cfg.repack_transforms.inputs)[index]
    data_tx = data_loader.TransformedDataset(raw_dataset, [*data_cfg.repack_transforms.inputs, *data_cfg.data_transforms.inputs])[index]
    model_tx = data_loader.TransformedDataset(
        raw_dataset,
        [*data_cfg.repack_transforms.inputs, *data_cfg.data_transforms.inputs, *data_cfg.model_transforms.inputs],
    )[index]

    arrays = {
        "state": _array(data_tx["state"]).astype(np.float32),
        "actions": _array(data_tx["actions"]).astype(np.float32),
        "front_hwc_uint8": _array(data_tx["image"]["base_0_rgb"]),
        "wrist_hwc_uint8": _array(data_tx["image"]["left_wrist_0_rgb"]),
        "right_hwc_uint8": _array(data_tx["image"]["right_wrist_0_rgb"]),
    }

    report = {
        "config_name": config_name,
        "repo_id": data_cfg.repo_id,
        "action_horizon": cfg.model.action_horizon,
        "raw_keys": sorted(raw.keys()),
        "raw": {
            "observation.state": _summary(raw["observation.state"]),
            "action": _summary(raw["action"]),
        },
        "after_quantycat_inputs": {
            "state": _summary(data_tx["state"]),
            "actions": _summary(data_tx["actions"]),
            "first_action": _values(data_tx["actions"][0], decimals=6),
            "last_action": _values(data_tx["actions"][-1], decimals=6),
            "image_keys": sorted(data_tx["image"].keys()),
            "image_shapes": {key: _summary(value) for key, value in data_tx["image"].items()},
            "image_mask": {key: bool(value) for key, value in data_tx["image_mask"].items()},
        },
        "after_model_transforms_no_norm": {
            "state": _summary(model_tx["state"]),
            "actions": _summary(model_tx["actions"]),
            "image_keys": sorted(model_tx["image"].keys()),
            "image_shapes": {key: _summary(value) for key, value in model_tx["image"].items()},
        },
    }
    return report, arrays


def _lerobot_sample(dataset_root: Path, repo_id: str, index: int, horizon: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id)
    ds = LeRobotDataset(
        repo_id,
        delta_timestamps={"action": [t / meta.fps for t in range(horizon)]},
    )
    raw = ds[index]
    parquet_raw = _read_v3_sample(dataset_root, index) if dataset_root.name.endswith("_v3") else None

    front_chw = _array(raw["observation.images.front"]).astype(np.float32)
    wrist_chw = _array(raw["observation.images.wrist"]).astype(np.float32)
    arrays = {
        "state": _array(raw["observation.state"]).astype(np.float32),
        "actions": _array(raw["action"]).astype(np.float32),
        "front_hwc_uint8": np.rint(np.transpose(front_chw, (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8),
        "wrist_hwc_uint8": np.rint(np.transpose(wrist_chw, (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8),
    }

    report = {
        "repo_id": repo_id,
        "dataset_root": str(dataset_root),
        "action_horizon": horizon,
        "raw_keys": sorted(raw.keys()),
        "sample": {
            "observation.state": _summary(raw["observation.state"]),
            "action": _summary(raw["action"]),
            "first_action": _values(raw["action"][0], decimals=6),
            "last_action": _values(raw["action"][-1], decimals=6),
            "observation.images.front": _summary(raw["observation.images.front"]),
            "observation.images.wrist": _summary(raw["observation.images.wrist"]),
        },
        "v3_parquet_sample_without_video_decode": {
            key: _summary(value) if isinstance(value, np.ndarray) else value
            for key, value in (parquet_raw or {}).items()
        },
    }
    return report, arrays


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, default=Path.home() / "quantycat-data")
    parser.add_argument("--openpi-config", default="pi05_quantycat_lora")
    parser.add_argument("--lerobot-repo-id", default="screwdriver_so101_clean_v2")
    parser.add_argument("--lerobot-root", type=Path, default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    os.environ.setdefault("HF_LEROBOT_HOME", str(args.data_home / "datasets"))
    lerobot_root = args.lerobot_root or args.data_home / "datasets" / args.lerobot_repo_id

    openpi_report, openpi_arrays = _openpi_sample(args.openpi_config, args.index)
    lerobot_report, lerobot_arrays = _lerobot_sample(
        lerobot_root, args.lerobot_repo_id, args.index, openpi_report["action_horizon"]
    )

    report = {
        "index": args.index,
        "openpi": openpi_report,
        "lerobot": lerobot_report,
        "comparisons": {
            "state": _diff_summary(openpi_arrays["state"], lerobot_arrays["state"]),
            "action_chunk": _diff_summary(openpi_arrays["actions"], lerobot_arrays["actions"]),
            "front_image_before_openpi_resize": _diff_summary(
                openpi_arrays["front_hwc_uint8"], lerobot_arrays["front_hwc_uint8"]
            ),
            "wrist_image_before_openpi_resize": _diff_summary(
                openpi_arrays["wrist_hwc_uint8"], lerobot_arrays["wrist_hwc_uint8"]
            ),
            "openpi_right_slot_equals_wrist": _diff_summary(
                openpi_arrays["right_hwc_uint8"], openpi_arrays["wrist_hwc_uint8"]
            ),
        },
    }

    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
