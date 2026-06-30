#!/usr/bin/env python3
"""Run OpenPI training with Quantycat Dacha sampling and loss weighting.

This launcher patches OpenPI at runtime so the changes stay outside the vendored
OpenPI checkout. It is intentionally narrow: it only activates for Dacha LeRobot
datasets and pi0/pi0.5 JAX training.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch


DEFAULT_DATASET_ROOT = Path.home() / "quantycat-data" / "datasets" / "dacha" / "dacha_v3_openpi_v21"


class WeightedIndexDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Any, indices: list[int]) -> None:
        self._dataset = dataset
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> Any:
        return self._dataset[self._indices[index]]


def _float_list(name: str, default: str) -> list[float]:
    value = os.environ.get(name, default)
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _int_list(name: str, default: str) -> list[int]:
    value = os.environ.get(name, default)
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _episode_index(path: Path) -> int:
    match = re.search(r"episode_(\d+)\.parquet$", path.name)
    if match is None:
        raise ValueError(f"Could not parse episode index from {path}")
    return int(match.group(1))


def _repeat_count(
    score_j3: float,
    score_j4: float,
    velocity_j3: float,
    velocity_j4: float,
    *,
    max_extra: int,
) -> int:
    extra = 0
    for threshold in _float_list("DACHA_WEIGHTED_J3_DELTA_THRESHOLDS", "2,4,6"):
        extra += int(score_j3 >= threshold)
    for threshold in _float_list("DACHA_WEIGHTED_J4_DELTA_THRESHOLDS", "1,2,3"):
        extra += int(score_j4 >= threshold)
    for threshold in _float_list("DACHA_WEIGHTED_J3_VEL_THRESHOLDS", "2,4"):
        extra += int(velocity_j3 >= threshold)
    for threshold in _float_list("DACHA_WEIGHTED_J4_VEL_THRESHOLDS", "1,2"):
        extra += int(velocity_j4 >= threshold)
    return 1 + min(extra, max_extra)


def _build_weighted_indices(dataset_len: int, train_episodes: list[int] | None, action_horizon: int) -> list[int] | None:
    dataset_root = Path(os.environ.get("DACHA_WEIGHTED_DATASET_ROOT", str(DEFAULT_DATASET_ROOT)))
    data_root = dataset_root / "data" / "chunk-000"
    if not data_root.is_dir():
        logging.warning("Dacha weighted sampling disabled; data root not found: %s", data_root)
        return None

    allowed = set(train_episodes) if train_episodes is not None else None
    max_extra = int(os.environ.get("DACHA_WEIGHTED_MAX_EXTRA_REPEATS", "6"))
    weighted: list[int] = []
    raw_index = 0
    episode_count = 0

    for path in sorted(data_root.glob("episode_*.parquet"), key=_episode_index):
        episode = _episode_index(path)
        if allowed is not None and episode not in allowed:
            continue

        table = pq.read_table(path, columns=["action", "observation.state"])
        action = np.asarray(table.column("action").to_pylist(), dtype=np.float64)
        state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float64)
        delta = action - state
        velocity = np.abs(np.diff(action, axis=0, prepend=action[:1]))

        for frame in range(len(action)):
            end = min(len(action), frame + max(1, action_horizon))
            score_j3 = float(np.max(np.abs(delta[frame:end, 3])))
            score_j4 = float(np.max(np.abs(delta[frame:end, 4])))
            velocity_j3 = float(np.max(velocity[frame:end, 3]))
            velocity_j4 = float(np.max(velocity[frame:end, 4]))
            repeat = _repeat_count(score_j3, score_j4, velocity_j3, velocity_j4, max_extra=max_extra)
            weighted.extend([raw_index] * repeat)
            raw_index += 1

        episode_count += 1

    if raw_index != dataset_len:
        logging.warning(
            "Dacha weighted sampling disabled; computed %s source frames but dataset has %s",
            raw_index,
            dataset_len,
        )
        return None

    logging.info(
        "Dacha weighted sampling enabled: episodes=%s source_frames=%s weighted_frames=%s multiplier=%.2f",
        episode_count,
        raw_index,
        len(weighted),
        len(weighted) / max(1, raw_index),
    )
    return weighted


def _patch_weighted_dataset() -> None:
    import openpi.training.data_loader as data_loader

    original = data_loader.create_torch_dataset

    def patched_create_torch_dataset(data_config, action_horizon, model_config):
        dataset = original(data_config, action_horizon, model_config)
        enabled = os.environ.get("DACHA_WEIGHTED_SAMPLING", "1").lower() not in ("0", "false", "no")
        weighted_repo_ids = {
            item.strip()
            for item in os.environ.get(
                "DACHA_WEIGHTED_REPO_IDS",
                "dacha/dacha_v3_openpi_v21,"
                "dacha/dacha_v5_openpi_v21,"
                "dacha/dacha_v5_train_openpi_v21,"
                "dacha/dacha_v5_ep16_99_train_openpi_v21",
            ).split(",")
            if item.strip()
        }
        if not enabled or data_config.repo_id not in weighted_repo_ids:
            return dataset
        indices = _build_weighted_indices(len(dataset), data_config.train_episodes, action_horizon)
        return WeightedIndexDataset(dataset, indices) if indices is not None else dataset

    data_loader.create_torch_dataset = patched_create_torch_dataset


def _patch_weighted_pi0_loss() -> None:
    import jax
    import jax.numpy as jnp

    from openpi.models import model as _model
    import openpi.models.pi0 as pi0

    base_weights = _float_list("DACHA_JOINT_LOSS_WEIGHTS", "1,1,1,2,3,1")
    early_end = int(os.environ.get("DACHA_EARLY_HORIZON_END", "-1"))
    early_horizon_weight = float(os.environ.get("DACHA_EARLY_HORIZON_WEIGHT", "1"))
    early_joints = _int_list("DACHA_EARLY_JOINTS", "3,4")
    early_joint_multipliers = _float_list("DACHA_EARLY_JOINT_MULTIPLIERS", "1,1")
    if len(early_joint_multipliers) < len(early_joints):
        early_joint_multipliers.extend([1.0] * (len(early_joints) - len(early_joint_multipliers)))

    def weighted_compute_loss(self, rng, observation, actions, *, train: bool = False):
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        action_dim = actions.shape[-1]
        weights = list(base_weights[:action_dim])
        if len(weights) < action_dim:
            weights.extend([1.0] * (action_dim - len(weights)))
        weights_arr = jnp.broadcast_to(jnp.asarray(weights, dtype=jnp.float32), (self.action_horizon, action_dim))
        if early_end >= 0:
            early_count = min(self.action_horizon, early_end + 1)
            horizon_scale = jnp.where(
                jnp.arange(self.action_horizon)[:, None] < early_count,
                early_horizon_weight,
                1.0,
            )
            weights_arr = weights_arr * horizon_scale
            for joint, multiplier in zip(early_joints, early_joint_multipliers, strict=False):
                if 0 <= joint < action_dim:
                    early_mask = jnp.arange(self.action_horizon) < early_count
                    dim_mask = jnp.arange(action_dim) == joint
                    weights_arr = weights_arr * jnp.where(early_mask[:, None] & dim_mask[None, :], multiplier, 1.0)
        per_dim = jnp.square(v_t - u_t)
        normalization = jnp.mean(jnp.sum(weights_arr, axis=-1))
        return jnp.sum(per_dim * weights_arr, axis=-1) / normalization

    pi0.Pi0.compute_loss = weighted_compute_loss
    logging.info("Dacha joint loss weights enabled: %s", base_weights)
    logging.info(
        "Dacha early horizon loss weights enabled: end=%s horizon_weight=%s early_joints=%s early_multipliers=%s",
        early_end,
        early_horizon_weight,
        early_joints,
        early_joint_multipliers,
    )


def _load_train_module(openpi_repo: Path):
    train_path = openpi_repo / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("quantycat_openpi_train", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load OpenPI training script from {train_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    openpi_repo = Path(os.environ.get("OPENPI_REPO", "/home/caroline/quantycat-positronic/vendor/openpi"))
    repo = Path(os.environ.get("QUANTYCAT_POSITRONIC_REPO", "/home/caroline/quantycat-positronic"))
    config_dir = repo / "models" / "openpi" / "vendor_patches" / "src"
    sys.path.insert(0, str(config_dir))
    sys.path.insert(1, str(openpi_repo / "src"))

    _patch_weighted_dataset()
    _patch_weighted_pi0_loss()

    train_module = _load_train_module(openpi_repo)
    train_module.main(train_module._config.cli())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
