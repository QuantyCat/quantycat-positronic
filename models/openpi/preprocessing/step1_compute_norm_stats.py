"""
Stage 1 — Compute openpi normalisation statistics for the SO-101 dataset.

openpi normalises observations and actions before passing them to the model.
This script computes the per-key mean/std (or quantiles) from the training
dataset and writes them to:

    $WORK_DIR/openpi_assets/<training_run_name>/

Those stats are loaded automatically during training (from assets_base_dir in
config.yaml) and are bundled into each checkpoint for inference.

Prerequisites
-------------
- The LeRobot dataset must be accessible at the path/repo id given by
  hf_repo_id in config.yaml.
- openpi must be installed (source models/openpi/run_scripts/setup.sh).
- The so101_policy.py file must be importable (setup.sh handles this).

Run from repo root:
    python3 models/openpi/preprocessing/step1_compute_norm_stats.py
"""

from __future__ import annotations

import pathlib
import sys

import yaml

CONFIG_PATH = "models/openpi/config.yaml"


def main():
    with open(CONFIG_PATH) as f:
        yaml_cfg = yaml.safe_load(f)

    repo_root = pathlib.Path(__file__).resolve().parents[3]
    openpi_repo = pathlib.Path.home() / "openpi"
    training_config_dir = repo_root / "models" / "openpi" / "training_config"

    for p in (str(openpi_repo / "src"), str(training_config_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    if not openpi_repo.is_dir():
        print(f"ERROR: openpi repo not found at {openpi_repo}")
        sys.exit(1)

    hf_repo_id = yaml_cfg.get("hf_repo_id") or None
    if hf_repo_id is None:
        print("ERROR: hf_repo_id must be set in config.yaml before computing norm stats.")
        print("  Set it to your HuggingFace repo id (e.g. 'username/my_so101_dataset')")
        print("  or to an absolute local LeRobot dataset path.")
        sys.exit(1)

    from so101_train_config import build_train_config
    import openpi.shared.normalize as _normalize
    import openpi.training.config as _config
    import openpi.training.data_loader as _data_loader
    import openpi.transforms as _transforms
    import numpy as np
    import tqdm

    train_cfg = build_train_config(yaml_cfg)
    data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)

    assets_dir = train_cfg.assets_dirs / (data_cfg.asset_id or train_cfg.name)
    assets_dir.mkdir(parents=True, exist_ok=True)

    print(f"Computing norm stats for dataset: {hf_repo_id}")
    print(f"Output: {assets_dir}")
    print()

    class _RemoveStrings(_transforms.DataTransformFn):
        def __call__(self, x: dict) -> dict:
            return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}

    dataset = _data_loader.create_torch_dataset(data_cfg, train_cfg.model.action_horizon, train_cfg.model)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_cfg.repack_transforms.inputs,
            *data_cfg.data_transforms.inputs,
            _RemoveStrings(),
        ],
    )

    batch_size = 16
    num_batches = len(dataset) // batch_size
    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=int(yaml_cfg.get("num_workers", 2)),
        shuffle=False,
        num_batches=num_batches,
    )

    use_quantile = data_cfg.use_quantile_norm
    norm_stats = _normalize.RunningStats(use_quantiles=use_quantile)

    for batch in tqdm.tqdm(loader, desc="Computing norm stats", total=num_batches):
        norm_stats.update(batch)

    stats = norm_stats.finish()
    _normalize.save(assets_dir, stats)
    print(f"\nNorm stats saved to {assets_dir}")
    print("Keys:", list(stats.keys()))


if __name__ == "__main__":
    main()
