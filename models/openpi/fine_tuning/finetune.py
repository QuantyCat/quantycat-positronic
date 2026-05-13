"""
Fine-tune openpi (pi0 / pi0-FAST) on your SO-101 dataset.

Reads all paths and hyperparameters from models/openpi/config.yaml, builds an
openpi TrainConfig for the SO-101 robot, and runs the PyTorch training loop
directly (no subprocess — the training function is imported from openpi).

Prerequisites
-------------
1. source models/openpi/run_scripts/setup.sh
2. Set hf_repo_id in config.yaml to your LeRobot dataset.
3. Run models/openpi/preprocessing/step1_compute_norm_stats.py first.

Run from repo root:
    bash models/openpi/run_scripts/finetune.sh
  or directly:
    python3 models/openpi/fine_tuning/finetune.py
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
    scripts_dir = openpi_repo / "scripts"

    for p in (str(openpi_repo / "src"), str(training_config_dir), str(scripts_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    if not openpi_repo.is_dir():
        print(f"ERROR: openpi repo not found at {openpi_repo}")
        sys.exit(1)

    hf_repo_id = yaml_cfg.get("hf_repo_id") or None
    if hf_repo_id is None:
        print("ERROR: hf_repo_id must be set in config.yaml")
        print("  Set it to your HuggingFace dataset repo id or a local LeRobot dataset path.")
        sys.exit(1)

    from so101_train_config import build_train_config

    train_cfg = build_train_config(yaml_cfg)

    work_dir = pathlib.Path(yaml_cfg["work_dir"]).resolve()
    assets_dir = train_cfg.assets_dirs / (train_cfg.name,)
    if not (assets_dir / "norm_stats.json").exists() and not list(assets_dir.glob("*")):
        print("WARNING: norm stats not found. Run step1_compute_norm_stats.py first.")
        print(f"  Expected at: {assets_dir}")

    print(f"Starting fine-tune")
    print(f"  Config:         {CONFIG_PATH}")
    print(f"  Model:          {yaml_cfg.get('model_type', 'pi0')}")
    print(f"  Dataset:        {hf_repo_id}")
    print(f"  Checkpoint dir: {train_cfg.checkpoint_dir}")
    print(f"  Steps:          {train_cfg.num_train_steps}")
    print(f"  Batch size:     {train_cfg.batch_size}")
    print(f"  LR:             {yaml_cfg.get('lr', 5e-5)}")
    print(f"  wandb:          {train_cfg.wandb_enabled}")
    print()

    # Import and run openpi's training loop directly
    # train_loop is the core function in scripts/train_pytorch.py
    import train_pytorch
    train_pytorch.init_logging()
    train_pytorch.train_loop(train_cfg)


if __name__ == "__main__":
    main()
