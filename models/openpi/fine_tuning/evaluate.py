"""
Evaluate an openpi SO-101 checkpoint on a held-out validation split.

Runs the model in inference mode over every item in the validation set,
collects per-step action MSE and per-joint errors, and writes a JSON
summary to the checkpoint directory.

Run from repo root:
    python3 models/openpi/fine_tuning/evaluate.py \\
        --checkpoint my_data/training_pipeline/fine_tuning/<run>/29999
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import yaml

CONFIG_PATH = "models/openpi/config.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint directory. Defaults to config.yaml checkpoint.")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        yaml_cfg = yaml.safe_load(f)

    repo_root = pathlib.Path(__file__).resolve().parents[3]
    openpi_repo = pathlib.Path.home() / "openpi"
    training_config_dir = repo_root / "models" / "openpi" / "training_config"

    for p in (str(openpi_repo / "src"), str(training_config_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    ckpt_path = pathlib.Path(args.checkpoint or yaml_cfg.get("checkpoint") or "")
    if not ckpt_path.is_dir():
        print(f"ERROR: checkpoint directory not found: {ckpt_path}")
        sys.exit(1)

    from so101_train_config import build_train_config
    import openpi.training.data_loader as _data_loader
    import openpi.transforms as _transforms
    import safetensors.torch
    import torch

    train_cfg = build_train_config(yaml_cfg)
    model_cfg = train_cfg.model
    data_cfg = train_cfg.data.create(train_cfg.assets_dirs, model_cfg)

    # Load model
    import openpi.models_pytorch.pi0_pytorch as _pi0
    device = torch.device(f"cuda:{yaml_cfg.get('gpu', 0)}" if torch.cuda.is_available() else "cpu")
    model = _pi0.PI0Pytorch(model_cfg).to(device)
    safetensors.torch.load_model(model, ckpt_path / "model.safetensors", device=str(device))
    model.eval()

    # Load norm stats from checkpoint
    import openpi.shared.normalize as _normalize
    asset_id = data_cfg.asset_id or train_cfg.name
    norm_stats = _normalize.load(ckpt_path / "assets" / asset_id)

    # Build data loader (no shuffle, single pass)
    dataset = _data_loader.create_torch_dataset(data_cfg, model_cfg.action_horizon, model_cfg)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_cfg.repack_transforms.inputs,
            *data_cfg.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_cfg.use_quantile_norm),
            *data_cfg.model_transforms.inputs,
        ],
    )
    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=1,
        num_workers=0,
        shuffle=False,
        num_batches=len(dataset),
    )

    action_dim = int(yaml_cfg["action_dim"])
    action_horizon = int(yaml_cfg["action_horizon"])
    mse_per_step = np.zeros(action_horizon, dtype=np.float64)
    mse_per_joint = np.zeros(action_dim, dtype=np.float64)
    count = 0

    with torch.inference_mode():
        for observation, gt_actions in loader:
            observation = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in observation.items()}
            gt_actions = gt_actions.to(device)
            pred_actions = model.sample_actions(observation)
            err = (pred_actions - gt_actions).cpu().numpy()
            mse_per_step += (err ** 2).mean(axis=(0, 2))   # mean over batch & joints
            mse_per_joint += (err ** 2).mean(axis=(0, 1))  # mean over batch & timesteps
            count += 1

    if count > 0:
        mse_per_step /= count
        mse_per_joint /= count

    summary = {
        "checkpoint": str(ckpt_path),
        "num_samples": count,
        "mse_per_step": mse_per_step.tolist(),
        "mse_per_joint": mse_per_joint.tolist(),
        "overall_mse": float(mse_per_step.mean()),
    }

    out_path = ckpt_path / "eval_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Eval complete — {count} samples")
    print(f"  Overall MSE:  {summary['overall_mse']:.6f}")
    print(f"  Per-joint MSE: {[round(v, 6) for v in mse_per_joint.tolist()]}")
    print(f"  Summary:      {out_path}")


if __name__ == "__main__":
    main()
