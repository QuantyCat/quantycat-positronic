"""
SO-101 TrainConfig builder for openpi.

Provides build_train_config(yaml_cfg) which constructs the openpi TrainConfig
for the SO-101 robot from the values in models/openpi/config.yaml.

Calling register_with_openpi(openpi_src_path, positronic_models_openpi_path)
ensures so101_policy is importable and appends the SO-101 configs to
openpi's _CONFIGS / _CONFIGS_DICT so that train_pytorch.py can find them.

Usage (called by finetune.py and step1_compute_norm_stats.py):

    import sys, pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    openpi_repo = pathlib.Path.home() / "openpi"
    sys.path.insert(0, str(openpi_repo / "src"))
    sys.path.insert(0, str(repo_root / "models" / "openpi" / "training_config"))
    from so101_train_config import build_train_config, register_with_openpi
    register_with_openpi()
    cfg = build_train_config(yaml_cfg)
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys


def _ensure_imports():
    """Make sure openpi and so101_policy are importable."""
    openpi_src = pathlib.Path.home() / "openpi" / "src"
    training_config_dir = pathlib.Path(__file__).resolve().parent
    for p in (str(openpi_src), str(training_config_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)


def build_train_config(yaml_cfg: dict):
    """Construct an openpi TrainConfig from config.yaml values.

    Parameters
    ----------
    yaml_cfg : dict
        Loaded from models/openpi/config.yaml via yaml.safe_load.

    Returns
    -------
    openpi.training.config.TrainConfig
    """
    _ensure_imports()

    import openpi.models.pi0_config as pi0_config
    import openpi.models.pi0_fast as pi0_fast
    import openpi.training.config as _config
    import openpi.training.optimizer as _optimizer
    import openpi.training.weight_loaders as weight_loaders
    import openpi.transforms as _transforms
    import so101_policy

    action_dim = int(yaml_cfg["action_dim"])
    action_horizon = int(yaml_cfg["action_horizon"])
    max_token_len = int(yaml_cfg["max_token_len"])
    model_type_str = str(yaml_cfg.get("model_type", "pi0"))
    hf_repo_id = yaml_cfg.get("hf_repo_id") or None
    training_run_name = str(yaml_cfg.get("training_run_name", "so101_openpi"))
    work_dir = pathlib.Path(yaml_cfg["work_dir"]).resolve()
    base_pytorch_ckpt = yaml_cfg.get("base_pytorch_checkpoint") or None

    # Build the model config
    if model_type_str == "pi0_fast":
        model_cfg = pi0_fast.Pi0FASTConfig(
            action_dim=action_dim,
            action_horizon=action_horizon,
            max_token_len=max_token_len,
        )
        model_type_enum = None  # resolved from model_cfg inside the data config
    elif model_type_str == "pi0_lora":
        model_cfg = pi0_config.Pi0Config(
            action_dim=action_dim,
            action_horizon=action_horizon,
            max_token_len=max_token_len,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        )
    else:  # pi0 (full fine-tune, default)
        model_cfg = pi0_config.Pi0Config(
            action_dim=action_dim,
            action_horizon=action_horizon,
            max_token_len=max_token_len,
        )

    # Repack transform: maps LeRobot dataset keys → intermediate keys
    # SO-101 LeRobot dataset stores:
    #   observation.images.front  — front camera (C,H,W float)
    #   observation.images.wrist  — wrist camera (C,H,W float)
    #   observation.state         — 6D joint angles (degrees)
    #   action                    — 6D absolute joint target (degrees)
    repack = _transforms.Group(
        inputs=[
            _transforms.RepackTransform(
                {
                    "observation/image": "observation.images.front",
                    "observation/wrist_image": "observation.images.wrist",
                    "observation/state": "observation.state",
                    "actions": "action",
                }
            )
        ]
    )

    # Delta-action conversion: joints 0-4 become relative to current state;
    # gripper (joint 5) stays absolute — matches rynnvla-002 training convention.
    delta_mask = _transforms.make_bool_mask(action_dim - 1, -1)  # [True]*5 + [False]

    data_transforms = _transforms.Group(
        inputs=[so101_policy.SO101Inputs(model_type=model_cfg.model_type)],
        outputs=[so101_policy.SO101Outputs(action_dim=action_dim)],
    ).push(
        inputs=[_transforms.DeltaActions(delta_mask)],
        outputs=[_transforms.AbsoluteActions(delta_mask)],
    )

    model_transforms = _config.ModelTransformFactory()(model_cfg)

    data_cfg = _config.DataConfig(
        repo_id=hf_repo_id,
        asset_id=training_run_name,
        repack_transforms=repack,
        data_transforms=data_transforms,
        model_transforms=model_transforms,
        use_quantile_norm=(model_type_str != "pi0"),
        prompt_from_task=True,
        action_sequence_keys=("action",),
    )

    # SimpleDataConfig wrapper (satisfies create() protocol)
    @dataclasses.dataclass(frozen=True)
    class _DirectDataConfig(_config.DataConfigFactory):
        _data_cfg: _config.DataConfig = dataclasses.field(default_factory=_config.DataConfig)

        def create(self, assets_dirs, model_config):
            return self._data_cfg

    # Weight loader: prefer local PyTorch checkpoint if supplied, else GCS base
    if base_pytorch_ckpt:
        wl = weight_loaders.NoOpWeightLoader()
    else:
        if model_type_str == "pi0_fast":
            gcs_path = "gs://openpi-assets/checkpoints/pi0_fast_base/params"
        else:
            gcs_path = "gs://openpi-assets/checkpoints/pi0_base/params"
        wl = weight_loaders.CheckpointWeightLoader(gcs_path)

    # Build TrainConfig
    lr = float(yaml_cfg.get("lr", 5e-5))
    warmup = int(yaml_cfg.get("warmup_steps", 1000))
    train_cfg = _config.TrainConfig(
        name=f"pi0_so101_{model_type_str}",
        model=model_cfg,
        data=_DirectDataConfig(_data_cfg=data_cfg),
        weight_loader=wl,
        pytorch_weight_path=base_pytorch_ckpt,
        batch_size=int(yaml_cfg.get("batch_size", 4)),
        num_workers=int(yaml_cfg.get("num_workers", 2)),
        num_train_steps=int(yaml_cfg.get("num_train_steps", 30000)),
        seed=int(yaml_cfg.get("seed", 42)),
        save_interval=int(yaml_cfg.get("save_interval", 1000)),
        wandb_enabled=bool(yaml_cfg.get("wandb_enabled", False)),
        overwrite=bool(yaml_cfg.get("overwrite", False)),
        resume=bool(yaml_cfg.get("resume", False)),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=warmup,
            peak_lr=lr,
            decay_steps=int(yaml_cfg.get("num_train_steps", 30000)),
            decay_lr=lr * 0.1,
        ),
        assets_base_dir=str(work_dir / "openpi_assets"),
        checkpoint_base_dir=str(work_dir / "fine_tuning"),
        exp_name=training_run_name,
    )

    return train_cfg


def register_with_openpi(train_cfg=None):
    """Register SO-101 config(s) with openpi's global _CONFIGS_DICT.

    After this call, train_pytorch.py can look up the config by name.
    Safe to call multiple times.
    """
    _ensure_imports()
    import openpi.training.config as _config

    if train_cfg is None:
        return

    if train_cfg.name not in _config._CONFIGS_DICT:
        _config._CONFIGS.append(train_cfg)
        _config._CONFIGS_DICT[train_cfg.name] = train_cfg
