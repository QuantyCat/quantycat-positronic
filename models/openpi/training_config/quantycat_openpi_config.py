"""Config snippet for adding Quantycat SO-101 pi05 training to openpi.

This file is a source-of-truth snippet for the changes that should be copied
or adapted into openpi's `src/openpi/training/config.py`.

Expected openpi-side install:
  1. Copy `quantycat_policy.py` to `src/openpi/policies/quantycat_policy.py`.
  2. Add `import openpi.policies.quantycat_policy as quantycat_policy`.
  3. Add `LeRobotQuantycatDataConfig`.
  4. Add the `pi05_quantycat` TrainConfig to `_CONFIGS`.
"""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Sequence

from typing_extensions import override

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.policies.quantycat_policy as quantycat_policy
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
from openpi.training.config import AssetsConfig
from openpi.training.config import DataConfig
from openpi.training.config import DataConfigFactory
from openpi.training.config import ModelTransformFactory
from openpi.training.config import TrainConfig


@dataclasses.dataclass(frozen=True)
class LeRobotQuantycatDataConfig(DataConfigFactory):
    """LeRobot data config for Quantycat SO-101 screwdriver data."""

    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/images/front": "observation.images.front",
                        "observation/images/wrist": "observation.images.wrist",
                        "observation/state": "observation.state",
                        "action": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[quantycat_policy.QuantycatInputs()],
            outputs=[quantycat_policy.QuantycatOutputs(action_dim=6)],
        )

        # Raw dataset actions are absolute 6-D joint/gripper targets.
        # Train joints 0-4 as deltas relative to current state; leave gripper absolute.
        delta_action_mask = _transforms.make_bool_mask(5, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


PI05_QUANTYCAT_CONFIG = TrainConfig(
    name="pi05_quantycat",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=20),
    data=LeRobotQuantycatDataConfig(
        repo_id="/home/caroline/quantycat-positronic/my_data/input_data",
        assets=AssetsConfig(
            assets_dir="/home/caroline/quantycat-positronic/my_data/training_pipeline",
            asset_id="openpi",
        ),
        base_config=DataConfig(prompt_from_task=True),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    num_train_steps=5_000,
    batch_size=8,
    num_workers=2,
    save_interval=1000,
    keep_period=5000,
    wandb_enabled=False,
    checkpoint_base_dir="/home/caroline/quantycat-positronic/my_data/training_pipeline/openpi/checkpoints",
)
