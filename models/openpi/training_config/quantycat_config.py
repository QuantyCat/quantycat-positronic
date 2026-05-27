"""Quantycat SO-101 screwdriver fine-tuning configs for openpi/pi0.

Dataset discovery uses HF_LEROBOT_HOME (set by run scripts to
$QUANTYCAT_DATA_HOME/datasets). Norm stats live under
$QUANTYCAT_DATA_HOME/norm_stats/openpi/<config_name>/<repo_id>/.
"""

import os
import pathlib

_DATA_HOME = os.environ.get("QUANTYCAT_DATA_HOME", str(pathlib.Path.home() / "quantycat-data"))


def get_quantycat_configs():
    # Deferred imports avoid circular dependency with config.py.
    import dataclasses

    import openpi.models.model as _model
    import openpi.models.pi0_config as pi0_config
    import openpi.training.weight_loaders as weight_loaders
    import openpi.transforms as _transforms
    import quantycat_policy
    from typing_extensions import override

    from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory, TrainConfig

    @dataclasses.dataclass(frozen=True)
    class LeRobotQuantycatDataConfig(DataConfigFactory):
        """LeRobot data config for Quantycat SO-101 data.

        repo_id is a short dataset name resolved via HF_LEROBOT_HOME.
        right_wrist_source controls which image fills the right_wrist_0_rgb slot;
        see quantycat_policy.QuantycatInputs for valid values.
        """

        right_wrist_source: str = "wrist"
        action_sequence_keys: tuple = ("action",)

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
                inputs=[quantycat_policy.QuantycatInputs(right_wrist_source=self.right_wrist_source)],
                outputs=[quantycat_policy.QuantycatOutputs(action_dim=6)],
            )

            # Joints 0-4 trained as deltas; gripper stays absolute.
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

    _lora_freeze = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=20,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter()

    def _lora_model():
        return pi0_config.Pi0Config(
            pi05=True,
            action_horizon=20,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        )

    _norm_stats_base = os.path.join(_DATA_HOME, "norm_stats", "openpi")
    _checkpoint_base = os.path.join(_DATA_HOME, "checkpoints", "openpi")
    _pi05_weights = "gs://openpi-assets/checkpoints/pi05_base/params"

    return [
        #
        # Quantycat SO-101 screwdriver fine-tuning configs.
        #
        # Dataset paths resolved via HF_LEROBOT_HOME=$QUANTYCAT_DATA_HOME/datasets.
        # Norm stats at $QUANTYCAT_DATA_HOME/norm_stats/openpi/<config>/<repo_id>/.
        #
        TrainConfig(
            name="pi05_quantycat",
            model=pi0_config.Pi0Config(pi05=True, action_horizon=20),
            data=LeRobotQuantycatDataConfig(
                repo_id="screwdriver_so101_clean",
                base_config=DataConfig(prompt_from_task=True),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_pi05_weights),
            num_train_steps=5_000,
            ema_decay=None,
            batch_size=4,
            num_workers=2,
            save_interval=1000,
            keep_period=5000,
            wandb_enabled=False,
            assets_base_dir=_norm_stats_base,
            checkpoint_base_dir=_checkpoint_base,
        ),
        TrainConfig(
            name="pi05_quantycat_lora",
            model=_lora_model(),
            data=LeRobotQuantycatDataConfig(
                repo_id="screwdriver_so101_clean",
                base_config=DataConfig(prompt_from_task=True),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_pi05_weights),
            num_train_steps=5_000,
            freeze_filter=_lora_freeze,
            ema_decay=None,
            batch_size=4,
            num_workers=2,
            save_interval=1000,
            keep_period=5000,
            wandb_enabled=False,
            assets_base_dir=_norm_stats_base,
            checkpoint_base_dir=_checkpoint_base,
        ),
        TrainConfig(
            name="pi05_quantycat_lora_achieved_delta",
            model=_lora_model(),
            data=LeRobotQuantycatDataConfig(
                repo_id="screwdriver_so101_clean_achieved_delta",
                base_config=DataConfig(prompt_from_task=True),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_pi05_weights),
            num_train_steps=5_000,
            freeze_filter=_lora_freeze,
            ema_decay=None,
            batch_size=4,
            num_workers=2,
            save_interval=1000,
            keep_period=5000,
            wandb_enabled=False,
            assets_base_dir=_norm_stats_base,
            checkpoint_base_dir=_checkpoint_base,
        ),
        TrainConfig(
            name="pi05_quantycat_lora_achieved_delta_train39",
            model=_lora_model(),
            data=LeRobotQuantycatDataConfig(
                repo_id="screwdriver_so101_clean_achieved_delta_train39",
                base_config=DataConfig(prompt_from_task=True),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_pi05_weights),
            num_train_steps=5_000,
            freeze_filter=_lora_freeze,
            ema_decay=None,
            batch_size=4,
            num_workers=2,
            save_interval=1000,
            keep_period=5000,
            wandb_enabled=False,
            assets_base_dir=_norm_stats_base,
            checkpoint_base_dir=_checkpoint_base,
        ),
    ]
