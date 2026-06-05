"""Quantycat SO-101 screwdriver fine-tuning configs for openpi/pi0.

Dataset discovery uses HF_LEROBOT_HOME (set by run scripts to
$QUANTYCAT_DATA_HOME/datasets). Norm stats are loaded directly from the
LeRobot dataset's meta/stats.json (populated by augment_dataset_quantile_stats).
"""

import json
import os
import pathlib

import numpy as np

_DATA_HOME = os.environ.get("QUANTYCAT_DATA_HOME", str(pathlib.Path.home() / "quantycat-data"))


def get_quantycat_configs():
    # Deferred imports avoid circular dependency with config.py.
    import dataclasses

    import openpi.models.model as _model
    import openpi.models.pi0_config as pi0_config
    import openpi.training.weight_loaders as weight_loaders
    import openpi.transforms as _transforms
    from openpi.policies import quantycat_policy
    from typing_extensions import override

    from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory, TrainConfig

    @dataclasses.dataclass(frozen=True)
    class LeRobotQuantycatDataConfig(DataConfigFactory):
        """LeRobot data config for Quantycat SO-101 data.

        repo_id is a short dataset name resolved via HF_LEROBOT_HOME.
        stats_repo_id is the dataset to read norm stats from (defaults to repo_id).
          Use this when the training dataset is v2 format but stats are on a v3 copy.
        right_wrist_source controls which image fills the right_wrist_0_rgb slot;
        see quantycat_policy.QuantycatInputs for valid values.
        """

        right_wrist_source: str = "wrist"
        action_sequence_keys: tuple = ("action",)
        stats_repo_id: str | None = None

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

            model_transforms = ModelTransformFactory()(model_config)

            norm_stats = self._load_lerobot_norm_stats()

            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                repack_transforms=repack_transform,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                action_sequence_keys=self.action_sequence_keys,
                norm_stats=norm_stats,
            )

        def _load_lerobot_norm_stats(self) -> dict[str, _transforms.NormStats]:
            hf_home = os.environ.get("HF_LEROBOT_HOME", os.path.join(_DATA_HOME, "datasets"))
            stats_repo = self.stats_repo_id or self.repo_id
            stats_path = pathlib.Path(hf_home) / stats_repo / "meta" / "stats.json"

            if not stats_path.exists():
                raise FileNotFoundError(
                    f"LeRobot stats.json not found at {stats_path}.\n"
                    "Run preprocessing first:\n"
                    "  bash models/lerobot/run_scripts/preprocess.sh"
                )

            stats = json.loads(stats_path.read_text())

            for lerobot_key in ("observation.state", "action"):
                entry = stats.get(lerobot_key, {})
                if "q01" not in entry or "q99" not in entry:
                    raise ValueError(
                        f"q01/q99 missing for '{lerobot_key}' in {stats_path}.\n"
                        "Run preprocessing first:\n"
                        "  bash models/lerobot/run_scripts/preprocess.sh"
                    )

            def _to_norm_stats(entry: dict) -> _transforms.NormStats:
                return _transforms.NormStats(
                    mean=np.array(entry["mean"], dtype=np.float32),
                    std=np.array(entry["std"], dtype=np.float32),
                    q01=np.array(entry["q01"], dtype=np.float32),
                    q99=np.array(entry["q99"], dtype=np.float32),
                )

            # Map LeRobot key names to the keys QuantycatInputs produces.
            return {
                "state": _to_norm_stats(stats["observation.state"]),
                "actions": _to_norm_stats(stats["action"]),
            }

    _lora_freeze = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=50,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter()

    def _lora_model():
        return pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        )

    _checkpoint_base = os.path.join(_DATA_HOME, "checkpoints", "openpi")
    _pi05_weights = "gs://openpi-assets/checkpoints/pi05_base/params"

    return [
        #
        # Stable Quantycat SO-101 screwdriver LoRA fine-tuning config.
        #
        # Dataset paths resolved via HF_LEROBOT_HOME=$QUANTYCAT_DATA_HOME/datasets.
        # Norm stats loaded from screwdriver_so101_clean_v3/meta/stats.json
        # (v3 is identical data to v2 but in the format augment_dataset_quantile_stats requires).
        # Experimental variants are archived under quantycat-research/openpi_experiments/configs/.
        #
        TrainConfig(
            name="pi05_quantycat_lora",
            model=_lora_model(),
            data=LeRobotQuantycatDataConfig(
                repo_id="screwdriver_so101_clean_v2",
                stats_repo_id="screwdriver_so101_clean_v3",
                base_config=DataConfig(prompt_from_task=True, train_episodes=list(range(44))),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_pi05_weights),
            num_train_steps=10_000,
            freeze_filter=_lora_freeze,
            ema_decay=None,
            batch_size=4,
            num_workers=2,
            save_interval=1000,
            keep_period=5000,
            wandb_enabled=False,
            checkpoint_base_dir=_checkpoint_base,
        ),
    ]
