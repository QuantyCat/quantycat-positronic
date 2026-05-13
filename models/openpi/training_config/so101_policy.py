"""
SO-101 data transforms for openpi.

Drop-in equivalent of openpi's libero_policy.py for the SO-101 robot.
Maps LeRobot SO-101 dataset observations into the format expected by
pi0 models, and maps predicted actions back out to SO-101 space.

Key conventions shared with rynnvla-002 training:
  - state:   6D joint angles, stored in degrees in the LeRobot dataset
  - actions: 6D absolute joint positions in degrees (raw from dataset)
  - The DeltaActions transform in the data config converts joints 0-4 to
    delta actions; gripper (joint 5) stays absolute.

This file is installed into ~/openpi/src/openpi/policies/ by setup.sh
so that openpi's training infrastructure can import it.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_so101_example() -> dict:
    """Random input example — useful for smoke-testing the policy."""
    return {
        "observation/state": np.random.rand(6).astype(np.float32),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        # LeRobot stores images as (C, H, W); openpi expects (H, W, C)
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SO101Inputs(transforms.DataTransformFn):
    """Convert SO-101 observations to openpi's standard input format.

    Applied during both training (after repack) and inference.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad missing right-wrist slot with zeros (SO-101 has only one wrist camera)
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # pi0-FAST uses all image slots; pi0 masks the padded ones
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    """Strip padding from model output actions and return SO-101-shaped actions.

    Applied during inference only.
    """

    action_dim: int = 6

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
