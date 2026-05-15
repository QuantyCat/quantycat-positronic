"""SO-101 policy transforms for Quantycat openpi / pi05 training.

These transforms map the local LeRobot SO-101 dataset format into the
image/state/action format expected by openpi models.

Dataset keys after repacking:
  observation/images/front
  observation/images/wrist
  observation/state
  action
  prompt

Conventions:
  - state is 6-D absolute joint/gripper position.
  - action is 6-D absolute joint/gripper target before DeltaActions.
  - padding to the model action dimension is intentionally left to openpi's
    ModelTransformFactory, after normalization.
"""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms


REAL_ACTION_DIM = 6


def make_quantycat_example() -> dict:
    """Create a random unbatched example for transform smoke tests."""
    return {
        "observation/state": np.random.rand(REAL_ACTION_DIM).astype(np.float32),
        "observation/images/front": np.random.randint(256, size=(360, 640, 3), dtype=np.uint8),
        "observation/images/wrist": np.random.randint(256, size=(360, 640, 3), dtype=np.uint8),
        "prompt": "Put the screwdriver into the cup",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class QuantycatInputs(transforms.DataTransformFn):
    """Convert Quantycat SO-101 observations to openpi model inputs."""

    def __call__(self, data: dict) -> dict:
        front_image = _parse_image(data["observation/images/front"])
        wrist_image = _parse_image(data["observation/images/wrist"])

        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": front_image,
                "left_wrist_0_rgb": wrist_image,
                # SO-101 has one wrist camera. Feed it into both wrist slots
                # instead of masking out the missing right-wrist view.
                "right_wrist_0_rgb": wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "action" in data:
            inputs["actions"] = np.asarray(data["action"], dtype=np.float32)
        elif "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class QuantycatOutputs(transforms.DataTransformFn):
    """Trim openpi model actions back to the real SO-101 6-D action space."""

    action_dim: int = REAL_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
