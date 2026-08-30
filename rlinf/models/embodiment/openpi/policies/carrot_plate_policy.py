# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Policy transforms for the carrot-on-conveyor to plate task.

State and action use the same absolute eight-dimensional layout:
``[x, y, z, qx, qy, qz, qw, gripper]``.  The dataset converter canonicalizes
quaternion signs before these transforms run, so no component-wise rotation
delta is applied here.
"""

import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

ACTION_DIM = 8


def _parse_image(image: np.ndarray) -> np.ndarray:
    """Return an image as uint8 HWC RGB."""
    parsed = np.asarray(image)
    parsed = np.squeeze(parsed)
    if np.issubdtype(parsed.dtype, np.floating):
        parsed = (255 * parsed).clip(0, 255).astype(np.uint8)
    if parsed.ndim == 3 and parsed.shape[0] == 3:
        parsed = einops.rearrange(parsed, "c h w -> h w c")
    if parsed.ndim != 3 or parsed.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got {parsed.shape}")
    return parsed


@dataclasses.dataclass(frozen=True)
class CarrotPlateInputs(transforms.DataTransformFn):
    """Map front/wrist images and absolute quaternion poses to Pi0.5 inputs."""

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (ACTION_DIM,):
            raise ValueError(
                f"Expected carrot-plate state shape ({ACTION_DIM},), got {state.shape}"
            )

        front_image = _parse_image(data["observation/front_image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        inputs = {
            "state": transforms.pad_to_dim(state, self.action_dim),
            "image": {
                "base_0_rgb": front_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(front_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM:
                raise ValueError(
                    "Expected carrot-plate actions shape "
                    f"(H, {ACTION_DIM}), got {actions.shape}"
                )
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt
        return inputs


@dataclasses.dataclass(frozen=True)
class CarrotPlateOutputs(transforms.DataTransformFn):
    """Slice padded Pi0.5 output back to absolute eight-dimensional actions."""

    output_action_dim: int = ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.output_action_dim])}
