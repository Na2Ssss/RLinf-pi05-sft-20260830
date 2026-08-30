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
"""Tests for the absolute-quaternion carrot-plate Pi0.5 data path."""

import numpy as np

from rlinf.data.datasets.openpi_rlinf import _resolve_env
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi.policies.carrot_plate_policy import (
    CarrotPlateInputs,
    CarrotPlateOutputs,
)
from toolkits.carrot_plate.convert_selected_raw_to_lerobot import (
    build_state_actions,
    canonicalize_quaternion_sequence,
)


def _z_quaternion(angle: float) -> np.ndarray:
    return np.array([0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)], dtype=np.float32)


def test_quaternion_sign_canonicalization_preserves_rotation() -> None:
    source = np.stack([_z_quaternion(0.0), -_z_quaternion(0.1), _z_quaternion(0.2)])
    canonical, flips = canonicalize_quaternion_sequence(source, _z_quaternion(0.0))

    assert flips == 1
    assert np.all(np.sum(canonical[1:] * canonical[:-1], axis=-1) >= 0)
    np.testing.assert_allclose(
        np.abs(np.sum(source * canonical, axis=-1)), np.ones(3), atol=1e-6
    )


def test_build_state_actions_keeps_absolute_8d_layout() -> None:
    length = 3
    state_quaternion = np.stack([_z_quaternion(value) for value in (0.0, 0.1, 0.2)])
    action_quaternion = np.stack([_z_quaternion(value) for value in (0.05, 0.15, 0.25)])
    data = {
        "robot_eef_pose": np.concatenate(
            [np.zeros((length, 3), dtype=np.float32), state_quaternion], axis=-1
        ),
        "robot_gripper": np.array([1, 1, 0]),
        "action": np.concatenate(
            [np.ones((length, 3), dtype=np.float32), action_quaternion], axis=-1
        ),
        "action_gripper": np.array([1, 0, 0]),
    }
    state, actions, corrections = build_state_actions(data, length, _z_quaternion(0.0))

    assert state.shape == actions.shape == (length, 8)
    assert corrections == {"state_sign_flips": 0, "action_sign_flips": 0}
    np.testing.assert_array_equal(state[:, 7], [1, 1, 0])
    np.testing.assert_array_equal(actions[:, 7], [1, 0, 0])
    np.testing.assert_allclose(actions[:, :3], 1.0)


def test_policy_maps_two_views_and_pads_state_actions() -> None:
    transform = CarrotPlateInputs(action_dim=32)
    result = transform(
        {
            "observation/state": np.arange(8, dtype=np.float32),
            "observation/front_image": np.zeros((3, 10, 12), dtype=np.float32),
            "observation/wrist_image": np.ones((10, 12, 3), dtype=np.uint8),
            "actions": np.ones((32, 8), dtype=np.float32),
            "prompt": b"Pick up the carrot.",
        }
    )

    assert result["state"].shape == (32,)
    assert result["actions"].shape == (32, 32)
    assert result["image"]["base_0_rgb"].shape == (10, 12, 3)
    assert result["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.True_,
        "right_wrist_0_rgb": np.False_,
    }
    assert result["prompt"] == "Pick up the carrot."
    np.testing.assert_array_equal(
        CarrotPlateOutputs()(result)["actions"], result["actions"][:, :8]
    )


def test_config_and_sft_loader_are_registered() -> None:
    config = get_openpi_config("pi05_carrot_plate_absolute_quat")

    assert config.model.pi05 is True
    assert config.model.action_horizon == 32
    assert config.model.action_dim == 32
    assert config.model.discrete_state_input is True
    assert _resolve_env(config.name) == "carrot_plate"
