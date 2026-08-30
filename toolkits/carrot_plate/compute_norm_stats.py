#!/usr/bin/env python3
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
"""Compute OpenPI quantile statistics for the carrot-plate SFT dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import openpi.shared.normalize as normalize
import openpi.training.data_loader as data_loader
import openpi.transforms as transforms
import tqdm

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

CONFIG_NAME = "pi05_carrot_plate_absolute_quat"


class RemoveStrings(transforms.DataTransformFn):
    """Drop prompts before numeric running-stat aggregation."""

    def __call__(self, value: dict) -> dict:
        return {
            key: item
            for key, item in value.items()
            if not np.issubdtype(np.asarray(item).dtype, np.str_)
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    model_path = args.model_path.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing stats: {output}")
    if not (dataset / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Not a local LeRobot dataset: {dataset}")
    if not (model_path / "model.safetensors").is_file():
        raise FileNotFoundError(f"Missing Pi0.5 model.safetensors: {model_path}")

    config = get_openpi_config(
        CONFIG_NAME,
        model_path=str(model_path),
        batch_size=args.batch_size,
        repo_id=str(dataset),
        data_kwargs={"norm_stats_path": str(output)},
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset_impl = data_loader.create_torch_dataset(
        data_config, config.model.action_horizon, config.model
    )
    transformed = data_loader.TransformedDataset(
        dataset_impl,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )
    num_batches = len(transformed) // args.batch_size
    loader = data_loader.TorchDataLoader(
        transformed,
        local_batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        num_batches=num_batches,
        framework="pytorch",
    )
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for batch in tqdm.tqdm(loader, total=num_batches, desc="Computing OpenPI stats"):
        for key in stats:
            stats[key].update(np.asarray(batch[key]))

    output.parent.mkdir(parents=True, exist_ok=True)
    normalize.save(
        output.parent,
        {key: running.get_statistics() for key, running in stats.items()},
    )
    if not output.is_file():
        raise RuntimeError(f"OpenPI did not write expected statistics file: {output}")
    print(
        f"OK: {len(transformed)} samples, horizon={config.model.action_horizon}, "
        f"padded state/action dim={config.model.action_dim}, stats={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
