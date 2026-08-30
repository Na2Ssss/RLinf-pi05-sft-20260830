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
"""Convert a selected subset of carrot demonstrations to LeRobot v2.1.

The raw dataset is read-only.  Output state/action use absolute
``[xyz, quaternion_xyzw, gripper]`` values.  Quaternion signs are made
continuous within every trajectory and aligned to one dataset-wide reference;
this changes only the equivalent ``q``/``-q`` representation, never rotation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pickle
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ACTION_DIM = 8
CHUNK_SIZE = 1000
DEFAULT_FPS = 20
DEFAULT_IMAGE_SIZE = 224
IMAGE_DIRS = {
    "front_view": "Images_front_view",
    "wrist_view": "Images_wrist_view",
}
REQUIRED_ARRAYS = (
    "timestamp",
    "robot_eef_pose",
    "robot_gripper",
    "action",
    "action_gripper",
)


@dataclass(frozen=True)
class EpisodeSpec:
    """A validated source episode with deterministic destination indices."""

    source: Path
    source_episode_id: int
    episode_index: int
    length: int
    global_index: int


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        value = pickle.load(handle)  # noqa: S301 - trusted local robot data.
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected dict, got {type(value).__name__}")
    return value


def _frame_paths(image_dir: Path) -> list[Path]:
    frames = sorted(image_dir.glob("frame_*.jpg"))
    expected = [f"frame_{index:04d}.jpg" for index in range(len(frames))]
    if [path.name for path in frames] != expected:
        raise ValueError(
            f"{image_dir}: frame names are not contiguous from frame_0000.jpg"
        )
    return frames


def validate_episode(source: Path) -> int:
    """Validate one raw episode and return its common stream length."""
    data_path = source / "data.pkl"
    if not data_path.is_file():
        raise ValueError(f"{source}: missing data.pkl")
    data = _load_pickle(data_path)
    missing = [key for key in REQUIRED_ARRAYS if key not in data]
    if missing:
        raise ValueError(f"{source}: missing keys {missing}")

    lengths = {key: len(np.asarray(data[key])) for key in REQUIRED_ARRAYS}
    for view, subdir in IMAGE_DIRS.items():
        image_dir = source / subdir
        if not image_dir.is_dir():
            raise ValueError(f"{source}: missing {subdir}")
        lengths[view] = len(_frame_paths(image_dir))
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{source}: stream length mismatch {lengths}")
    length = next(iter(lengths.values()))
    if length < 32:
        raise ValueError(f"{source}: only {length} frames, fewer than horizon 32")

    timestamps = np.asarray(data["timestamp"], dtype=np.float64)
    if not np.all(np.isfinite(timestamps)) or not np.all(np.diff(timestamps) > 0):
        raise ValueError(f"{source}: invalid or non-monotonic timestamps")
    for key in ("robot_eef_pose", "action"):
        poses = np.asarray(data[key])
        if poses.shape != (length, 7) or not np.all(np.isfinite(poses)):
            raise ValueError(f"{source}: invalid {key} shape/data {poses.shape}")
        norms = np.linalg.norm(poses[:, 3:7], axis=-1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(
                f"{source}: {key} quaternion norm range "
                f"[{norms.min():.6f}, {norms.max():.6f}]"
            )
    for key in ("robot_gripper", "action_gripper"):
        gripper = np.asarray(data[key])
        if not np.all(np.isfinite(gripper)) or not np.all(np.isin(gripper, (0, 1))):
            raise ValueError(f"{source}: {key} is not finite binary data")
    return length


def canonicalize_quaternion_sequence(
    quaternion_xyzw: np.ndarray, reference_xyzw: np.ndarray
) -> tuple[np.ndarray, int]:
    """Normalize and sign-canonicalize one quaternion trajectory.

    Returns the equivalent quaternion sequence and the number of individual
    sign flips performed.  The first quaternion is aligned with the shared
    reference, then every subsequent quaternion is aligned with its predecessor.
    """
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float32).copy()
    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise ValueError(f"Expected quaternion shape (N, 4), got {quaternion.shape}")
    norms = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Encountered zero-norm quaternion")
    quaternion /= norms

    reference = np.asarray(reference_xyzw, dtype=np.float32)
    reference /= np.linalg.norm(reference)
    flips = 0
    if float(np.dot(quaternion[0], reference)) < 0:
        quaternion[0] *= -1
        flips += 1
    for index in range(1, len(quaternion)):
        if float(np.dot(quaternion[index], quaternion[index - 1])) < 0:
            quaternion[index] *= -1
            flips += 1
    return quaternion, flips


def build_state_actions(
    data: dict, length: int, reference_xyzw: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Build absolute 8-D arrays and record quaternion sign corrections."""
    state_pose = np.asarray(data["robot_eef_pose"], dtype=np.float32)[:length].copy()
    action_pose = np.asarray(data["action"], dtype=np.float32)[:length].copy()
    state_pose[:, 3:7], state_flips = canonicalize_quaternion_sequence(
        state_pose[:, 3:7], reference_xyzw
    )
    action_pose[:, 3:7], action_flips = canonicalize_quaternion_sequence(
        action_pose[:, 3:7], reference_xyzw
    )
    state = np.concatenate(
        [
            state_pose,
            np.asarray(data["robot_gripper"], dtype=np.float32)[:length, None],
        ],
        axis=-1,
    )
    actions = np.concatenate(
        [
            action_pose,
            np.asarray(data["action_gripper"], dtype=np.float32)[:length, None],
        ],
        axis=-1,
    )
    if state.shape != (length, ACTION_DIM) or actions.shape != (length, ACTION_DIM):
        raise AssertionError(f"Unexpected state/actions {state.shape}/{actions.shape}")
    return (
        state,
        actions,
        {"state_sign_flips": state_flips, "action_sign_flips": action_flips},
    )


def _encode_video(
    image_dir: Path,
    destination: Path,
    *,
    fps: int,
    length: int,
    image_size: int,
    ffmpeg_threads: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.mp4")
    scale_pad = (
        f"scale={image_size}:{image_size}:force_original_aspect_ratio=decrease,"
        f"pad={image_size}:{image_size}:(ow-iw)/2:(oh-ih)/2:black"
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-start_number",
        "0",
        "-i",
        str(image_dir / "frame_%04d.jpg"),
        "-frames:v",
        str(length),
        "-vf",
        scale_pad,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-threads",
        str(ffmpeg_threads),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    temporary.replace(destination)


def _write_parquet(
    destination: Path,
    *,
    state: np.ndarray,
    actions: np.ndarray,
    spec: EpisodeSpec,
    fps: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame_index = np.arange(spec.length, dtype=np.int64)
    frame = pd.DataFrame(
        {
            "state": [row.tolist() for row in state],
            "actions": [row.tolist() for row in actions],
            "timestamp": frame_index.astype(np.float32) / np.float32(fps),
            "frame_index": frame_index,
            "episode_index": np.full(spec.length, spec.episode_index, dtype=np.int64),
            "index": frame_index + np.int64(spec.global_index),
            "task_index": np.zeros(spec.length, dtype=np.int64),
        }
    )
    temporary = destination.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(destination)


def process_episode(
    spec: EpisodeSpec,
    output: Path,
    reference_xyzw: np.ndarray,
    fps: int,
    image_size: int,
    ffmpeg_threads: int,
) -> dict:
    """Convert one episode and return its source/correction metadata."""
    data = _load_pickle(spec.source / "data.pkl")
    state, actions, correction = build_state_actions(data, spec.length, reference_xyzw)
    chunk = f"chunk-{spec.episode_index // CHUNK_SIZE:03d}"
    episode_name = f"episode_{spec.episode_index:06d}"
    _write_parquet(
        output / "data" / chunk / f"{episode_name}.parquet",
        state=state,
        actions=actions,
        spec=spec,
        fps=fps,
    )
    for view, source_subdir in IMAGE_DIRS.items():
        _encode_video(
            spec.source / source_subdir,
            output / "videos" / chunk / view / f"{episode_name}.mp4",
            fps=fps,
            length=spec.length,
            image_size=image_size,
            ffmpeg_threads=ffmpeg_threads,
        )
    return {
        "episode_index": spec.episode_index,
        "source_episode_id": spec.source_episode_id,
        "source_path": str(spec.source),
        "length": spec.length,
        **correction,
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _episode_feature_stats(values: np.ndarray) -> dict[str, list[float]]:
    """Return the per-feature statistics required by LeRobot v2.1."""
    values = np.asarray(values)
    keepdims = values.ndim == 1
    return {
        "min": np.min(values, axis=0, keepdims=keepdims).tolist(),
        "max": np.max(values, axis=0, keepdims=keepdims).tolist(),
        "mean": np.mean(values, axis=0, keepdims=keepdims).tolist(),
        "std": np.std(values, axis=0, keepdims=keepdims).tolist(),
        "count": [len(values)],
    }


def write_episode_statistics(output: Path) -> None:
    """Write ``episodes_stats.jsonl`` for LeRobot 0.3/v2.1 readers."""
    entries = []
    scalar_keys = (
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    )
    for parquet in sorted((output / "data").rglob("episode_*.parquet")):
        frame = pd.read_parquet(parquet)
        episode_index = int(frame["episode_index"].iloc[0])
        stats = {
            "state": _episode_feature_stats(np.stack(frame["state"].to_numpy())),
            "actions": _episode_feature_stats(np.stack(frame["actions"].to_numpy())),
        }
        for key in scalar_keys:
            stats[key] = _episode_feature_stats(frame[key].to_numpy())
        entries.append({"episode_index": episode_index, "stats": stats})
    _write_jsonl(output / "meta" / "episodes_stats.jsonl", entries)


def write_metadata(
    output: Path,
    converted: list[dict],
    selection_manifest: dict,
    reference_xyzw: np.ndarray,
    *,
    fps: int,
    image_size: int,
) -> None:
    total_frames = sum(entry["length"] for entry in converted)
    prompt = selection_manifest["task_prompt"]
    pose_names = ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]
    features = {
        key: {"dtype": "float32", "shape": [ACTION_DIM], "names": pose_names}
        for key in ("state", "actions")
    }
    for view in IMAGE_DIRS:
        features[view] = {
            "dtype": "video",
            "shape": [image_size, image_size, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": image_size,
                "video.width": image_size,
                "video.codec": "libx264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }
    for key, dtype in (
        ("timestamp", "float32"),
        ("frame_index", "int64"),
        ("episode_index", "int64"),
        ("index", "int64"),
        ("task_index", "int64"),
    ):
        features[key] = {"dtype": dtype, "shape": [1], "names": None}

    info = {
        "codebase_version": "v2.1",
        "robot_type": "panda",
        "total_episodes": len(converted),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(converted) * len(IMAGE_DIRS),
        "total_chunks": 1,
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{len(converted)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    _write_json(output / "meta" / "info.json", info)
    _write_jsonl(output / "meta" / "tasks.jsonl", [{"task_index": 0, "task": prompt}])
    _write_jsonl(
        output / "meta" / "episodes.jsonl",
        [
            {
                "episode_index": entry["episode_index"],
                "tasks": [prompt],
                "length": entry["length"],
            }
            for entry in converted
        ],
    )
    _write_jsonl(output / "meta" / "source_episodes.jsonl", converted)
    output_manifest = dict(selection_manifest)
    output_manifest["quaternion_reference_xyzw"] = reference_xyzw.tolist()
    output_manifest["output_episode_mapping"] = [
        {
            "output_episode_index": entry["episode_index"],
            "source_episode_id": entry["source_episode_id"],
        }
        for entry in converted
    ]
    _write_json(output / "meta" / "selection_manifest.json", output_manifest)


def _numeric_stats(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def write_raw_statistics(output: Path) -> None:
    state_chunks = []
    action_chunks = []
    timestamp_chunks = []
    for parquet in tqdm(
        sorted((output / "data").rglob("episode_*.parquet")),
        desc="Dataset statistics",
    ):
        frame = pd.read_parquet(parquet, columns=["state", "actions", "timestamp"])
        state_chunks.append(np.stack(frame["state"].to_numpy()).astype(np.float32))
        action_chunks.append(np.stack(frame["actions"].to_numpy()).astype(np.float32))
        timestamp_chunks.append(frame["timestamp"].to_numpy(dtype=np.float32)[:, None])
    stats = {
        "state": _numeric_stats(np.concatenate(state_chunks)),
        "actions": _numeric_stats(np.concatenate(action_chunks)),
        "timestamp": _numeric_stats(np.concatenate(timestamp_chunks)),
    }
    _write_json(output / "meta" / "stats.json", stats)


def verify_output(output: Path, specs: list[EpisodeSpec], image_size: int) -> None:
    """Verify indexing, quaternion continuity, videos, and aggregate counts."""
    info = json.loads((output / "meta" / "info.json").read_text())
    if info["total_episodes"] != len(specs):
        raise ValueError("Output episode count mismatch")
    if info["total_frames"] != sum(spec.length for spec in specs):
        raise ValueError("Output frame count mismatch")

    previous_end = 0
    for spec in tqdm(specs, desc="Verifying output"):
        chunk = f"chunk-{spec.episode_index // CHUNK_SIZE:03d}"
        episode_name = f"episode_{spec.episode_index:06d}"
        parquet = output / "data" / chunk / f"{episode_name}.parquet"
        frame = pd.read_parquet(parquet)
        if len(frame) != spec.length:
            raise ValueError(
                f"{parquet}: expected {spec.length} rows, got {len(frame)}"
            )
        indices = frame["index"].to_numpy()
        if indices[0] != previous_end or indices[-1] != previous_end + spec.length - 1:
            raise ValueError(f"{parquet}: non-contiguous global indices")
        previous_end += spec.length

        for key in ("state", "actions"):
            values = np.stack(frame[key].to_numpy()).astype(np.float32)
            if values.shape != (spec.length, ACTION_DIM) or not np.all(
                np.isfinite(values)
            ):
                raise ValueError(f"{parquet}: invalid {key}")
            quaternion = values[:, 3:7]
            if not np.allclose(np.linalg.norm(quaternion, axis=-1), 1.0, atol=1e-5):
                raise ValueError(f"{parquet}: non-unit {key} quaternion")
            if np.any(np.sum(quaternion[1:] * quaternion[:-1], axis=-1) < -1e-6):
                raise ValueError(f"{parquet}: discontinuous q/-q signs in {key}")

        for view in IMAGE_DIRS:
            video = output / "videos" / chunk / view / f"{episode_name}.mp4"
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,nb_read_frames",
                    "-of",
                    "json",
                    str(video),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            stream = json.loads(result.stdout)["streams"][0]
            if (int(stream["width"]), int(stream["height"])) != (
                image_size,
                image_size,
            ):
                raise ValueError(f"{video}: unexpected dimensions {stream}")
            if int(stream["nb_read_frames"]) != spec.length:
                raise ValueError(f"{video}: unexpected frame count {stream}")


def load_selection(path: Path) -> tuple[dict, list[int]]:
    manifest = json.loads(path.read_text())
    selected = [
        int(value) for value in manifest["selected_episode_ids_in_output_order"]
    ]
    if len(selected) != 20 or len(set(selected)) != 20:
        raise ValueError(f"Selection must contain 20 unique episodes, got {selected}")
    split_selected = [
        int(value)
        for split in manifest["splits"]
        for value in split["selected_episode_ids"]
    ]
    if split_selected != selected:
        raise ValueError("Split selections do not match output-order selection")
    for split in manifest["splits"]:
        lower, upper = map(int, split["population_episode_ids"])
        values = [int(value) for value in split["selected_episode_ids"]]
        if len(values) != int(split["sample_size"]):
            raise ValueError(f"{split['name']}: wrong sample size")
        if any(value < lower or value > upper for value in values):
            raise ValueError(f"{split['name']}: selected ID outside [{lower}, {upper}]")
    return manifest, selected


def discover_specs(raw_root: Path, selected: list[int]) -> list[EpisodeSpec]:
    specs = []
    global_index = 0
    for output_index, source_id in enumerate(tqdm(selected, desc="Validating source")):
        source = raw_root / f"episode_{source_id:04d}"
        length = validate_episode(source)
        specs.append(EpisodeSpec(source, source_id, output_index, length, global_index))
        global_index += length
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--ffmpeg-threads", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selection_manifest = args.selection_manifest.resolve()
    manifest, selected = load_selection(selection_manifest)
    raw_root = Path(manifest["raw_dataset_root"]).resolve()
    output = args.output.resolve()
    staging = output.with_name(f".{output.name}.incomplete")
    if not raw_root.is_dir():
        print(f"error: raw dataset does not exist: {raw_root}", file=sys.stderr)
        return 2
    if output.exists() or staging.exists():
        print(
            f"error: output or staging path already exists: {output} / {staging}",
            file=sys.stderr,
        )
        return 2

    specs = discover_specs(raw_root, selected)
    first_data = _load_pickle(specs[0].source / "data.pkl")
    reference = np.asarray(first_data["robot_eef_pose"], dtype=np.float32)[0, 3:7]
    reference /= np.linalg.norm(reference)
    staging.mkdir(parents=True)
    print(
        f"Validated {len(specs)} episodes / {sum(spec.length for spec in specs)} frames"
    )
    print(f"Quaternion reference xyzw: {reference.tolist()}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_episode,
                spec,
                staging,
                reference,
                args.fps,
                args.image_size,
                args.ffmpeg_threads,
            ): spec
            for spec in specs
        }
        results = {}
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Converting episodes",
        ):
            spec = futures[future]
            results[spec.episode_index] = future.result()
    converted = [results[index] for index in range(len(specs))]

    write_metadata(
        staging,
        converted,
        manifest,
        reference,
        fps=args.fps,
        image_size=args.image_size,
    )
    write_episode_statistics(staging)
    write_raw_statistics(staging)
    verify_output(staging, specs, args.image_size)
    staging.replace(output)
    print(
        f"OK: {len(specs)} episodes, {sum(spec.length for spec in specs)} frames, "
        f"absolute [xyz, quaternion_xyzw, gripper] at {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
