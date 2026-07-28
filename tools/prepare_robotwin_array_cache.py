#!/usr/bin/env python3
"""Decode one RoboTwin clean-50 task into compact memory-mapped arrays."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np


CAMERAS = ("head_camera", "left_camera", "right_camera")


def episode_files(raw_root: Path, expected_episodes: int) -> list[Path]:
    data_root = raw_root / "data"
    paths = [data_root / f"episode{i}.hdf5" for i in range(expected_episodes)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} RoboTwin episodes; first: {missing[0]}"
        )
    unexpected = sorted(data_root.glob("episode*.hdf5"))
    if len(unexpected) != expected_episodes:
        raise ValueError(
            f"Expected exactly {expected_episodes} episodes in {data_root}, "
            f"found {len(unexpected)}"
        )
    return paths


def inspect_episode(path: Path) -> tuple[int, int]:
    with h5py.File(path, "r") as source:
        vector = source["joint_action/vector"]
        if vector.ndim != 2:
            raise ValueError(f"Invalid joint vector in {path}: {vector.shape}")
        frames, state_dim = vector.shape
        if frames < 2:
            raise ValueError(f"Episode is too short: {path}")
        for camera in CAMERAS:
            rgb = source[f"observation/{camera}/rgb"]
            if rgb.shape[0] != frames:
                raise ValueError(
                    f"{path} {camera} has {rgb.shape[0]} frames, expected {frames}"
                )
        return frames - 1, state_dim


def decode_rgb(encoded, image_size: int) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("OpenCV failed to decode a RoboTwin JPEG frame")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(
        frame, (image_size, image_size), interpolation=cv2.INTER_AREA
    )
    return np.moveaxis(frame, -1, 0)


def validate_existing(output_root: Path, expected_episodes: int, image_size: int):
    marker_path = output_root / ".jepa_robotwin_cache.json"
    if not marker_path.is_file():
        raise FileExistsError(
            f"Refusing to replace unvalidated existing output: {output_root}"
        )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected = {
        "cache_format_version": 1,
        "partial": False,
        "total_episodes": expected_episodes,
        "image_size": image_size,
        "color_order": "RGB",
        "transition_alignment": "obs[t], action=qpos[t+1]",
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise ValueError(
                f"Existing RoboTwin cache has {key}={marker.get(key)!r}, "
                f"expected {value!r}"
            )
    total_frames = int(marker["total_frames"])
    expected_shapes = {
        "head_camera.npy": (total_frames, 3, image_size, image_size),
        "left_camera.npy": (total_frames, 3, image_size, image_size),
        "right_camera.npy": (total_frames, 3, image_size, image_size),
        "state.npy": (total_frames, int(marker["state_dim"])),
        "action.npy": (total_frames, int(marker["action_dim"])),
        "episode_ends.npy": (expected_episodes,),
    }
    for filename, shape in expected_shapes.items():
        array = np.load(output_root / filename, mmap_mode="r")
        if array.shape != shape:
            raise ValueError(f"Invalid {filename}: {array.shape}, expected {shape}")
    print(json.dumps(marker, indent=2, sort_keys=True))


def build_cache(
    raw_root: Path,
    output_root: Path,
    expected_episodes: int,
    image_size: int,
):
    if output_root.exists():
        validate_existing(output_root, expected_episodes, image_size)
        return

    paths = episode_files(raw_root, expected_episodes)
    inspections = [inspect_episode(path) for path in paths]
    lengths = [item[0] for item in inspections]
    state_dims = {item[1] for item in inspections}
    if len(state_dims) != 1:
        raise ValueError(f"Inconsistent state dimensions: {sorted(state_dims)}")
    state_dim = state_dims.pop()
    action_dim = state_dim
    total_frames = int(sum(lengths))

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.building.", dir=output_root.parent
        )
    )
    marker_path = staging / ".jepa_robotwin_cache.json"
    marker_path.write_text(
        json.dumps(
            {
                "cache_format_version": 1,
                "partial": True,
                "total_episodes": expected_episodes,
                "total_frames": total_frames,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    image_shape = (total_frames, 3, image_size, image_size)
    arrays = {
        "head_camera": np.lib.format.open_memmap(
            staging / "head_camera.npy", mode="w+", dtype=np.uint8, shape=image_shape
        ),
        "left_camera": np.lib.format.open_memmap(
            staging / "left_camera.npy", mode="w+", dtype=np.uint8, shape=image_shape
        ),
        "right_camera": np.lib.format.open_memmap(
            staging / "right_camera.npy", mode="w+", dtype=np.uint8, shape=image_shape
        ),
        "state": np.lib.format.open_memmap(
            staging / "state.npy",
            mode="w+",
            dtype=np.float32,
            shape=(total_frames, state_dim),
        ),
        "action": np.lib.format.open_memmap(
            staging / "action.npy",
            mode="w+",
            dtype=np.float32,
            shape=(total_frames, action_dim),
        ),
    }

    cv2.setNumThreads(1)
    offset = 0
    episode_ends = []
    for episode_index, (path, length) in enumerate(zip(paths, lengths)):
        with h5py.File(path, "r") as source:
            vector = np.asarray(source["joint_action/vector"], dtype=np.float32)
            left = np.concatenate(
                [
                    np.asarray(source["joint_action/left_arm"], dtype=np.float32),
                    np.asarray(
                        source["joint_action/left_gripper"], dtype=np.float32
                    )[:, None],
                    np.asarray(source["joint_action/right_arm"], dtype=np.float32),
                    np.asarray(
                        source["joint_action/right_gripper"], dtype=np.float32
                    )[:, None],
                ],
                axis=1,
            )
            if vector.shape != left.shape or not np.allclose(
                vector, left, atol=1e-5, rtol=1e-5
            ):
                raise ValueError(f"Inconsistent joint representations in {path}")

            arrays["state"][offset : offset + length] = vector[:-1]
            arrays["action"][offset : offset + length] = vector[1:]
            for camera in CAMERAS:
                encoded = source[f"observation/{camera}/rgb"]
                destination = arrays[camera]
                for frame_index in range(length):
                    destination[offset + frame_index] = decode_rgb(
                        encoded[frame_index], image_size
                    )

        offset += length
        episode_ends.append(offset)
        print(
            f"{output_root.name}: episode {episode_index + 1}/"
            f"{expected_episodes}, frames={offset}/{total_frames}",
            flush=True,
        )

    if offset != total_frames:
        raise AssertionError((offset, total_frames))
    np.save(staging / "episode_ends.npy", np.asarray(episode_ends, dtype=np.int64))
    for array in arrays.values():
        array.flush()

    state = np.load(staging / "state.npy", mmap_mode="r")
    action = np.load(staging / "action.npy", mmap_mode="r")
    stats = {
        "state": {
            "min": np.min(state, axis=0).astype(float).tolist(),
            "max": np.max(state, axis=0).astype(float).tolist(),
        },
        "action": {
            "min": np.min(action, axis=0).astype(float).tolist(),
            "max": np.max(action, axis=0).astype(float).tolist(),
        },
    }
    (staging / "stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8"
    )

    marker = {
        "cache_format_version": 1,
        "partial": False,
        "source_root": str(raw_root),
        "total_episodes": expected_episodes,
        "total_frames": total_frames,
        "episode_lengths": lengths,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "image_size": image_size,
        "cameras": list(CAMERAS),
        "color_order": "RGB",
        "transition_alignment": "obs[t], action=qpos[t+1]",
    }
    marker_path.write_text(
        json.dumps(marker, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.rename(staging, output_root)
    validate_existing(output_root, expected_episodes, image_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=128)
    args = parser.parse_args()
    build_cache(
        args.raw_root.resolve(),
        args.output_root.resolve(),
        args.expected_episodes,
        args.image_size,
    )


if __name__ == "__main__":
    main()
