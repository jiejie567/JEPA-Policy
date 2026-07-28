#!/usr/bin/env python3
"""Build a node-local, shared-memory RoboCasa training cache.

LeRobot's video path is optimized for storage size, not shuffled training:
every sample opens and seeks three H.264 files.  This converter decodes each
video once, applies the policy's deterministic 128x128 resize, and stores
global frame arrays as uint8 NumPy memmaps.  State and action are cached too,
so the hot training path is only clamped NumPy indexing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize


CACHE_FORMAT_VERSION = 2
MARKER_NAME = ".jepa_array_cache.json"
IMAGE_KEYS = (
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
)
ARRAY_FILENAMES = {
    "observation.images.robot0_agentview_left": "agentview_left_image.npy",
    "observation.images.robot0_agentview_right": "agentview_right_image.npy",
    "observation.images.robot0_eye_in_hand": "eye_in_hand_image.npy",
    "observation.state": "observation_state.npy",
    "action": "action.npy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--workers", type=int, default=min(32, os.cpu_count() or 1)
    )
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--decode-batch-size", type=int, default=64)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Build a partial converter test cache; it is rejected for training.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild and replace the destination even when it is current.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_episodes(source: Path) -> list[dict]:
    episodes = []
    with (source / "meta" / "episodes.jsonl").open(
        "r", encoding="utf-8"
    ) as stream:
        for line in stream:
            if line.strip():
                episodes.append(json.loads(line))
    for expected_index, episode in enumerate(episodes):
        if int(episode["episode_index"]) != expected_index:
            raise ValueError(
                "RoboCasa episodes must be in canonical order: "
                f"row={expected_index} value={episode['episode_index']}"
            )
        if int(episode["length"]) <= 0:
            raise ValueError(f"Invalid episode length: {episode}")
    return episodes


def array_specs(
    total_frames: int, height: int, width: int, info: dict
) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    state_dim = int(info["features"]["observation.state"]["shape"][0])
    action_dim = int(info["features"]["action"]["shape"][0])
    image_shape = (total_frames, 3, height, width)
    return {
        IMAGE_KEYS[0]: (image_shape, np.dtype(np.uint8)),
        IMAGE_KEYS[1]: (image_shape, np.dtype(np.uint8)),
        IMAGE_KEYS[2]: (image_shape, np.dtype(np.uint8)),
        "observation.state": (
            (total_frames, state_dim),
            np.dtype(np.float32),
        ),
        "action": ((total_frames, action_dim), np.dtype(np.float32)),
    }


def expected_marker(
    source: Path,
    info: dict,
    episodes: list[dict],
    height: int,
    width: int,
    max_episodes: int | None,
) -> dict:
    total_frames = sum(int(episode["length"]) for episode in episodes)
    partial = max_episodes is not None
    return {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "source": str(source),
        "source_info_sha256": sha256(source / "meta" / "info.json"),
        "source_episodes_sha256": sha256(
            source / "meta" / "episodes.jsonl"
        ),
        "partial": partial,
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "frame_shape_chw": [3, height, width],
        "image_dtype": "uint8",
        "state_dtype": "float32",
        "action_dtype": "float32",
        "resize": {
            "implementation": "torchvision.transforms.functional.resize",
            "interpolation": "bilinear",
            "antialias": True,
            "quantization": "round(clamp(resized_float_0_1 * 255, 0, 255))",
        },
        "arrays": {
            key: f"arrays/{filename}"
            for key, filename in ARRAY_FILENAMES.items()
        },
        "episode_ends": "arrays/episode_ends.npy",
        "source_total_episodes": int(info["total_episodes"]),
        "source_total_frames": int(info["total_frames"]),
    }


def cache_is_current(destination: Path, marker: dict, info: dict) -> bool:
    marker_path = destination / MARKER_NAME
    if not marker_path.is_file():
        return False
    try:
        actual = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if actual != marker:
        return False

    specs = array_specs(
        marker["total_frames"],
        marker["frame_shape_chw"][1],
        marker["frame_shape_chw"][2],
        info,
    )
    try:
        for key, (shape, dtype) in specs.items():
            array = np.load(destination / marker["arrays"][key], mmap_mode="r")
            if array.shape != shape or array.dtype != dtype:
                return False
        episode_ends = np.load(
            destination / marker["episode_ends"], mmap_mode="r"
        )
        if (
            episode_ends.shape != (marker["total_episodes"],)
            or episode_ends.dtype != np.int64
            or int(episode_ends[-1]) != marker["total_frames"]
        ):
            return False
    except (OSError, ValueError, KeyError):
        return False
    return True


def fixed_size_list_to_numpy(column, width: int) -> np.ndarray:
    array = column.combine_chunks()
    values = array.values.to_numpy(zero_copy_only=False)
    result = values.reshape(len(array), width)
    return result.astype(np.float32, copy=False)


def write_tabular_arrays(
    source: Path,
    info: dict,
    episodes: list[dict],
    offsets: np.ndarray,
    state_array: np.ndarray,
    action_array: np.ndarray,
) -> None:
    data_template = info["data_path"]
    chunks_size = int(info["chunks_size"])
    state_dim = state_array.shape[1]
    action_dim = action_array.shape[1]

    for episode_index, episode in enumerate(episodes):
        start = int(offsets[episode_index])
        end = int(offsets[episode_index + 1])
        expected_length = end - start
        relative = data_template.format(
            episode_chunk=episode_index // chunks_size,
            episode_index=episode_index,
        )
        path = source / relative
        table = pq.read_table(
            path,
            columns=[
                "observation.state",
                "action",
                "index",
                "episode_index",
                "frame_index",
            ],
        )
        if len(table) != expected_length:
            raise ValueError(
                f"Parquet length mismatch for episode {episode_index}: "
                f"expected={expected_length} actual={len(table)}"
            )
        global_indices = table["index"].combine_chunks().to_numpy()
        frame_indices = table["frame_index"].combine_chunks().to_numpy()
        episode_indices = (
            table["episode_index"].combine_chunks().to_numpy()
        )
        if not np.array_equal(
            global_indices, np.arange(start, end, dtype=np.int64)
        ):
            raise ValueError(
                f"Noncanonical global indices in episode {episode_index}"
            )
        if not np.array_equal(
            frame_indices, np.arange(expected_length, dtype=np.int64)
        ):
            raise ValueError(
                f"Noncanonical frame indices in episode {episode_index}"
            )
        if not np.all(episode_indices == episode_index):
            raise ValueError(
                f"Incorrect episode indices in episode {episode_index}"
            )

        state_array[start:end] = fixed_size_list_to_numpy(
            table["observation.state"], state_dim
        )
        action_array[start:end] = fixed_size_list_to_numpy(
            table["action"], action_dim
        )

        if (episode_index + 1) % 100 == 0 or (
            episode_index + 1
        ) == len(episodes):
            print(
                f"tabular_episodes={episode_index + 1}/{len(episodes)}",
                flush=True,
            )
    state_array.flush()
    action_array.flush()


def resize_and_write_batch(
    frames: list[np.ndarray],
    output: np.ndarray,
    first_index: int,
    height: int,
    width: int,
) -> int:
    if not frames:
        return first_index
    frame_array = np.stack(frames, axis=0)
    tensor = (
        torch.from_numpy(frame_array)
        .permute(0, 3, 1, 2)
        .to(dtype=torch.float32)
        .div_(255.0)
    )
    tensor = resize(
        tensor,
        [height, width],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    tensor = tensor.mul_(255.0).round_().clamp_(0, 255).to(torch.uint8)
    count = len(frames)
    output[first_index : first_index + count] = tensor.numpy()
    return first_index + count


def decode_video_into_array(
    source_video: Path,
    output: np.ndarray,
    start: int,
    expected_length: int,
    height: int,
    width: int,
    decode_batch_size: int,
) -> None:
    write_index = start
    frames: list[np.ndarray] = []
    with av.open(str(source_video), mode="r") as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) == decode_batch_size:
                write_index = resize_and_write_batch(
                    frames, output, write_index, height, width
                )
                frames.clear()
    write_index = resize_and_write_batch(
        frames, output, write_index, height, width
    )
    actual_length = write_index - start
    if actual_length != expected_length:
        raise ValueError(
            f"Video frame mismatch for {source_video}: "
            f"expected={expected_length} actual={actual_length}"
        )


def write_image_array(
    source: Path,
    info: dict,
    episodes: list[dict],
    offsets: np.ndarray,
    image_key: str,
    output: np.ndarray,
    *,
    workers: int,
    height: int,
    width: int,
    decode_batch_size: int,
) -> None:
    video_template = info["video_path"]
    chunks_size = int(info["chunks_size"])
    torch.set_num_threads(1)
    jobs = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for episode_index, episode in enumerate(episodes):
            start = int(offsets[episode_index])
            length = int(episode["length"])
            relative = video_template.format(
                episode_chunk=episode_index // chunks_size,
                video_key=image_key,
                episode_index=episode_index,
            )
            video_path = source / relative
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            jobs.append(
                executor.submit(
                    decode_video_into_array,
                    video_path,
                    output,
                    start,
                    length,
                    height,
                    width,
                    decode_batch_size,
                )
            )

        completed = 0
        for future in as_completed(jobs):
            future.result()
            completed += 1
            if completed % 50 == 0 or completed == len(jobs):
                print(
                    f"image_key={image_key} "
                    f"videos={completed}/{len(jobs)}",
                    flush=True,
                )
    output.flush()


def build_cache(
    source: Path,
    staging: Path,
    info: dict,
    episodes: list[dict],
    marker: dict,
    args: argparse.Namespace,
) -> None:
    # The optimized reader needs metadata plus generated arrays. Parquet and
    # videos remain in the immutable source dataset and would only waste
    # shared-memory capacity after their values have been materialized here.
    shutil.copytree(source / "meta", staging / "meta")

    arrays_dir = staging / "arrays"
    arrays_dir.mkdir()
    total_frames = int(marker["total_frames"])
    specs = array_specs(total_frames, args.height, args.width, info)
    arrays = {
        key: np.lib.format.open_memmap(
            arrays_dir / ARRAY_FILENAMES[key],
            mode="w+",
            dtype=dtype,
            shape=shape,
        )
        for key, (shape, dtype) in specs.items()
    }

    lengths = np.asarray(
        [int(episode["length"]) for episode in episodes], dtype=np.int64
    )
    episode_ends = np.cumsum(lengths, dtype=np.int64)
    offsets = np.concatenate(
        [np.zeros(1, dtype=np.int64), episode_ends]
    )
    np.save(arrays_dir / "episode_ends.npy", episode_ends)

    write_tabular_arrays(
        source,
        info,
        episodes,
        offsets,
        arrays["observation.state"],
        arrays["action"],
    )
    for image_key in IMAGE_KEYS:
        write_image_array(
            source,
            info,
            episodes,
            offsets,
            image_key,
            arrays[image_key],
            workers=args.workers,
            height=args.height,
            width=args.width,
            decode_batch_size=args.decode_batch_size,
        )

    for array in arrays.values():
        array.flush()
    del arrays
    (staging / MARKER_NAME).write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    if destination == source or source in destination.parents:
        raise ValueError(
            "Cache destination must not be the source dataset or below it: "
            f"source={source} destination={destination}"
        )
    if destination == Path(destination.anchor):
        raise ValueError(
            f"Refusing to use a filesystem root as destination: {destination}"
        )
    if not (source / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Invalid LeRobot dataset: {source}")
    if (
        args.workers < 1
        or args.height < 1
        or args.width < 1
        or args.decode_batch_size < 1
    ):
        raise ValueError(
            "workers, height, width and decode-batch-size must be positive"
        )

    info = json.loads(
        (source / "meta" / "info.json").read_text(encoding="utf-8")
    )
    all_episodes = load_episodes(source)
    if len(all_episodes) != int(info["total_episodes"]):
        raise ValueError(
            "Episode-count mismatch: "
            f"info={info['total_episodes']} metadata={len(all_episodes)}"
        )
    if sum(int(ep["length"]) for ep in all_episodes) != int(
        info["total_frames"]
    ):
        raise ValueError("Frame-count mismatch between info and episodes")

    if args.max_episodes is None:
        episodes = all_episodes
    else:
        if not 1 <= args.max_episodes <= len(all_episodes):
            raise ValueError("--max-episodes is outside the dataset")
        episodes = all_episodes[: args.max_episodes]

    marker = expected_marker(
        source,
        info,
        episodes,
        args.height,
        args.width,
        args.max_episodes,
    )
    if not args.force and cache_is_current(destination, marker, info):
        print(
            f"ROBOCASA_ARRAY_CACHE_READY destination={destination} "
            f"frames={marker['total_frames']} reused=1 "
            f"partial={int(marker['partial'])}"
        )
        return 0

    if destination.exists():
        if not args.force:
            raise FileExistsError(
                f"Stale or incomplete destination exists: {destination}; "
                "pass --force to replace this generated cache"
            )
        shutil.rmtree(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    try:
        print(
            "Building RoboCasa array cache: "
            f"episodes={len(episodes)} frames={marker['total_frames']} "
            f"shape={marker['frame_shape_chw']} workers={args.workers}",
            flush=True,
        )
        build_cache(source, staging, info, episodes, marker, args)
        if not cache_is_current(staging, marker, info):
            raise RuntimeError("Generated cache failed structural validation")
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(
        f"ROBOCASA_ARRAY_CACHE_READY destination={destination} "
        f"frames={marker['total_frames']} reused=0 "
        f"partial={int(marker['partial'])}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
