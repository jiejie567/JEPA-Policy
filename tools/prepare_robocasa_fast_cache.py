#!/usr/bin/env python3
"""Build a node-local RoboCasa cache optimized for random frame access.

The published RoboCasa LeRobot videos use a long GOP (typically 250 frames).
PyAV must decode from the preceding keyframe for every randomly sampled
training window.  This tool copies the non-video dataset files and transcodes
the videos to a short GOP without changing their frame rate, frame count, or
resolution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


CACHE_FORMAT_VERSION = 1
MARKER_NAME = ".jepa_fast_cache.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--gop", type=int, default=8)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_ffmpeg() -> str:
    override = os.environ.get("ROBOCASA_FFMPEG")
    if override:
        return override
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(
            "FFmpeg is unavailable; install imageio-ffmpeg or set ROBOCASA_FFMPEG"
        ) from exc


def expected_marker(
    source: Path, video_count: int, gop: int, crf: int, preset: str
) -> dict:
    return {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "source": str(source),
        "source_info_sha256": sha256(source / "meta" / "info.json"),
        "video_count": video_count,
        "gop": gop,
        "crf": crf,
        "preset": preset,
    }


def cache_is_current(destination: Path, marker: dict) -> bool:
    marker_path = destination / MARKER_NAME
    if not marker_path.is_file():
        return False
    try:
        actual = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if actual != marker:
        return False
    return (
        sum(1 for _ in (destination / "videos").rglob("*.mp4"))
        == marker["video_count"]
    )


def transcode_one(
    ffmpeg: str,
    source_root: Path,
    destination_root: Path,
    source_video: Path,
    gop: int,
    crf: int,
    preset: str,
) -> None:
    relative = source_video.relative_to(source_root)
    destination_video = destination_root / relative
    destination_video.parent.mkdir(parents=True, exist_ok=True)
    partial = destination_video.with_suffix(".partial.mp4")
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-g",
        str(gop),
        "-keyint_min",
        str(gop),
        "-sc_threshold",
        "0",
        "-threads",
        "1",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(partial),
    ]
    subprocess.run(command, check=True)
    if not partial.is_file() or partial.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg produced an empty file: {partial}")
    partial.replace(destination_video)


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    if destination == source or source in destination.parents:
        raise ValueError(
            "Cache destination must not be the source dataset or live below it: "
            f"source={source} destination={destination}"
        )
    if destination == Path(destination.anchor):
        raise ValueError(f"Refusing to use a filesystem root as destination: {destination}")
    if not (source / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Invalid LeRobot dataset: {source}")
    if args.workers < 1 or args.gop < 1:
        raise ValueError("--workers and --gop must be positive")

    videos = sorted((source / "videos").rglob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No MP4 videos found below {source / 'videos'}")
    marker = expected_marker(
        source, len(videos), args.gop, args.crf, args.preset
    )
    if cache_is_current(destination, marker):
        print(
            f"ROBOCASA_FAST_CACHE_READY destination={destination} "
            f"videos={len(videos)} reused=1"
        )
        return 0

    if destination.exists():
        if not args.force:
            raise FileExistsError(
                f"Stale or incomplete destination exists: {destination}; "
                "pass --force to replace this node-local cache"
            )
        shutil.rmtree(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-", dir=destination.parent
        )
    )
    try:
        for child in source.iterdir():
            if child.name == "videos":
                continue
            target = staging / child.name
            if child.is_dir():
                shutil.copytree(child, target)
            else:
                shutil.copy2(child, target)

        ffmpeg = get_ffmpeg()
        print(
            f"Transcoding {len(videos)} videos with workers={args.workers}, "
            f"gop={args.gop}, crf={args.crf}"
        )
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    transcode_one,
                    ffmpeg,
                    source,
                    staging,
                    video,
                    args.gop,
                    args.crf,
                    args.preset,
                )
                for video in videos
            ]
            completed = 0
            for future in as_completed(futures):
                future.result()
                completed += 1
                if completed % 100 == 0 or completed == len(videos):
                    print(f"transcoded={completed}/{len(videos)}", flush=True)

        output_count = sum(1 for _ in (staging / "videos").rglob("*.mp4"))
        if output_count != len(videos):
            raise RuntimeError(
                f"Video count mismatch: expected={len(videos)} actual={output_count}"
            )
        (staging / MARKER_NAME).write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(
        f"ROBOCASA_FAST_CACHE_READY destination={destination} "
        f"videos={len(videos)} reused=0"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
