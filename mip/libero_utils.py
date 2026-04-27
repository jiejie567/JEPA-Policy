"""Helpers for mapping local task aliases onto LIBERO benchmark tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from urllib.parse import unquote, urlparse


@dataclass(frozen=True)
class LiberoTaskSpec:
    alias: str
    benchmark_name: str
    task_name: str

    @property
    def dataset_filename(self) -> str:
        return f"{self.benchmark_name}/{self.task_name}_demo.hdf5"

    @property
    def bddl_file(self) -> str:
        return f"{self.task_name}.bddl"

    @property
    def init_states_file(self) -> str:
        return f"{self.task_name}.pruned_init"


LIBERO_TASK_SPECS = {
    "moka_moka": LiberoTaskSpec(
        alias="moka_moka",
        benchmark_name="libero_10",
        task_name="KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    ),
    "mug_mug": LiberoTaskSpec(
        alias="mug_mug",
        benchmark_name="libero_10",
        task_name="LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    ),
}


def is_libero_task(task_config) -> bool:
    env_type = getattr(task_config, "env_type", None)
    env_name = getattr(task_config, "env_name", None)
    return env_type == "libero" or env_name in LIBERO_TASK_SPECS


def get_libero_task_spec(task_config) -> LiberoTaskSpec:
    env_name = getattr(task_config, "env_name", None)
    if env_name in LIBERO_TASK_SPECS:
        return LIBERO_TASK_SPECS[env_name]

    benchmark_name = getattr(task_config, "libero_benchmark_name", None)
    task_name = getattr(task_config, "libero_task_name", None)
    if benchmark_name and task_name:
        return LiberoTaskSpec(
            alias=env_name or task_name.lower(),
            benchmark_name=benchmark_name,
            task_name=task_name,
        )

    raise ValueError(
        "LIBERO task requires either a known env_name alias or both "
        "libero_benchmark_name and libero_task_name."
    )


def get_libero_dataset_filename(task_config) -> str:
    spec = get_libero_task_spec(task_config)
    return spec.dataset_filename


def _discover_installed_libero_root() -> Path | None:
    try:
        dist = distribution("libero")
    except PackageNotFoundError:
        return None

    direct_url = dist.read_text("direct_url.json")
    if not direct_url:
        return None

    try:
        url = json.loads(direct_url).get("url")
    except json.JSONDecodeError:
        return None

    if not url:
        return None

    parsed = urlparse(url)
    if parsed.scheme != "file":
        return None

    return Path(unquote(parsed.path))


def get_libero_root_candidates(task_config) -> list[Path]:
    candidates = []
    libero_root = getattr(task_config, "libero_root", None)
    if libero_root:
        candidates.append(Path(libero_root).expanduser())

    installed_root = _discover_installed_libero_root()
    if installed_root is not None:
        candidates.append(installed_root)

    deduped = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved not in seen:
            deduped.append(candidate)
            seen.add(resolved)
    return deduped


def get_libero_import_paths(task_config) -> list[Path]:
    paths = []
    for root in get_libero_root_candidates(task_config):
        paths.append(root)
        paths.append(root / "libero")
    return paths


def resolve_libero_asset_dir(task_config, asset_name: str) -> Path:
    for root in get_libero_root_candidates(task_config):
        candidates = {
            "bddl_files": [
                root / "libero" / "libero" / "bddl_files",
                root / "bddl_files",
            ],
            "init_states": [
                root / "libero" / "libero" / "init_files",
                root / "init_files",
                root / "init_states",
            ],
            "datasets": [
                root / "datasets",
                root.parent / "datasets",
            ],
        }
        for candidate in candidates.get(asset_name, [root / asset_name]):
            if candidate.exists():
                return candidate
        fallback = candidates.get(asset_name, [root / asset_name])[0]
        if root == get_libero_root_candidates(task_config)[-1]:
            return fallback

    try:
        from libero.libero import get_libero_path
    except ImportError as exc:
        raise ImportError(
            "LIBERO assets are required for environment creation. Install the "
            "`libero` package or set task.libero_root to a LIBERO checkout/data root."
        ) from exc

    return Path(get_libero_path(asset_name))
