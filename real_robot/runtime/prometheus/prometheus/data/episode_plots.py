from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np

# Break plot lines when samples are missing for longer than this (e.g. recorder.pause()).
DEFAULT_PLOT_GAP_THRESHOLD_S = 0.2
PLOT_SUBPROCESS_TIMEOUT_S = 120.0


def time_gaps(
    time_s: np.ndarray,
    *,
    gap_threshold_s: float = DEFAULT_PLOT_GAP_THRESHOLD_S,
) -> list[tuple[float, float]]:
    gaps: list[tuple[float, float]] = []
    if time_s.size < 2:
        return gaps
    for index in range(int(time_s.size) - 1):
        start = float(time_s[index])
        end = float(time_s[index + 1])
        if end - start > float(gap_threshold_s):
            gaps.append((start, end))
    return gaps


def insert_nan_at_time_gaps(
    time_s: np.ndarray,
    values: np.ndarray,
    *,
    gap_threshold_s: float = DEFAULT_PLOT_GAP_THRESHOLD_S,
) -> tuple[np.ndarray, np.ndarray]:
    if time_s.size <= 1:
        return time_s, values
    dt = np.diff(time_s)
    gap_indices = np.flatnonzero(dt > float(gap_threshold_s))
    if gap_indices.size == 0:
        return time_s, values

    expanded_size = int(time_s.size) + int(gap_indices.size)
    plot_t = np.full(expanded_size, np.nan, dtype=np.float64)
    if values.ndim == 1:
        plot_v = np.full(expanded_size, np.nan, dtype=np.float64)
    else:
        plot_v = np.full((expanded_size, values.shape[1]), np.nan, dtype=values.dtype)

    write_index = 0
    for read_index in range(int(time_s.size)):
        plot_t[write_index] = time_s[read_index]
        plot_v[write_index] = values[read_index]
        write_index += 1
        if read_index in gap_indices:
            write_index += 1
    return plot_t, plot_v


def _shade_time_gaps(
    axes: list[Any] | Any,
    gaps: list[tuple[float, float]],
    plt: Any,
    *,
    labeled: bool = False,
) -> None:
    if not gaps:
        return
    if isinstance(axes, np.ndarray):
        axes = axes.reshape(-1).tolist()
    elif isinstance(axes, tuple):
        axes = list(axes)
    elif not isinstance(axes, list):
        axes = [axes]
    label = "recording gap" if labeled else None
    for start, end in gaps:
        for ax in axes:
            ax.axvspan(
                start,
                end,
                color="#bdbdbd",
                alpha=0.35,
                linewidth=0,
                label=label,
            )
        label = None


def series_to_arrays(
    series: dict[str, Any],
    value_key: str,
    *,
    expected_dim: int | None = None,
    fallback_hz: float = 120.0,
) -> tuple[np.ndarray, np.ndarray]:
    values = list(series.get(value_key, []))
    if not values:
        raise ValueError(f"series has no {value_key} data")
    valid_rows = [np.asarray(row, dtype=np.float64).ravel() for row in values if row is not None]
    if not valid_rows and expected_dim is None:
        raise ValueError(f"series has no {value_key} data")
    row_dim = int(expected_dim or valid_rows[0].size)
    rows = []
    for value in values:
        if value is None:
            rows.append(np.full(row_dim, np.nan, dtype=np.float64))
            continue
        row = np.asarray(value, dtype=np.float64).ravel()
        if row.size != row_dim:
            raise ValueError(f"{value_key} row has dimension {row.size}, expected {row_dim}")
        rows.append(row)
    data = np.stack(rows, axis=0)
    timestamps = np.asarray(series.get("timestamps", []), dtype=np.float64)
    if timestamps.shape[0] != data.shape[0]:
        timestamps = np.arange(data.shape[0], dtype=np.float64) * (1000.0 / max(1.0, fallback_hz))
    time_s = (timestamps - timestamps[0]) / 1000.0 if timestamps.size else np.arange(data.shape[0]) / fallback_hz
    return data, time_s


def plot_episode_state(
    episode_dir: str | Path,
    robot_state: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    dpi: int = 220,
    joint_plot_ranges: list[list[float]] | None = None,
) -> list[str]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/prometheus_matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/prometheus_cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    plt = _import_pyplot()

    episode_dir = Path(episode_dir)
    output_dir = Path(output_dir) if output_dir is not None else episode_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        plot_joint_state(
            output_dir,
            robot_state,
            plt,
            dpi=dpi,
            episode_name=episode_dir.name,
            joint_plot_ranges=joint_plot_ranges,
        ),
        plot_eef_xyz(
            output_dir,
            robot_state,
            plt,
            dpi=dpi,
            episode_name=episode_dir.name,
        ),
    ]
    return [str(path) for path in outputs]


def plot_episode_state_isolated(
    episode_dir: str | Path,
    robot_state_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    dpi: int = 220,
    joint_plot_ranges: list[list[float]] | None = None,
    timeout_s: float = PLOT_SUBPROCESS_TIMEOUT_S,
) -> list[str]:
    """Render episode plots in a fresh process with the Python env C++ runtime.

    ROS and hardware SDKs may load the system libstdc++ into the workflow process
    before matplotlib is imported. A fresh interpreter lets matplotlib use the
    newer libstdc++ shipped with the active Conda environment without changing
    the hardware process dynamic-library state.
    """
    episode_dir = Path(episode_dir).expanduser().resolve()
    robot_state_path = Path(robot_state_path).expanduser().resolve()
    resolved_output_dir = (
        episode_dir if output_dir is None else Path(output_dir).expanduser().resolve()
    )
    if not robot_state_path.is_file():
        raise FileNotFoundError(f"robot state file not found: {robot_state_path}")
    if float(timeout_s) <= 0:
        raise ValueError("plot subprocess timeout_s must be positive")

    command = [
        sys.executable,
        "-m",
        "prometheus.data.episode_plots",
        "--episode-dir",
        str(episode_dir),
        "--robot-state-path",
        str(robot_state_path),
        "--output-dir",
        str(resolved_output_dir),
        "--dpi",
        str(int(dpi)),
    ]
    if joint_plot_ranges is not None:
        command.extend(
            ["--joint-plot-ranges-json", json.dumps(joint_plot_ranges)]
        )

    result = subprocess.run(
        command,
        env=_plot_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=float(timeout_s),
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            f"episode plot subprocess failed with returncode {result.returncode}: "
            f"{detail or 'no output'}"
        )

    outputs = [
        resolved_output_dir / "state_joint_vis.png",
        resolved_output_dir / "state_eef_xyz_vis.png",
    ]
    missing = [str(path) for path in outputs if not path.is_file()]
    if missing:
        raise RuntimeError(
            "episode plot subprocess completed without expected outputs: "
            + ", ".join(missing)
        )
    return [str(path) for path in outputs]


def _plot_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    python_lib = Path(sys.prefix) / "lib"
    if (python_lib / "libstdc++.so.6").is_file():
        current = env.get("LD_LIBRARY_PATH", "")
        entries = [str(python_lib)]
        entries.extend(item for item in current.split(os.pathsep) if item)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(entries))

    project_root = str(Path(__file__).resolve().parents[2])
    current_pythonpath = env.get("PYTHONPATH", "")
    python_entries = [project_root]
    python_entries.extend(item for item in current_pythonpath.split(os.pathsep) if item)
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(python_entries))
    env.setdefault("MPLCONFIGDIR", "/tmp/prometheus_matplotlib")
    env.setdefault("XDG_CACHE_HOME", "/tmp/prometheus_cache")
    return env


def _import_pyplot() -> Any:
    try:
        return _import_pyplot_once()
    except (ImportError, ValueError, EOFError) as exc:
        if "bad marshal data" not in str(exc):
            raise
        _clear_python_caches()
        return _import_pyplot_once()


def _import_pyplot_once() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _clear_python_caches() -> None:
    roots = {
        Path(__file__).resolve().parents[2],
        Path(os.environ.get("MPLCONFIGDIR", "/tmp/prometheus_matplotlib")),
        Path(os.environ.get("XDG_CACHE_HOME", "/tmp/prometheus_cache")),
    }
    for module_name in ("matplotlib", "numpy", "PIL"):
        spec = find_spec(module_name)
        locations = list(spec.submodule_search_locations or []) if spec else []
        if spec and spec.origin:
            locations.append(str(Path(spec.origin).parent))
        roots.update(Path(location).resolve() for location in locations)
    for root in roots:
        if not root.exists():
            continue
        for module_name in tuple(sys.modules):
            if module_name == "matplotlib" or module_name.startswith("matplotlib."):
                sys.modules.pop(module_name, None)
        for cache_dir in root.rglob("__pycache__"):
            shutil.rmtree(cache_dir, ignore_errors=True)


def plot_joint_state(
    output_dir: Path,
    robot_state: dict[str, Any],
    plt: Any,
    *,
    dpi: int,
    episode_name: str,
    joint_plot_ranges: list[list[float]] | None = None,
) -> Path:
    left_qpos, left_t_s = series_to_arrays(robot_state["left"], "joint", expected_dim=7)
    right_qpos, right_t_s = series_to_arrays(robot_state["right"], "joint", expected_dim=7)
    sample_count = min(len(left_qpos), len(right_qpos))
    qpos = np.concatenate((left_qpos[:sample_count], right_qpos[:sample_count]), axis=1)
    t_s = left_t_s[:sample_count] if len(left_t_s) <= len(right_t_s) else right_t_s[:sample_count]
    names = (
        [f"left_joint_{idx + 1}" for idx in range(6)]
        + ["left_gripper"]
        + [f"right_joint_{idx + 1}" for idx in range(6)]
        + ["right_gripper"]
    )
    ranges = normalize_joint_plot_ranges(joint_plot_ranges)
    plot_t, plot_qpos = insert_nan_at_time_gaps(t_s, qpos)
    gaps = time_gaps(t_s)

    n_joints = qpos.shape[1]
    fig_h = max(8.0, 1.25 * n_joints)
    fig, axes = plt.subplots(n_joints, 1, figsize=(18, fig_h), sharex=True)
    if n_joints == 1:
        axes = [axes]
    _shade_time_gaps(axes, gaps, plt, labeled=True)
    for idx, ax in enumerate(axes):
        ax.plot(plot_t, plot_qpos[:, idx], color="#1f77b4", linewidth=0.85)
        ax.set_ylim(*ranges[idx % 7])
        ax.set_ylabel(names[idx], fontsize=8)
        ax.grid(True, alpha=0.35, linewidth=0.5)
        ax.tick_params(labelsize=8)
    axes[-1].set_xlabel("time (s)", fontsize=10)
    duration = t_s[-1] - t_s[0] if t_s.size > 1 else 0.0
    gap_summary = _gap_summary(gaps)
    fig.suptitle(
        f"Prometheus joint state - {episode_name}\n"
        f"N={qpos.shape[0]} duration={duration:.2f}s dim={qpos.shape[1]}{gap_summary}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = output_dir / "state_joint_vis.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def normalize_joint_plot_ranges(ranges: list[list[float]] | None) -> list[tuple[float, float]]:
    source = ranges or [
        [-3.14, 2.618],
        [-0.05, 3.50],
        [-0.10, 3.20],
        [-1.60, 1.55],
        [-1.57, 1.57],
        [-2.00, 2.00],
        [-0.005, 0.09],
    ]
    if len(source) != 7:
        raise ValueError("joint_plot_ranges must contain seven ranges")
    normalized = []
    for index, item in enumerate(source):
        if len(item) != 2:
            raise ValueError(f"joint_plot_ranges[{index}] must contain [min,max]")
        lower, upper = float(item[0]), float(item[1])
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError(f"joint_plot_ranges[{index}] requires finite min < max")
        normalized.append((lower, upper))
    return normalized


def plot_eef_xyz(
    output_dir: Path,
    robot_state: dict[str, Any],
    plt: Any,
    *,
    dpi: int,
    episode_name: str,
) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(18, 8), sharex=True)
    labels = ("x", "y", "z")
    colors = {"left": "#d62728", "right": "#1f77b4"}
    any_data = False
    all_gaps: list[tuple[float, float]] = []
    for side in ("left", "right"):
        series = robot_state.get(side) if isinstance(robot_state, dict) else None
        if not isinstance(series, dict) or not series.get("eef"):
            continue
        pose, t_s = series_to_arrays(series, "eef", expected_dim=7)
        if pose.shape[1] < 3 or not np.isfinite(pose[:, :3]).any():
            continue
        any_data = True
        all_gaps.extend(time_gaps(t_s))
        plot_t, plot_pose = insert_nan_at_time_gaps(t_s, pose)
        for idx, ax in enumerate(axes):
            ax.plot(plot_t, plot_pose[:, idx], color=colors[side], linewidth=0.9, label=side)
    if not any_data:
        plt.close(fig)
        raise ValueError("robot_state has no eef xyz data")

    merged_gaps = _merge_time_gaps(all_gaps)
    _shade_time_gaps(axes, merged_gaps, plt, labeled=True)
    for idx, ax in enumerate(axes):
        ax.set_ylabel(f"eef {labels[idx]} (m)", fontsize=10)
        ax.grid(True, alpha=0.35, linewidth=0.5)
        ax.tick_params(labelsize=9)
        if idx == 0:
            ax.legend(loc="upper right")
    axes[-1].set_xlabel("time (s)", fontsize=10)
    gap_summary = _gap_summary(merged_gaps)
    fig.suptitle(f"Prometheus EEF xyz - {episode_name}{gap_summary}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = output_dir / "state_eef_xyz_vis.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def _merge_time_gaps(gaps: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not gaps:
        return []
    ordered = sorted(gaps, key=lambda item: item[0])
    merged: list[tuple[float, float]] = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
            continue
        merged.append((start, end))
    return merged


def _gap_summary(gaps: list[tuple[float, float]]) -> str:
    if not gaps:
        return ""
    total = sum(end - start for start, end in gaps)
    return f" | gaps={len(gaps)} missing={total:.2f}s"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Prometheus episode state plots")
    parser.add_argument("--episode-dir", required=True)
    parser.add_argument("--robot-state-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--joint-plot-ranges-json")
    args = parser.parse_args()

    with Path(args.robot_state_path).open("rb") as stream:
        robot_state = pickle.load(stream)
    joint_plot_ranges = (
        None
        if args.joint_plot_ranges_json is None
        else json.loads(args.joint_plot_ranges_json)
    )
    outputs = plot_episode_state(
        args.episode_dir,
        robot_state,
        output_dir=args.output_dir,
        dpi=args.dpi,
        joint_plot_ranges=joint_plot_ranges,
    )
    print(json.dumps({"outputs": outputs}, ensure_ascii=True))


if __name__ == "__main__":
    main()
