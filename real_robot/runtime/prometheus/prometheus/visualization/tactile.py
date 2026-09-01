from __future__ import annotations

from typing import Any

import numpy as np

from prometheus.data.raw_episode import pointcloud_to_array

TACTILE_GRID_HEIGHT = 35
TACTILE_GRID_WIDTH = 20
TACTILE_CHANNELS = 6
DZ_CHANNEL = 5
DEFAULT_PERCENTILE_LOW = 1.0
DEFAULT_PERCENTILE_HIGH = 99.0


def _jet_bgr_lut() -> np.ndarray:
    values = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    red = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    return np.column_stack(
        [
            (blue * 255.0).astype(np.uint8),
            (green * 255.0).astype(np.uint8),
            (red * 255.0).astype(np.uint8),
        ]
    )


_JET_BGR_LUT = _jet_bgr_lut()


def pointcloud_msg_to_dz_grid(msg: Any) -> np.ndarray:
    flat = pointcloud_to_array(msg)
    return flat.reshape(TACTILE_GRID_HEIGHT, TACTILE_GRID_WIDTH, TACTILE_CHANNELS)[:, :, DZ_CHANNEL]


def normalize_grid_to_u8(
    grid: np.ndarray,
    *,
    percentile_low: float = DEFAULT_PERCENTILE_LOW,
    percentile_high: float = DEFAULT_PERCENTILE_HIGH,
) -> np.ndarray:
    grid_f = np.asarray(grid, dtype=np.float32)
    finite = np.isfinite(grid_f)
    if not np.any(finite):
        return np.zeros(grid_f.shape, dtype=np.uint8)

    valid = grid_f[finite]
    lo = float(np.percentile(valid, percentile_low))
    hi = float(np.percentile(valid, percentile_high))
    if hi <= lo + 1e-8:
        return np.zeros(grid_f.shape, dtype=np.uint8)

    clipped = np.clip(grid_f, lo, hi)
    normalized = (clipped - lo) / (hi - lo)
    normalized[~finite] = 0.0
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def dz_grid_to_bgr(
    grid: np.ndarray,
    *,
    percentile_low: float = DEFAULT_PERCENTILE_LOW,
    percentile_high: float = DEFAULT_PERCENTILE_HIGH,
) -> np.ndarray:
    grid_u8 = normalize_grid_to_u8(
        grid,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
    )
    return _JET_BGR_LUT[grid_u8]


def pointcloud_msg_to_dz_bgr(
    msg: Any,
    *,
    percentile_low: float = DEFAULT_PERCENTILE_LOW,
    percentile_high: float = DEFAULT_PERCENTILE_HIGH,
) -> np.ndarray:
    return dz_grid_to_bgr(
        pointcloud_msg_to_dz_grid(msg),
        percentile_low=percentile_low,
        percentile_high=percentile_high,
    )


def is_tactile_flow_stream(name: str) -> bool:
    return str(name).endswith("_tactile_flow")