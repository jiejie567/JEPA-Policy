from __future__ import annotations

import numpy as np


def rpy_to_quaternion_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cy = np.cos(float(yaw) * 0.5)
    sy = np.sin(float(yaw) * 0.5)
    cp = np.cos(float(pitch) * 0.5)
    sp = np.sin(float(pitch) * 0.5)
    cr = np.cos(float(roll) * 0.5)
    sr = np.sin(float(roll) * 0.5)
    quat = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float32,
    )
    return normalize_quaternion_wxyz(quat)


def normalize_quaternion_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (quat / norm).astype(np.float32)


def quat_wxyz_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quaternion_wxyz(quat).astype(np.float64).reshape(4)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_rpy(rot: np.ndarray) -> np.ndarray:
    """Extract roll, pitch, yaw from a rotation matrix (intrinsic ZYX / extrinsic XYZ)."""
    rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    sy = float(np.hypot(rot[0, 0], rot[1, 0]))
    if sy > 1e-6:
        roll = float(np.arctan2(rot[2, 1], rot[2, 2]))
        pitch = float(np.arctan2(-rot[2, 0], sy))
        yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
    else:
        roll = float(np.arctan2(-rot[1, 2], rot[1, 1]))
        pitch = float(np.arctan2(-rot[2, 0], sy))
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float32)


def quat_wxyz_to_rpy(quat: np.ndarray) -> np.ndarray:
    return rotation_matrix_to_rpy(quat_wxyz_to_rotation_matrix(quat))
