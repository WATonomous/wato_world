"""SE(3) helpers and quaternion <-> rotation-matrix conversions.

Quaternion convention: (qx, qy, qz, qw) — Hamilton, x/y/z first, w last.  This
matches ROS 2 (geometry_msgs/Quaternion) and `scipy.spatial.transform.Rotation`.
"""

from __future__ import annotations

import numpy as np


def quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert a unit quaternion (xyzw) to a 3x3 rotation matrix."""
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = qx * qx * s, qy * qy * s, qz * qz * s
    xy, xz, yz = qx * qy * s, qx * qz * s, qy * qz * s
    wx, wy, wz = qw * qx * s, qw * qy * s, qw * qz * s
    return np.array([
        [1.0 - (yy + zz), xy - wz,         xz + wy],
        [xy + wz,         1.0 - (xx + zz), yz - wx],
        [xz - wy,         yz + wx,         1.0 - (xx + yy)],
    ])


def matrix_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """3x3 rotation matrix → (qx, qy, qz, qw)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return float(qx), float(qy), float(qz), float(qw)


def make_se3(translation: np.ndarray, quat_xyzw: np.ndarray | tuple[float, ...]) -> np.ndarray:
    """Compose a 4x4 transform from a 3-vector and an xyzw quaternion."""
    qx, qy, qz, qw = quat_xyzw
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(qx, qy, qz, qw)
    T[:3, 3] = translation
    return T


def split_se3(T: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Inverse of make_se3.  Returns (translation, (qx, qy, qz, qw))."""
    return T[:3, 3].copy(), matrix_to_quat(T[:3, :3])


def invert_se3(T: np.ndarray) -> np.ndarray:
    """Invert a rigid transform without going through np.linalg.inv."""
    inv = np.eye(4, dtype=T.dtype)
    R = T[:3, :3]
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ T[:3, 3]
    return inv


def flatten_se3(T: np.ndarray) -> list[float]:
    """Row-major flatten for parquet storage."""
    return T.flatten().tolist()


def unflatten_se3(flat: list[float]) -> np.ndarray:
    return np.asarray(flat, dtype=np.float64).reshape(4, 4)
