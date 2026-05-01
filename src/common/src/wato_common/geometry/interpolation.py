"""Pose interpolation: linear for translation, SLERP for orientation.

Used by the frame-index builder to estimate world_T_ego at LiDAR sweep
timestamps that fall between actual pose samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wato_common.geometry.transforms import make_se3, quat_to_matrix


@dataclass(frozen=True)
class PoseSample:
    timestamp_ns: int
    translation: np.ndarray  # shape (3,)
    quat_xyzw: np.ndarray    # shape (4,) [qx, qy, qz, qw]


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Quaternion SLERP, xyzw convention.  t in [0, 1]."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))

    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        # Quaternions are nearly parallel — fall back to linear + renormalize.
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    a = np.sin((1.0 - t) * theta) / sin_theta
    b = np.sin(t * theta) / sin_theta
    return a * q0 + b * q1


def interpolate_pose(
    samples: list[PoseSample],
    target_timestamp_ns: int,
) -> tuple[np.ndarray, float]:
    """Estimate world_T_ego at `target_timestamp_ns`.

    `samples` must be sorted by timestamp.  Returns (4x4, interp_error_ns)
    where interp_error_ns is the gap to the nearest sample (a sanity metric:
    if it's huge, the pose stream had a hole and you should not trust the
    interpolation).
    """
    if not samples:
        raise ValueError("interpolate_pose: empty pose samples")
    if len(samples) == 1:
        s = samples[0]
        return make_se3(s.translation, s.quat_xyzw), float(abs(s.timestamp_ns - target_timestamp_ns))

    timestamps = np.fromiter((s.timestamp_ns for s in samples), dtype=np.int64, count=len(samples))
    idx = int(np.searchsorted(timestamps, target_timestamp_ns))

    if idx <= 0:
        s = samples[0]
        return make_se3(s.translation, s.quat_xyzw), float(abs(s.timestamp_ns - target_timestamp_ns))
    if idx >= len(samples):
        s = samples[-1]
        return make_se3(s.translation, s.quat_xyzw), float(abs(s.timestamp_ns - target_timestamp_ns))

    s0, s1 = samples[idx - 1], samples[idx]
    span = float(s1.timestamp_ns - s0.timestamp_ns)
    if span <= 0:
        # Coincident timestamps — pick the closer one.
        s = s0 if abs(s0.timestamp_ns - target_timestamp_ns) <= abs(s1.timestamp_ns - target_timestamp_ns) else s1
        return make_se3(s.translation, s.quat_xyzw), 0.0

    t = (float(target_timestamp_ns) - float(s0.timestamp_ns)) / span
    translation = (1.0 - t) * s0.translation + t * s1.translation
    quat = slerp(s0.quat_xyzw, s1.quat_xyzw, t)
    interp_error = float(min(abs(s0.timestamp_ns - target_timestamp_ns),
                             abs(s1.timestamp_ns - target_timestamp_ns)))

    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(*quat)
    T[:3, 3] = translation
    return T, interp_error
