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
    quat_xyzw: np.ndarray  # shape (4,) [qx, qy, qz, qw]


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
        return make_se3(s.translation, s.quat_xyzw), float(
            abs(s.timestamp_ns - target_timestamp_ns)
        )

    timestamps = np.fromiter(
        (s.timestamp_ns for s in samples), dtype=np.int64, count=len(samples)
    )
    idx = int(np.searchsorted(timestamps, target_timestamp_ns))

    if idx <= 0:
        s = samples[0]
        return make_se3(s.translation, s.quat_xyzw), float(
            abs(s.timestamp_ns - target_timestamp_ns)
        )
    if idx >= len(samples):
        s = samples[-1]
        return make_se3(s.translation, s.quat_xyzw), float(
            abs(s.timestamp_ns - target_timestamp_ns)
        )

    s0, s1 = samples[idx - 1], samples[idx]
    span = float(s1.timestamp_ns - s0.timestamp_ns)
    if span <= 0:
        # Coincident timestamps — pick the closer one.
        s = (
            s0
            if abs(s0.timestamp_ns - target_timestamp_ns)
            <= abs(s1.timestamp_ns - target_timestamp_ns)
            else s1
        )
        return make_se3(s.translation, s.quat_xyzw), 0.0

    t = (float(target_timestamp_ns) - float(s0.timestamp_ns)) / span
    translation = (1.0 - t) * s0.translation + t * s1.translation
    quat = slerp(s0.quat_xyzw, s1.quat_xyzw, t)
    interp_error = float(
        min(
            abs(s0.timestamp_ns - target_timestamp_ns),
            abs(s1.timestamp_ns - target_timestamp_ns),
        )
    )

    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(*quat)
    T[:3, 3] = translation
    return T, interp_error


def _quats_to_matrices(quats_xyzw: np.ndarray) -> np.ndarray:
    """Vectorised xyzw quaternion -> (N, 3, 3) rotation matrices.

    Mirrors transforms.quat_to_matrix's formula but broadcast over an axis.
    Quaternions must already be unit-length (the caller is responsible).
    """
    qx = quats_xyzw[:, 0]
    qy = quats_xyzw[:, 1]
    qz = quats_xyzw[:, 2]
    qw = quats_xyzw[:, 3]
    n = qx * qx + qy * qy + qz * qz + qw * qw
    s = np.where(n < 1e-12, 0.0, 2.0 / np.where(n < 1e-12, 1.0, n))
    xx, yy, zz = qx * qx * s, qy * qy * s, qz * qz * s
    xy, xz, yz = qx * qy * s, qx * qz * s, qy * qz * s
    wx, wy, wz = qw * qx * s, qw * qy * s, qw * qz * s
    R = np.empty((qx.shape[0], 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1.0 - (yy + zz)
    R[:, 0, 1] = xy - wz
    R[:, 0, 2] = xz + wy
    R[:, 1, 0] = xy + wz
    R[:, 1, 1] = 1.0 - (xx + zz)
    R[:, 1, 2] = yz - wx
    R[:, 2, 0] = xz - wy
    R[:, 2, 1] = yz + wx
    R[:, 2, 2] = 1.0 - (xx + yy)
    # Degenerate (zero-norm) quaternions fall back to identity.
    degenerate = n < 1e-12
    if degenerate.any():
        R[degenerate] = np.eye(3)
    return R


def batch_interpolate_poses(
    samples: list[PoseSample],
    timestamps_ns: np.ndarray,
) -> np.ndarray:
    """Vectorized pose interpolation for N timestamps.

    Returns shape (N, 4, 4) float64.  One np.searchsorted call instead of N
    and one quat→matrix conversion across the whole batch, so it's efficient
    for per-point deskewing of large sweeps.  Clamps to nearest sample for
    timestamps outside the sample range.
    """
    n = len(timestamps_ns)
    if not samples:
        raise ValueError("batch_interpolate_poses: empty pose samples")

    if len(samples) == 1:
        s = samples[0]
        T = make_se3(s.translation, s.quat_xyzw)
        result = np.empty((n, 4, 4), dtype=np.float64)
        result[:] = T
        return result

    sample_ts = np.fromiter(
        (s.timestamp_ns for s in samples), dtype=np.int64, count=len(samples)
    )
    translations = np.stack([s.translation for s in samples])  # (M, 3)
    quats = np.stack([s.quat_xyzw for s in samples])  # (M, 4) xyzw

    idx = np.searchsorted(sample_ts, timestamps_ns)
    idx = np.clip(idx, 1, len(samples) - 1)

    ts0 = sample_ts[idx - 1].astype(np.float64)
    ts1 = sample_ts[idx].astype(np.float64)
    span = ts1 - ts0
    # Avoid division by zero for coincident timestamps.
    safe_span = np.where(span > 0, span, 1.0)
    t = np.clip((timestamps_ns.astype(np.float64) - ts0) / safe_span, 0.0, 1.0)
    t = np.where(span > 0, t, 0.0)

    trans = (1.0 - t[:, None]) * translations[idx - 1] + t[:, None] * translations[idx]

    q0 = quats[idx - 1]  # (N, 4)
    q1 = quats[idx]  # (N, 4)

    # Vectorised SLERP.
    q0 = q0 / np.linalg.norm(q0, axis=1, keepdims=True)
    q1 = q1 / np.linalg.norm(q1, axis=1, keepdims=True)
    dot = np.einsum("ni,ni->n", q0, q1)
    flip = dot < 0.0
    q1 = np.where(flip[:, None], -q1, q1)
    dot = np.where(flip, -dot, dot)

    linear = dot > 0.9995
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    safe_sin = np.where(linear, 1.0, sin_theta)
    a = np.where(linear, 1.0 - t, np.sin((1.0 - t) * theta) / safe_sin)
    b = np.where(linear, t, np.sin(t * theta) / safe_sin)
    q_interp = a[:, None] * q0 + b[:, None] * q1
    q_interp = q_interp / np.linalg.norm(q_interp, axis=1, keepdims=True)

    # Vectorised SE(3) assembly — no Python loop over N.
    R = _quats_to_matrices(q_interp)  # (N, 3, 3)
    result = np.zeros((n, 4, 4), dtype=np.float64)
    result[:, :3, :3] = R
    result[:, :3, 3] = trans
    result[:, 3, 3] = 1.0
    return result
