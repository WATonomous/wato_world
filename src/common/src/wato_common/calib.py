"""Calibration / pose helpers: 4x4 transforms, intrinsics, projection."""

from __future__ import annotations

import numpy as np


def invert_se3(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 rigid transform without re-doing matrix inversion."""
    R = T[:3, :3]
    t = T[:3, 3]
    inv = np.eye(4, dtype=T.dtype)
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv


def project_points(
    points_world: np.ndarray,
    K: np.ndarray,
    cam_T_world: np.ndarray,
) -> np.ndarray:
    """Project Nx3 world points into pixel coords. Returns Nx3 (u, v, z_cam)."""
    n = points_world.shape[0]
    homog = np.hstack([points_world, np.ones((n, 1), dtype=points_world.dtype)])
    cam = (cam_T_world @ homog.T).T[:, :3]
    z = cam[:, 2:3]
    pix = (K @ (cam / np.clip(z, 1e-6, None)).T).T
    return np.hstack([pix[:, :2], z])
