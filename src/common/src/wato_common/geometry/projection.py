"""3D <-> 2D projection used by proposal generation, label refinement, and ingest checks."""

from __future__ import annotations

import numpy as np


def project_points(
    points_world: np.ndarray,
    K: np.ndarray,
    cam_T_world: np.ndarray,
    image_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project Nx3 world points into a camera.

    Returns (uv, mask) where uv is Nx2 pixel coords (NaN for points behind the
    camera) and mask is the boolean array of points that landed inside the
    image (or in front of the camera if image_size is None).
    """
    n = points_world.shape[0]
    if n == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=bool)

    homog = np.hstack([points_world, np.ones((n, 1), dtype=points_world.dtype)])
    cam = (cam_T_world @ homog.T).T[:, :3]
    z = cam[:, 2]
    in_front = z > 1e-6

    pix = np.full((n, 2), np.nan, dtype=np.float64)
    valid = in_front.copy()
    if np.any(in_front):
        proj = (K @ (cam[in_front] / z[in_front, None]).T).T
        pix[in_front] = proj[:, :2]

    if image_size is not None:
        w, h = image_size
        valid &= (
            (pix[:, 0] >= 0) & (pix[:, 0] < w) & (pix[:, 1] >= 0) & (pix[:, 1] < h)
        )

    return pix, valid
