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
        valid &= (pix[:, 0] >= 0) & (pix[:, 0] < w) & (pix[:, 1] >= 0) & (pix[:, 1] < h)

    return pix, valid


def project_lidar_to_image(
    points_lidar: np.ndarray,
    K: np.ndarray,
    cam_T_lidar: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project N LiDAR-frame points into a camera image.

    Args:
        points_lidar: (N, 3) float — points in the LiDAR sensor frame.
        K: (3, 3) intrinsic matrix.
        cam_T_lidar: (4, 4) SE3 — transforms LiDAR frame → camera frame.
        image_size: (width, height).

    Returns:
        uv: (N, 2) float — pixel coordinates (NaN for invalid points).
        depth_cam: (N,) float — z-depth in camera frame (0 for invalid).
        valid: (N,) bool — True for points in front of camera and inside image.
    """
    n = points_lidar.shape[0]
    if n == 0:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0, dtype=bool)

    homog = np.hstack([points_lidar, np.ones((n, 1), dtype=points_lidar.dtype)])
    cam_pts = (cam_T_lidar @ homog.T).T[:, :3]
    z = cam_pts[:, 2]
    in_front = z > 1e-6

    uv = np.full((n, 2), np.nan, dtype=np.float64)
    depth_cam = np.zeros(n, dtype=np.float64)
    valid = in_front.copy()

    if np.any(in_front):
        proj = (K @ (cam_pts[in_front] / z[in_front, None]).T).T
        uv[in_front] = proj[:, :2]
        depth_cam[in_front] = z[in_front]

    w, h = image_size
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)

    return uv, depth_cam, valid


def lift_pixel_to_world(
    u: float,
    v: float,
    depth_m: float,
    K: np.ndarray,
    world_T_ego: np.ndarray,
    ego_T_cam: np.ndarray,
) -> np.ndarray:
    """Back-project a single pixel at known depth into the world frame.

    Args:
        u, v: pixel coordinates.
        depth_m: metric depth (z in camera frame, metres).
        K: (3, 3) intrinsic matrix.
        world_T_ego: (4, 4) SE3 — ego pose in world frame.
        ego_T_cam: (4, 4) SE3 — camera extrinsic (cam frame → ego frame).

    Returns:
        (3,) world-frame point.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (u - cx) * depth_m / fx
    y_cam = (v - cy) * depth_m / fy
    z_cam = depth_m
    p_cam = np.array([x_cam, y_cam, z_cam, 1.0], dtype=np.float64)
    p_world = (world_T_ego @ ego_T_cam @ p_cam)[:3]
    return p_world
