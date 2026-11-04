"""LiDAR → image projection (perception_2d depth anchors, semantic_lifting)."""

from __future__ import annotations

import numpy as np

from wato_common.geometry.camera_model import CameraModel, validate_transform


def _camera(K: np.ndarray | CameraModel) -> CameraModel:
    return K if isinstance(K, CameraModel) else CameraModel(K)


def project_with_depth(
    points_world: np.ndarray,
    K: np.ndarray | CameraModel,
    cam_T_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Unclipped pixels and optical z using full intrinsics and lens model.

    Pass CameraCalibration.model for calibrated images, or a bare K for an
    explicitly undistorted pinhole camera. Invalid points have NaN pixels.
    """
    uv, depth, _ = _camera(K).project(points_world, cam_T_world)
    return uv, depth


def project_points(
    points_world: np.ndarray,
    K: np.ndarray | CameraModel,
    cam_T_world: np.ndarray,
    image_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project Nx3 world points into a camera.

    Returns (uv, mask) where uv is Nx2 pixel coords (NaN for points behind the
    camera) and mask is the boolean array of points that landed inside the
    image (or in front of the camera if image_size is None).
    """
    pix, _, valid = _camera(K).project(points_world, cam_T_world, image_size)
    return pix, valid


def project_lidar_to_image(
    points_lidar: np.ndarray,
    K: np.ndarray | CameraModel,
    cam_T_lidar: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project N LiDAR-frame points into a camera image.

    Args:
        points_lidar: (N, 3) float — points in the LiDAR sensor frame.
        K: calibrated CameraModel, or a bare undistorted (3, 3) matrix.
        cam_T_lidar: (4, 4) SE3 — transforms LiDAR frame → camera frame.
        image_size: (width, height).

    Returns:
        uv: (N, 2) float — pixel coordinates (NaN for invalid points).
        depth_cam: (N,) float — z-depth in camera frame (0 for invalid).
        valid: (N,) bool — True for points in front of camera and inside image.
    """
    return _camera(K).project(points_lidar, cam_T_lidar, image_size)


def lift_pixel_to_world(
    u: float,
    v: float,
    depth_m: float,
    K: np.ndarray | CameraModel,
    world_T_ego: np.ndarray,
    ego_T_cam: np.ndarray,
) -> np.ndarray:
    """Back-project a single pixel at known depth into the world frame.

    Args:
        u, v: pixel coordinates.
        depth_m: metric depth (z in camera frame, metres).
        K: calibrated CameraModel, or a bare undistorted (3, 3) matrix.
        world_T_ego: (4, 4) SE3 — ego pose in world frame.
        ego_T_cam: (4, 4) SE3 — camera extrinsic (cam frame → ego frame).

    Returns:
        (3,) world-frame point.
    """
    p_cam = np.r_[_camera(K).unproject(np.array([[u, v]]), depth_m)[0], 1.0]
    p_world = (validate_transform(world_T_ego) @ validate_transform(ego_T_cam) @ p_cam)[
        :3
    ]
    return p_world
