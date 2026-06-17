"""LiDAR-to-image projection for semantic_lifting.

Thin wrapper around wato_common.geometry.project_lidar_to_image that
also handles mask-in-image point assignment.
"""

from __future__ import annotations

import numpy as np

from wato_common.geometry import project_lidar_to_image


def project_and_clip(
    points_lidar: np.ndarray,
    K: np.ndarray,
    cam_T_lidar: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project LiDAR points and return pixel coords, depths, and valid mask.

    Args:
        points_lidar: (N, 3) float64 points in LiDAR frame.
        K: (3, 3) intrinsic matrix.
        cam_T_lidar: (4, 4) SE3 transform from LiDAR to camera.
        image_size: (W, H) in pixels.

    Returns:
        uv: (N, 2) float32 pixel coordinates (may be outside image for invalid).
        depth_cam: (N,) float32 camera-frame depths.
        valid: (N,) bool — True for points that project inside the image with
               positive depth.
    """
    uv, depth_cam, valid = project_lidar_to_image(
        points_lidar, K, cam_T_lidar, image_size
    )
    return uv, depth_cam, valid


def assign_masks_to_points(
    uv_valid: np.ndarray,
    masks: list[np.ndarray],
) -> np.ndarray:
    """For each visible point, find which mask it falls inside (or -1).

    When a point falls inside multiple masks, the smallest-area mask wins
    (innermost-object heuristic per design doc).

    Args:
        uv_valid: (M, 2) float32 pixel coords of visible points.
        masks: list of (H, W) bool arrays, one per detection.

    Returns:
        (M,) int32 index into `masks`, or -1 if no mask contains the point.
    """
    M = uv_valid.shape[0]
    assignment = np.full(M, -1, dtype=np.int32)
    if not masks or M == 0:
        return assignment

    # Precompute mask areas.
    areas = np.array([float(m.sum()) for m in masks], dtype=np.float32)

    ui = np.clip(np.round(uv_valid[:, 0]).astype(int), 0, masks[0].shape[1] - 1)
    vi = np.clip(np.round(uv_valid[:, 1]).astype(int), 0, masks[0].shape[0] - 1)

    best_area = np.full(M, np.inf, dtype=np.float32)
    for k, mask in enumerate(masks):
        inside = mask[vi, ui]
        better = inside & (areas[k] < best_area)
        assignment[better] = k
        best_area[better] = areas[k]

    return assignment
