"""LiDAR-to-image projection and mask lookup for semantic_lifting."""

from __future__ import annotations

import numpy as np

from wato_common.geometry import invert_se3, project_lidar_to_image


def compute_cam_T_world(
    world_T_ego_frame: np.ndarray,
    ego_T_cam: np.ndarray,
) -> np.ndarray:
    """The transform that projects world-frame points into a camera.

    The sweep's points are already in the SLAM world frame (deskewed per
    point at their own timestamps upstream), so the only pose that matters is
    the camera's world pose at the frame time:

        cam_T_world = inv(ego_T_cam) @ inv(world_T_ego_frame)

    Motion of *dynamic* objects between a point's time and the frame's is not
    compensated — see "Dynamic-point handling" in the design doc.
    """
    return invert_se3(ego_T_cam) @ invert_se3(world_T_ego_frame)


def project_and_clip(
    points_world: np.ndarray,
    K: np.ndarray,
    cam_T_world: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world-frame points; returns pixel coords, depths, valid mask.

    Args:
        points_world: (N, 3) float64 points in the SLAM world frame.
        K: (3, 3) intrinsic matrix.
        cam_T_world: (4, 4) SE3 transform from world to camera.
        image_size: (W, H) in pixels.

    Returns:
        uv: (N, 2) float32 pixel coordinates (may be outside image for invalid).
        depth_cam: (N,) float32 camera-frame depths.
        valid: (N,) bool — True for points that project inside the image with
               positive depth.
    """
    return project_lidar_to_image(points_world, K, cam_T_world, image_size)


def pixel_indices(
    uv: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest pixel (column, row) of each projected point, clipped to the image."""
    ui = np.clip(np.round(uv[:, 0]).astype(np.int64), 0, width - 1)
    vi = np.clip(np.round(uv[:, 1]).astype(np.int64), 0, height - 1)
    return ui, vi


def label_raster(masks: list[np.ndarray], labels: list[int]) -> np.ndarray:
    """One frame's masks as an (H, W) int32 raster of label, -1 = no mask.

    Where masks overlap the smallest one wins (the innermost object: a
    pedestrian's mask inside a bus's); at equal area, the earlier mask.
    """
    if not masks:
        raise ValueError("label_raster needs at least one mask")
    raster = np.full(masks[0].shape, -1, dtype=np.int32)
    areas = [int(m.sum()) for m in masks]
    # Paint largest first so smaller masks overwrite; reversed index order
    # within one area lets the earlier mask paint last.
    for k in sorted(range(len(masks)), key=lambda k: (-areas[k], -k)):
        raster[masks[k]] = labels[k]
    return raster
