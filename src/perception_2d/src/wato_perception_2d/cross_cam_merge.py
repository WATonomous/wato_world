"""Cross-camera object identity merging.

Groups masklets from different cameras that observe the same 3D object by
back-projecting each masklet's mask centroid into world frame (using the
depth from the nearest LiDAR return inside the mask) and clustering by 3D
proximity.  Assigns a shared global_object_id to matched masklets.

This is the "x-cam merge" step described in the perception_2d component
docstring and is the pure-geometry foundation for cross-modal reasoning.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from wato_common.geometry import project_points, invert_se3, unflatten_se3
from wato_perception_2d.io import CalibrationInfo
from wato_perception_2d.tracker_2d import Masklet

log = logging.getLogger(__name__)


def _mask_centroid_px(mask: np.ndarray) -> Optional[np.ndarray]:
    """Return (2,) float32 pixel [u, v] centroid of a binary mask, or None."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.mean(), ys.mean()], dtype=np.float32)


def _lift_to_world(
    u: float,
    v: float,
    depth_m: float,
    K: np.ndarray,
    world_T_ego: np.ndarray,
    ego_T_cam: np.ndarray,
) -> np.ndarray:
    """Back-project pixel (u, v) at given depth into world frame.

    Returns (3,) float64.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (u - cx) * depth_m / fx
    y_cam = (v - cy) * depth_m / fy
    z_cam = depth_m
    pt_cam = np.array([x_cam, y_cam, z_cam, 1.0], dtype=np.float64)
    world_T_cam = world_T_ego @ ego_T_cam
    return (world_T_cam @ pt_cam)[:3]


def _estimate_depth_from_lidar(
    u: float,
    v: float,
    lidar_world_pts: Optional[np.ndarray],
    K: np.ndarray,
    cam_T_world: np.ndarray,
    radius_px: float = 5.0,
) -> float:
    """Estimate depth at pixel (u, v) from nearby projected LiDAR returns.

    Returns median depth of LiDAR points within `radius_px` pixels of (u, v),
    or a fallback of 20.0 m if no close point is found.
    """
    fallback_depth = 20.0
    if lidar_world_pts is None or lidar_world_pts.shape[0] == 0:
        return fallback_depth
    pix, valid = project_points(lidar_world_pts, K, cam_T_world)
    if not valid.any():
        return fallback_depth
    pix_v = pix[valid]
    pts_v = lidar_world_pts[valid]
    dists = np.hypot(pix_v[:, 0] - u, pix_v[:, 1] - v)
    close = dists < radius_px
    if not close.any():
        return fallback_depth
    # Depth is the Z coordinate in camera frame.
    cam_pts = (cam_T_world @ np.hstack([pts_v, np.ones((pts_v.shape[0], 1))]).T).T
    return float(np.median(cam_pts[close, 2]))


def merge_cross_camera(
    masklets: list[Masklet],
    calibration: dict[str, CalibrationInfo],
    world_T_ego_by_cam: dict[str, np.ndarray],
    lidar_world_pts: Optional[np.ndarray] = None,
    radius_m: float = 1.5,
) -> list[Masklet]:
    """Assign global_object_id to masklets visible from multiple cameras.

    For each masklet, lifts its last-frame mask centroid to an approximate
    world-frame 3D position using LiDAR-assisted depth.  Then clusters
    masklets (across cameras) whose world positions are within `radius_m` of
    each other.  Masklets in the same cluster share a global_object_id.

    Args:
        masklets: all masklets from all cameras for this chunk.
        calibration: per-camera CalibrationInfo (K, ego_T_cam).
        world_T_ego_by_cam: per-camera 4×4 world_T_ego matrix at the last
            visible frame's timestamp.
        lidar_world_pts: (N, 3) merged world-frame LiDAR points (optional).
        radius_m: 3D clustering radius.

    Returns the same masklets list with global_object_id populated.
    """
    if not masklets:
        return masklets

    # Lift each masklet's mask centroid to world frame.
    world_positions: list[Optional[np.ndarray]] = []
    for mkl in masklets:
        calib = calibration.get(mkl.cam_id)
        if calib is None or mkl.cam_id not in world_T_ego_by_cam:
            world_positions.append(None)
            continue
        # Load the mask for the last visible frame.
        if not mkl.mask_paths:
            world_positions.append(None)
            continue
        try:
            from PIL import Image as PILImage
            mask = np.array(PILImage.open(mkl.mask_paths[-1])).astype(bool)
        except Exception:  # noqa: BLE001
            world_positions.append(None)
            continue

        centroid = _mask_centroid_px(mask)
        if centroid is None:
            world_positions.append(None)
            continue

        world_T_ego = world_T_ego_by_cam[mkl.cam_id]
        ego_T_cam = calib.ego_T_cam
        cam_T_ego = invert_se3(ego_T_cam)
        cam_T_world = cam_T_ego @ invert_se3(world_T_ego)

        depth = _estimate_depth_from_lidar(
            float(centroid[0]), float(centroid[1]),
            lidar_world_pts, calib.K, cam_T_world,
        )
        world_pt = _lift_to_world(
            float(centroid[0]), float(centroid[1]), depth,
            calib.K, world_T_ego, ego_T_cam,
        )
        world_positions.append(world_pt)

    # Union-Find clustering by 3D proximity.
    n = len(masklets)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    for i in range(n):
        if world_positions[i] is None:
            continue
        for j in range(i + 1, n):
            if world_positions[j] is None:
                continue
            if masklets[i].cam_id == masklets[j].cam_id:
                continue  # same-camera tracklets are not cross-camera duplicates
            dist = np.linalg.norm(world_positions[i] - world_positions[j])
            if dist < radius_m:
                union(i, j)

    # Assign global_object_id by root.
    from collections import defaultdict
    import uuid
    cluster_ids: dict[int, str] = defaultdict(lambda: str(uuid.uuid4())[:12])
    for i, mkl in enumerate(masklets):
        root = find(i)
        mkl.global_object_id = cluster_ids[root]

    return masklets
