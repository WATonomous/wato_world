"""Cross-camera object identity merging.

Groups masklets from different cameras that observe the same 3D object by
back-projecting each masklet's last-frame mask centroid into world frame using
the metric depth from the depth_2d artifact, then clustering by 3D proximity.
Assigns a shared global_object_id to matched masklets.
"""

from __future__ import annotations

import collections
import logging
import uuid
from typing import Optional

import numpy as np

from wato_common.artifact_store import depth_2d_path, local_path
from wato_common.geometry import invert_se3, lift_pixel_to_world
from wato_perception_2d.fusion.tracker_2d import Masklet
from wato_perception_2d.io import CalibrationInfo

log = logging.getLogger(__name__)


def _mask_centroid_px(mask: np.ndarray) -> Optional[np.ndarray]:
    """Return (2,) float32 pixel [u, v] centroid of a binary mask, or None."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.mean(), ys.mean()], dtype=np.float32)


def _load_depth_at(
    bag_id: str,
    chunk_id: str,
    cam_id: str,
    frame_seq: int,
    u: float,
    v: float,
) -> Optional[float]:
    """Read metric depth at pixel (u, v) from the depth_2d artifact.

    Returns None if the artifact is missing, fit failed (fit_status==2), or
    depth is non-positive.
    """
    path = local_path(depth_2d_path(bag_id, chunk_id, cam_id, frame_seq))
    if not _os_exists(path):
        return None
    try:
        data = np.load(path)
        if int(data.get("fit_status", 2)) == 2:
            return None
        depth_m = data["depth_m"]
        H, W = depth_m.shape
        ui, vi = int(round(u)), int(round(v))
        ui = max(0, min(ui, W - 1))
        vi = max(0, min(vi, H - 1))
        d = float(depth_m[vi, ui])
        return d if d > 0 else None
    except Exception:  # noqa: BLE001
        return None


def _os_exists(path: str) -> bool:
    import os
    return os.path.exists(path)


def merge_cross_camera(
    masklets: list[Masklet],
    calibration: dict[str, CalibrationInfo],
    world_T_ego_by_cam: dict[str, np.ndarray],
    bag_id: str,
    chunk_id: str,
    radius_m: float = 1.5,
) -> list[Masklet]:
    """Assign global_object_id to masklets visible from multiple cameras.

    For each masklet, lifts its last-frame mask centroid to an approximate
    world-frame 3D position using the depth_2d metric depth artifact.
    Clusters masklets (across cameras) whose world positions are within
    `radius_m` of each other.  Masklets in the same cluster share a
    global_object_id.

    Args:
        masklets: all masklets from all cameras for this chunk.
        calibration: per-camera CalibrationInfo (K, ego_T_cam).
        world_T_ego_by_cam: per-camera 4×4 world_T_ego at the last valid frame.
        bag_id: for depth_2d artifact lookup.
        chunk_id: for depth_2d artifact lookup.
        radius_m: 3D clustering radius.

    Returns the same masklets list with global_object_id populated.
    """
    if not masklets:
        return masklets

    world_positions: list[Optional[np.ndarray]] = []
    for mkl in masklets:
        calib = calibration.get(mkl.cam_id)
        if calib is None or mkl.cam_id not in world_T_ego_by_cam:
            world_positions.append(None)
            continue
        if not mkl.mask_paths or not mkl.frames_present:
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

        last_frame_seq = mkl.frames_present[-1]
        depth = _load_depth_at(
            bag_id, chunk_id, mkl.cam_id, last_frame_seq,
            float(centroid[0]), float(centroid[1]),
        )
        if depth is None:
            world_positions.append(None)
            continue

        world_T_ego = world_T_ego_by_cam[mkl.cam_id]
        world_pt = lift_pixel_to_world(
            float(centroid[0]),
            float(centroid[1]),
            depth,
            calib.K,
            world_T_ego,
            calib.ego_T_cam,
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
                continue
            dist = float(np.linalg.norm(world_positions[i] - world_positions[j]))
            if dist < radius_m:
                union(i, j)

    cluster_ids: dict[int, str] = collections.defaultdict(
        lambda: str(uuid.uuid4())[:12]
    )
    for i, mkl in enumerate(masklets):
        root = find(i)
        mkl.global_object_id = cluster_ids[root]

    return masklets
