"""Temporal matching between LiDAR sweeps and camera frames.

Associates each sweep to the nearest camera frame per camera within a
configurable time tolerance, and computes the ``cam_T_world`` transform that
projects a sweep's (already world-registered) points into that camera.

Note on ego motion: ``lidar_preprocessing`` deskews every sweep and expresses
its points in the SLAM world frame at each point's own sensor timestamp (see
``deskew/_core.py``). The sweep→frame ego motion is therefore *already* baked
into the point coordinates, so lifting only needs the camera's world pose at
the matched frame time — there is no residual ego motion left to compensate,
and no LiDAR-frame / sweep-pose term is involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from wato_common.geometry import invert_se3


@dataclass
class CameraFrameRef:
    """Minimal reference to a camera frame used for label lifting."""

    cam_id: str
    camera_seq: int
    timestamp_ns: int
    world_T_ego: np.ndarray  # (4, 4) float64


def match_sweep_to_frames(
    sweep_timestamp_ns: int,
    frame_refs: list[CameraFrameRef],
    max_offset_s: float = 0.05,
) -> dict[str, CameraFrameRef]:
    """Return the nearest camera frame per camera within max_offset_s.

    Args:
        sweep_timestamp_ns: LiDAR sweep timestamp in nanoseconds.
        frame_refs: all available camera frames (all cameras).
        max_offset_s: maximum |t_sweep - t_frame| in seconds.

    Returns:
        dict mapping cam_id → nearest CameraFrameRef for cameras within the
        time tolerance.  Cameras outside the tolerance are excluded.
    """
    max_offset_ns = int(max_offset_s * 1e9)

    # Group by camera.
    by_cam: dict[str, list[CameraFrameRef]] = {}
    for fr in frame_refs:
        by_cam.setdefault(fr.cam_id, []).append(fr)

    matched: dict[str, CameraFrameRef] = {}
    for cam_id, frames in by_cam.items():
        best: Optional[CameraFrameRef] = None
        best_offset = max_offset_ns + 1
        for fr in frames:
            offset = abs(fr.timestamp_ns - sweep_timestamp_ns)
            if offset <= max_offset_ns and offset < best_offset:
                best = fr
                best_offset = offset
        if best is not None:
            matched[cam_id] = best

    return matched


def compute_cam_T_world(
    world_T_ego_frame: np.ndarray,
    ego_T_cam: np.ndarray,
) -> np.ndarray:
    """Compute the transform that projects world-frame points into a camera.

    The sweep's points are already in the SLAM world frame (deskewed per-point
    at their own timestamps upstream), so the only pose that matters is the
    camera's world pose at the matched frame time:

        cam_T_world = inv(ego_T_cam) @ inv(world_T_ego_frame)

    Ego motion between sweep and frame is already accounted for by the upstream
    world registration; there is no LiDAR-frame or sweep-pose term here. Residual
    error on *dynamic* objects (scene motion over the sweep↔frame offset) is not
    addressed by this transform — see "Dynamic-point handling" in the design doc.

    Args:
        world_T_ego_frame: ego pose at camera frame timestamp, (4, 4) float64.
        ego_T_cam: fixed camera extrinsic (cam frame → ego frame), (4, 4) float64.

    Returns:
        (4, 4) float64 SE3 transform: world frame → camera frame.
    """
    return invert_se3(ego_T_cam) @ invert_se3(world_T_ego_frame)
