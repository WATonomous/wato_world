"""Temporal matching between LiDAR sweeps and camera frames.

Associates each sweep to the nearest camera frame per camera within a
configurable time tolerance, and computes the time-corrected cam_T_lidar
transform to account for ego motion between sweep and frame timestamps.
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


def compute_cam_T_lidar(
    world_T_ego_sweep: np.ndarray,
    world_T_ego_frame: np.ndarray,
    ego_T_lidar: np.ndarray,
    ego_T_cam: np.ndarray,
) -> np.ndarray:
    """Compute the time-corrected cam_T_lidar transform.

    Accounts for ego motion between the LiDAR sweep time and the camera frame
    time — critical at 25ms desync where dynamic points shift ~75cm.

    Formula:
        cam_T_lidar = inv(ego_T_cam) @ inv(world_T_ego_frame)
                      @ world_T_ego_sweep @ ego_T_lidar

    Args:
        world_T_ego_sweep: ego pose at sweep timestamp, (4, 4) float64.
        world_T_ego_frame: ego pose at camera frame timestamp, (4, 4) float64.
        ego_T_lidar: fixed LiDAR extrinsic, (4, 4) float64.
        ego_T_cam: fixed camera extrinsic, (4, 4) float64.

    Returns:
        (4, 4) float64 SE3 transform: LiDAR frame → camera frame.
    """
    cam_T_ego_frame = invert_se3(ego_T_cam)
    ego_T_world_frame = invert_se3(world_T_ego_frame)
    return cam_T_ego_frame @ ego_T_world_frame @ world_T_ego_sweep @ ego_T_lidar
