"""Tests for temporal_match.py — sweep↔frame matching and ego-motion math."""

from __future__ import annotations

import numpy as np
import pytest

from wato_semantic_lifting.temporal_match import (
    CameraFrameRef,
    compute_cam_T_lidar,
    match_sweep_to_frames,
)


def _frame(cam_id: str, seq: int, ts_ns: int) -> CameraFrameRef:
    return CameraFrameRef(
        cam_id=cam_id,
        camera_seq=seq,
        timestamp_ns=ts_ns,
        world_T_ego=np.eye(4, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# match_sweep_to_frames
# ---------------------------------------------------------------------------


def test_match_within_tolerance_returns_nearest():
    frames = [
        _frame("cam_front", 0, 1_000_000_000),   # 1.000 s
        _frame("cam_front", 1, 1_030_000_000),   # 1.030 s — nearest
        _frame("cam_front", 2, 1_060_000_000),   # 1.060 s
    ]
    result = match_sweep_to_frames(1_025_000_000, frames, max_offset_s=0.05)
    assert "cam_front" in result
    assert result["cam_front"].camera_seq == 1


def test_match_outside_tolerance_excluded():
    frames = [_frame("cam_front", 0, 2_000_000_000)]  # 1 s away from sweep
    result = match_sweep_to_frames(1_000_000_000, frames, max_offset_s=0.05)
    assert "cam_front" not in result


def test_match_multiple_cameras():
    frames = [
        _frame("cam_front", 0, 1_010_000_000),
        _frame("cam_back", 0, 1_020_000_000),
    ]
    result = match_sweep_to_frames(1_000_000_000, frames, max_offset_s=0.05)
    assert "cam_front" in result
    assert "cam_back" in result


def test_match_empty_frames_returns_empty():
    result = match_sweep_to_frames(1_000_000_000, [], max_offset_s=0.05)
    assert result == {}


def test_match_exact_timestamp_zero_offset():
    frames = [_frame("cam_front", 0, 1_000_000_000)]
    result = match_sweep_to_frames(1_000_000_000, frames, max_offset_s=0.0)
    assert "cam_front" in result


# ---------------------------------------------------------------------------
# compute_cam_T_lidar
# ---------------------------------------------------------------------------


def test_compute_cam_T_lidar_identity_transforms():
    """When all transforms are identity, cam_T_lidar should also be identity."""
    I = np.eye(4, dtype=np.float64)
    result = compute_cam_T_lidar(I, I, I, I)
    np.testing.assert_allclose(result, I, atol=1e-10)


def test_compute_cam_T_lidar_pure_translation():
    """Sweep ego shifted by (1, 0, 0) relative to frame ego → lidar shifted accordingly."""
    I = np.eye(4, dtype=np.float64)
    world_T_ego_sweep = I.copy()
    world_T_ego_sweep[0, 3] = 1.0  # sweep ego 1 m forward in world

    result = compute_cam_T_lidar(world_T_ego_sweep, I, I, I)
    # cam_T_lidar = inv(I) @ inv(I) @ world_T_ego_sweep @ I = world_T_ego_sweep
    np.testing.assert_allclose(result, world_T_ego_sweep, atol=1e-10)


def test_compute_cam_T_lidar_no_ego_motion_equals_static():
    """When sweep and frame have the same ego pose, the dynamic compensation cancels."""
    I = np.eye(4, dtype=np.float64)
    ego_T_lidar = I.copy()
    ego_T_lidar[0, 3] = 0.5  # LiDAR is 0.5 m forward of ego

    # Same ego pose at both timestamps.
    result = compute_cam_T_lidar(I, I, ego_T_lidar, I)
    np.testing.assert_allclose(result, ego_T_lidar, atol=1e-10)
