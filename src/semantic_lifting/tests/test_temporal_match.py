"""Tests for temporal_match.py — sweep↔frame matching and ego-motion math."""

from __future__ import annotations

import numpy as np
import pytest

from wato_semantic_lifting.temporal_match import (
    CameraFrameRef,
    compute_cam_T_world,
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
# compute_cam_T_world
# ---------------------------------------------------------------------------


def test_compute_cam_T_world_identity_transforms():
    """When ego pose and extrinsic are identity, cam_T_world is identity."""
    I = np.eye(4, dtype=np.float64)
    result = compute_cam_T_world(I, I)
    np.testing.assert_allclose(result, I, atol=1e-10)


def test_compute_cam_T_world_ego_translation():
    """Ego 1 m forward in world → a world point maps 1 m back in the camera."""
    I = np.eye(4, dtype=np.float64)
    world_T_ego_frame = I.copy()
    world_T_ego_frame[0, 3] = 1.0  # ego 1 m forward in world

    result = compute_cam_T_world(world_T_ego_frame, I)
    # cam_T_world = inv(I) @ inv(world_T_ego_frame) = translation of -1 on x.
    expected = I.copy()
    expected[0, 3] = -1.0
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_compute_cam_T_world_round_trips_a_point():
    """A world point at the ego origin projects to the camera's own offset."""
    I = np.eye(4, dtype=np.float64)
    world_T_ego_frame = I.copy()
    world_T_ego_frame[:3, 3] = [10.0, 5.0, 0.0]  # ego somewhere in the world
    ego_T_cam = I.copy()
    ego_T_cam[:3, 3] = [0.5, 0.0, 1.2]  # camera mounted forward + up of ego

    cam_T_world = compute_cam_T_world(world_T_ego_frame, ego_T_cam)
    # The world point coincident with the ego origin should land at -ego_T_cam
    # translation in the camera frame.
    p_world = np.array([10.0, 5.0, 0.0, 1.0])
    p_cam = cam_T_world @ p_world
    np.testing.assert_allclose(p_cam[:3], [-0.5, 0.0, -1.2], atol=1e-10)
