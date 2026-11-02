"""Tests for projection.py — world → camera transform and mask lookup."""

from __future__ import annotations

import numpy as np

from wato_semantic_lifting.projection import (
    compute_cam_T_world,
    label_raster,
    pixel_indices,
)


def test_compute_cam_T_world_identity_transforms():
    """When ego pose and extrinsic are identity, cam_T_world is identity."""
    eye = np.eye(4, dtype=np.float64)
    np.testing.assert_allclose(compute_cam_T_world(eye, eye), eye, atol=1e-10)


def test_compute_cam_T_world_ego_translation():
    """Ego 1 m forward in world → a world point maps 1 m back in the camera."""
    eye = np.eye(4, dtype=np.float64)
    world_T_ego_frame = eye.copy()
    world_T_ego_frame[0, 3] = 1.0
    expected = eye.copy()
    expected[0, 3] = -1.0
    np.testing.assert_allclose(
        compute_cam_T_world(world_T_ego_frame, eye), expected, atol=1e-10
    )


def test_compute_cam_T_world_round_trips_a_point():
    """A world point at the ego origin lands at minus the camera's mount offset."""
    eye = np.eye(4, dtype=np.float64)
    world_T_ego_frame = eye.copy()
    world_T_ego_frame[:3, 3] = [10.0, 5.0, 0.0]
    ego_T_cam = eye.copy()
    ego_T_cam[:3, 3] = [0.5, 0.0, 1.2]
    p_cam = compute_cam_T_world(world_T_ego_frame, ego_T_cam) @ [10.0, 5.0, 0.0, 1.0]
    np.testing.assert_allclose(p_cam[:3], [-0.5, 0.0, -1.2], atol=1e-10)


def test_pixel_indices_round_and_clip_to_the_image():
    uv = np.array([[0.4, 0.6], [9.7, -2.0], [3.5, 4.49]], dtype=np.float32)
    ui, vi = pixel_indices(uv, width=10, height=5)
    assert ui.tolist() == [0, 9, 4] and vi.tolist() == [1, 0, 4]


def test_label_raster_innermost_mask_wins():
    big = np.zeros((4, 6), bool)
    big[:, :] = True
    small = np.zeros((4, 6), bool)
    small[1:3, 1:3] = True
    raster = label_raster([small, big], [7, 3])
    assert raster[0, 0] == 3 and raster[1, 1] == 7
    raster = label_raster([big, small], [3, 7])  # order doesn't matter
    assert raster[0, 0] == 3 and raster[1, 1] == 7


def test_label_raster_equal_areas_go_to_the_earlier_mask():
    a = np.zeros((2, 4), bool)
    a[:, :2] = True
    b = np.zeros((2, 4), bool)
    b[:, 1:3] = True
    raster = label_raster([a, b], [0, 1])
    assert raster.tolist() == [[0, 0, 1, -1], [0, 0, 1, -1]]
