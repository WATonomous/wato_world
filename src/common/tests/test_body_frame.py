"""Tests for body-frame transforms used by 3DAL/DetZero-style aggregation."""

from __future__ import annotations

import numpy as np

from wato_common.geometry import (
    body_to_world,
    enlarged_box_indices,
    heading_to_rotation,
    world_to_body,
)


def test_heading_to_rotation_zero_is_identity():
    np.testing.assert_allclose(heading_to_rotation(0.0), np.eye(3), atol=1e-12)


def test_heading_to_rotation_rotates_x_to_y_at_pi_over_2():
    R = heading_to_rotation(np.pi / 2)
    p = R @ np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(p, [0.0, 1.0, 0.0], atol=1e-12)


def test_world_to_body_center_maps_to_origin():
    center = np.array([10.0, -5.0, 1.2])
    pts = center.reshape(1, 3)
    body = world_to_body(pts, center, heading=0.7)
    np.testing.assert_allclose(body, [[0.0, 0.0, 0.0]], atol=1e-12)


def test_world_to_body_roundtrip():
    center = np.array([3.0, 4.0, 0.5])
    heading = 1.3  # ~74 degrees
    rng = np.random.default_rng(0)
    pts_world = rng.normal(size=(50, 3)) * 5.0 + center
    pts_body = world_to_body(pts_world, center, heading)
    pts_world_round = body_to_world(pts_body, center, heading)
    np.testing.assert_allclose(pts_world_round, pts_world, atol=1e-10)


def test_world_to_body_cancels_heading():
    """A point on the object's +x axis (in world after rotation) should land
    on the body +x axis after the transform."""
    center = np.array([0.0, 0.0, 0.0])
    heading = np.pi / 4
    # World-frame +x of the object is (cos h, sin h, 0); shift by 2 along it.
    pt_world = np.array([[2 * np.cos(heading), 2 * np.sin(heading), 0.0]])
    pt_body = world_to_body(pt_world, center, heading)
    np.testing.assert_allclose(pt_body, [[2.0, 0.0, 0.0]], atol=1e-12)


def test_enlarged_box_indices_inside_and_outside():
    center = np.array([0.0, 0.0, 0.0])
    size = np.array([4.0, 2.0, 1.5])  # W=4, L=2, H=1.5
    heading = 0.0
    pts = np.array(
        [
            [0.0, 0.0, 0.0],     # at center
            [1.9, 0.0, 0.0],     # just inside (W/2=2)
            [2.1, 0.0, 0.0],     # just outside at margin=1.0
            [3.5, 0.0, 0.0],     # outside even at margin=2.0
            [0.0, 0.0, 0.74],    # just inside H/2=0.75
            [0.0, 0.0, 0.76],    # just outside H
        ]
    )
    mask_tight = enlarged_box_indices(pts, center, size, heading, margin=1.0)
    np.testing.assert_array_equal(
        mask_tight,
        [True, True, False, False, True, False],
    )
    # Margin 1.5 enlarges each axis by 1.5x.
    mask_loose = enlarged_box_indices(pts, center, size, heading, margin=1.5)
    # W half-extent now 3.0, so 2.1 is in, 3.5 is out.
    np.testing.assert_array_equal(
        mask_loose,
        [True, True, True, False, True, True],
    )


def test_enlarged_box_indices_respects_heading():
    """A box rotated 90° should accept points along world +y, not +x."""
    center = np.array([0.0, 0.0, 0.0])
    size = np.array([4.0, 1.0, 1.0])
    pts = np.array(
        [
            [1.9, 0.0, 0.0],   # along world +x: outside the rotated box's W axis
            [0.0, 1.9, 0.0],   # along world +y: inside (since the box's W now points along +y)
        ]
    )
    mask = enlarged_box_indices(pts, center, size, heading=np.pi / 2, margin=1.0)
    np.testing.assert_array_equal(mask, [False, True])


def test_enlarged_box_indices_empty_input():
    pts = np.zeros((0, 3))
    out = enlarged_box_indices(pts, np.zeros(3), np.ones(3), 0.0)
    assert out.shape == (0,)
    assert out.dtype == bool
