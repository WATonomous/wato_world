"""Tests for visibility.py — UniLiPs Eq.1 occlusion test."""

from __future__ import annotations

import numpy as np
import pytest

from wato_semantic_lifting.visibility import visibility_test


def test_visible_point_in_front_of_surface():
    """Point at depth exactly at the surface → visible."""
    depth_map = np.full((100, 100), 10.0, dtype=np.float32)
    uv = np.array([[50.0, 50.0]], dtype=np.float32)
    d_cam = np.array([10.0], dtype=np.float32)
    result = visibility_test(d_cam, depth_map, uv, tolerance_m=0.5)
    assert result[0]


def test_occluded_point_behind_surface():
    """Point 2 m behind the surface → occluded."""
    depth_map = np.full((100, 100), 5.0, dtype=np.float32)
    uv = np.array([[50.0, 50.0]], dtype=np.float32)
    d_cam = np.array([7.5], dtype=np.float32)
    result = visibility_test(d_cam, depth_map, uv, tolerance_m=0.5)
    assert not result[0]


def test_point_within_tolerance_is_visible():
    """Point 0.3 m behind surface → within 0.5 m tolerance → visible."""
    depth_map = np.full((100, 100), 5.0, dtype=np.float32)
    uv = np.array([[50.0, 50.0]], dtype=np.float32)
    d_cam = np.array([5.3], dtype=np.float32)
    result = visibility_test(d_cam, depth_map, uv, tolerance_m=0.5)
    assert result[0]


def test_zero_depth_map_passes_all():
    """Depth map all zeros (no measurement) → all points considered visible."""
    depth_map = np.zeros((100, 100), dtype=np.float32)
    uv = np.array([[50.0, 50.0], [10.0, 10.0]], dtype=np.float32)
    d_cam = np.array([100.0, 200.0], dtype=np.float32)
    result = visibility_test(d_cam, depth_map, uv)
    assert result.all()


def test_empty_input_returns_empty():
    depth_map = np.full((100, 100), 5.0, dtype=np.float32)
    uv = np.zeros((0, 2), dtype=np.float32)
    d_cam = np.zeros(0, dtype=np.float32)
    result = visibility_test(d_cam, depth_map, uv)
    assert result.shape == (0,)


def test_neighborhood_uses_local_min():
    """Point visible at (50, 50) because nearby pixels have smaller depth (min pooling)."""
    depth_map = np.full((100, 100), 20.0, dtype=np.float32)
    depth_map[48:53, 48:53] = 3.0  # deep hole near (50, 50)

    # Point at depth 15 m — deeper than the local minimum (3 m) + tolerance.
    uv = np.array([[50.0, 50.0]], dtype=np.float32)
    d_cam = np.array([15.0], dtype=np.float32)
    result = visibility_test(d_cam, depth_map, uv, tolerance_m=0.5, neighborhood=3)
    # local min in neighborhood is 3.0; 15 > 3.5 → occluded
    assert not result[0]
