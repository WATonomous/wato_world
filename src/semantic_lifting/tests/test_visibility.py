"""Tests for visibility.py — UniLiPs Eq. 1 occlusion test against the depth
map (trusted up close) and the sweep's own LiDAR returns."""

from __future__ import annotations

import numpy as np

from wato_semantic_lifting.config import VisibilityConfig
from wato_semantic_lifting.visibility import (
    depth_map_visible,
    lidar_occluder,
    pooled_occluder,
    visible,
)


def _visible(depth_map, d_cam, uv, tolerance_m=0.5, neighborhood=3):
    occ = pooled_occluder(np.asarray(depth_map, np.float32), neighborhood)
    uv = np.asarray(uv, int).reshape(-1, 2)
    return depth_map_visible(
        np.asarray(d_cam, np.float32), occ[uv[:, 1], uv[:, 0]], tolerance_m
    )


def test_point_on_the_surface_is_visible():
    assert _visible(np.full((100, 100), 10.0), [10.0], [50, 50])[0]


def test_point_behind_the_surface_is_occluded():
    assert not _visible(np.full((100, 100), 5.0), [7.5], [50, 50])[0]


def test_point_within_tolerance_is_visible():
    assert _visible(np.full((100, 100), 5.0), [5.3], [50, 50])[0]


def test_neighborhood_takes_the_nearest_surface():
    depth_map = np.full((100, 100), 20.0)
    depth_map[48:53, 48:53] = 3.0  # a near object beside the point's pixel
    assert not _visible(depth_map, [15.0], [45, 45], neighborhood=3)[0]
    assert _visible(depth_map, [15.0], [40, 40], neighborhood=3)[0]


def test_zero_depth_is_far_not_unknown():
    """0 = fitted inverse depth <= 0 (sky): never an occluder, and it doesn't
    hide a nearer surface in the same patch."""
    sky = np.zeros((100, 100))
    assert _visible(sky, [100.0, 200.0], [[50, 50], [10, 10]]).all()
    edge = np.zeros((100, 100))
    edge[:, 52:] = 4.0  # an object right of the point; sky on its pixel
    assert not _visible(edge, [30.0], [50, 50], neighborhood=3)[0]


def test_empty_input_returns_empty():
    assert _visible(np.full((10, 10), 5.0), np.zeros(0), np.zeros((0, 2))).shape == (0,)


# ---------------------------------------------------------------------------
# LiDAR occluder and the combined test.
# ---------------------------------------------------------------------------


def _lidar(ui, vi, d, occluder=None, size=(64, 48), cell_px=8, cells=1):
    d = np.asarray(d, np.float32)
    occluder = np.ones(d.shape, bool) if occluder is None else np.asarray(occluder)
    return lidar_occluder(
        np.asarray(ui), np.asarray(vi), d, occluder, size, cell_px, cells
    )


def test_lidar_occluder_is_the_nearest_return_in_the_cell_window():
    # Points 0/1 share a cell; point 2 is two cells away, outside a 1-cell window.
    occ = _lidar([10, 12, 30], [10, 10, 10], [20.0, 5.0, 3.0])
    assert occ.tolist() == [5.0, 5.0, 3.0]


def test_ground_never_occludes():
    """A ground return at 5 m doesn't hide the 20 m point beside it (it can
    itself be hidden, here by nothing nearer than that point)."""
    occ = _lidar([10, 12], [10, 10], [20.0, 5.0], occluder=[True, False])
    assert occ.tolist() == [20.0, 20.0]
    assert np.isinf(_lidar([10], [10], [5.0], occluder=[False])).all()


def test_combined_test_trusts_the_depth_map_only_up_close():
    cfg = VisibilityConfig(depth_map_max_range_m=10.0)
    d = np.array([30.0, 30.0], np.float32)
    lidar = np.full(2, np.inf, np.float32)
    # A depth-map surface at 8 m occludes; one at 15 m is past the range where
    # the map is trusted (its error grows ~ d²), so it doesn't.
    vis = visible(d, np.array([8.0, 15.0], np.float32), lidar, cfg)
    assert vis.tolist() == [False, True]


def test_combined_test_lidar_tolerance_grows_with_range():
    cfg = VisibilityConfig(tolerance_m=0.5, lidar_tolerance_per_m=0.05)
    far = np.inf * np.ones(3, np.float32)
    d = np.array([42.0, 43.0, 10.0], np.float32)
    lidar = np.array([40.0, 40.0, 8.0], np.float32)
    # 40 m: 0.5 + 2.0 m of slack; 8 m: 0.5 + 0.4 m.
    assert visible(d, far, lidar, cfg).tolist() == [True, False, False]
