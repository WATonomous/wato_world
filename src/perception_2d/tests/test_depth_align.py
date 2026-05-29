"""Tests for depth_align.py — RANSAC affine fit and anchor-pair construction."""

from __future__ import annotations

import numpy as np
import pytest

from wato_perception_2d.fusion.depth_align import (
    apply_affine,
    build_anchor_pairs,
    ransac_affine_fit,
)


# ---------------------------------------------------------------------------
# ransac_affine_fit
# ---------------------------------------------------------------------------


def test_known_affine_recovered():
    """Exact affine (a=2.5, b=1.0) is recoverable to within 0.5% from clean pairs."""
    a_true, b_true = 2.5, 1.0
    d_da = np.linspace(1.0, 20.0, 200).astype(np.float32)
    d_lidar = (a_true * d_da + b_true).astype(np.float32)

    result = ransac_affine_fit(d_lidar, d_da, n_iter=500)

    assert result["fit_status"] == 0
    assert abs(result["a"] - a_true) / a_true < 0.005
    assert abs(result["b"] - b_true) < 0.05


def test_noisy_affine_recovered():
    """Affine recovered even with 20% outlier noise."""
    rng = np.random.default_rng(42)
    a_true, b_true = 1.8, 0.5
    d_da = rng.uniform(1.0, 40.0, 300).astype(np.float32)
    d_lidar = (a_true * d_da + b_true).astype(np.float32)
    # Corrupt 20% of points.
    n_corrupt = 60
    idx = rng.choice(300, n_corrupt, replace=False)
    d_lidar[idx] += rng.uniform(-15, 15, n_corrupt).astype(np.float32)

    result = ransac_affine_fit(d_lidar, d_da, n_iter=500, inlier_threshold_m=0.5)

    assert result["fit_status"] == 0
    assert abs(result["a"] - a_true) / a_true < 0.02


def test_degenerate_fit_uses_fallback():
    """Depth range < min_depth_range_m triggers fallback (a, b)."""
    d_da = np.array([10.0, 10.1, 10.2], dtype=np.float32)
    d_lidar = np.array([20.0, 20.2, 20.4], dtype=np.float32)  # range = 0.4 m < 5 m

    result = ransac_affine_fit(
        d_lidar, d_da, min_depth_range_m=5.0, fallback=(2.0, 0.5)
    )

    assert result["fit_status"] == 1
    assert result["a"] == pytest.approx(2.0)
    assert result["b"] == pytest.approx(0.5)


def test_degenerate_fit_no_fallback_fails():
    """Degenerate fit with no fallback returns fit_status=2."""
    d_da = np.array([10.0, 10.05], dtype=np.float32)
    d_lidar = np.array([20.0, 20.1], dtype=np.float32)

    result = ransac_affine_fit(d_lidar, d_da, min_depth_range_m=5.0, fallback=None)

    assert result["fit_status"] == 2


def test_empty_pairs_returns_failed():
    """Empty input returns fit_status=2."""
    result = ransac_affine_fit(
        np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    )
    assert result["fit_status"] == 2


# ---------------------------------------------------------------------------
# build_anchor_pairs
# ---------------------------------------------------------------------------


def _make_K(fx: float = 500.0, cx: float = 320.0, cy: float = 240.0) -> np.ndarray:
    return np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=np.float64)


def test_sky_filter_removes_upper_fraction():
    """Points projecting into the top 30% of the image are filtered out."""
    K = _make_K()
    # Build a point that projects into the upper sky region (v < 144 for H=480).
    # Camera at origin, looking down +Z. Point at (0, -3, 5) → v ≈ 240 - 3*500/5 = 240 - 300 < 0
    # Instead: point directly ahead at (0, 0, 10) → centred at cx, cy = (320, 240).
    # Point with large negative y (up in camera) → small v.
    cam_T_world = np.eye(4)
    # Place a point above the principal point so it projects into the sky zone.
    sky_pt = np.array([[0.0, -2.0, 10.0]])  # y=-2 → v = cy - fy*2/10 = 240 - 100 = 140
    rel_depth = np.ones((480, 640), dtype=np.float32)

    d_lidar, d_da = build_anchor_pairs(
        sky_pt, rel_depth, K, cam_T_world, (640, 480), sky_mask_top_fraction=0.3
    )

    # 140 < 144 (30% of 480) → should be filtered.
    assert d_lidar.shape[0] == 0


def test_no_lidar_points_returns_empty():
    K = _make_K()
    d_lidar, d_da = build_anchor_pairs(
        np.zeros((0, 3)), np.ones((480, 640), dtype=np.float32),
        K, np.eye(4), (640, 480)
    )
    assert d_lidar.shape[0] == 0


def test_none_lidar_points_returns_empty():
    K = _make_K()
    d_lidar, d_da = build_anchor_pairs(
        None, np.ones((480, 640), dtype=np.float32),
        K, np.eye(4), (640, 480)
    )
    assert d_lidar.shape[0] == 0


def test_visible_point_produces_anchor_pair():
    """A point in the lower half of the image produces a valid anchor pair."""
    K = _make_K()
    cam_T_world = np.eye(4)
    # Point at (0, 1, 10) in world = camera frame. v = cy + fy*1/10 = 240 + 50 = 290.
    # 290 >= 144 (30% sky cutoff for H=480) → not filtered.
    pt = np.array([[0.0, 1.0, 10.0]])
    rel_depth = np.full((480, 640), 5.0, dtype=np.float32)

    d_lidar, d_da = build_anchor_pairs(
        pt, rel_depth, K, cam_T_world, (640, 480), sky_mask_top_fraction=0.3
    )

    assert d_lidar.shape[0] == 1
    assert d_lidar[0] == pytest.approx(10.0, abs=0.1)  # camera-frame depth
    assert d_da[0] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# apply_affine
# ---------------------------------------------------------------------------


def test_apply_affine_output_shape_and_dtype():
    rel = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out = apply_affine(rel, a=2.0, b=0.5)
    assert out.shape == (2, 2)
    assert out.dtype == np.float16


def test_apply_affine_clamps_to_250():
    rel = np.array([[200.0]], dtype=np.float32)
    out = apply_affine(rel, a=2.0, b=0.0)
    assert float(out[0, 0]) == pytest.approx(250.0, abs=0.1)
