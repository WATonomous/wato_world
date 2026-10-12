"""Tests for depth_align.py — RANSAC inverse-depth fit and anchor-pair construction."""

from __future__ import annotations

import numpy as np
import pytest

from wato_perception_2d.fusion.depth_align import (
    build_anchor_pairs,
    disparity_to_depth,
    ransac_disparity_fit,
)


# ---------------------------------------------------------------------------
# ransac_disparity_fit
# ---------------------------------------------------------------------------


def _disparity_for(d_lidar: np.ndarray, scale: float, shift: float) -> np.ndarray:
    """Disparity that a perfect DA-V2 would emit for these depths."""
    return ((1.0 / d_lidar - shift) / scale).astype(np.float32)


def test_known_fit_recovered():
    """Exact (scale, shift) recovered from clean disparity / depth pairs."""
    scale, shift = 0.004, 0.002
    d_lidar = np.linspace(3.0, 60.0, 200).astype(np.float32)
    disp = _disparity_for(d_lidar, scale, shift)

    result = ransac_disparity_fit(d_lidar, disp, n_iter=500)

    assert result["fit_status"] == 0
    assert result["scale"] == pytest.approx(scale, rel=1e-3)
    assert result["shift"] == pytest.approx(shift, abs=1e-5)
    assert result["n_inliers"] == 200
    assert result["rmse_inliers_m"] < 0.01


def test_depth_falls_as_disparity_rises():
    """The regression this fit exists for: DA-V2 disparity is *larger* for
    nearer points. A depth-affine model with positive slope can't fit that;
    the inverse-depth fit must, and must produce metric depth that matches."""
    d_lidar = np.array([4.0, 8.0, 15.0, 30.0, 50.0] * 10, dtype=np.float32)
    disp = _disparity_for(d_lidar, 0.01, 0.0)
    assert disp[0] > disp[4]  # near → large disparity

    result = ransac_disparity_fit(d_lidar, disp)
    depth = disparity_to_depth(
        disp, result["scale"], result["shift"], out_dtype="float32"
    )

    assert result["fit_status"] == 0
    np.testing.assert_allclose(depth, d_lidar, rtol=1e-3)


def test_noisy_fit_recovered():
    """Fit recovered with 20% gross outliers among metric-noisy anchors."""
    rng = np.random.default_rng(42)
    scale, shift = 0.006, -0.001
    d_lidar = rng.uniform(3.0, 60.0, 300).astype(np.float32)
    disp = _disparity_for(d_lidar, scale, shift)
    d_lidar = d_lidar + rng.normal(0.0, 0.1, 300).astype(np.float32)
    idx = rng.choice(300, 60, replace=False)
    d_lidar[idx] = rng.uniform(1.0, 80.0, 60).astype(np.float32)

    result = ransac_disparity_fit(d_lidar, disp, n_iter=500, inlier_threshold_m=0.5)

    assert result["fit_status"] == 0
    assert result["scale"] == pytest.approx(scale, rel=0.02)
    assert result["n_inliers"] >= 230


def test_degenerate_fit_uses_fallback():
    """Depth range < min_depth_range_m triggers fallback (scale, shift)."""
    d_lidar = np.array([20.0, 20.2, 20.4], dtype=np.float32)  # range = 0.4 m < 5 m
    disp = _disparity_for(d_lidar, 0.005, 0.0)

    result = ransac_disparity_fit(
        d_lidar, disp, min_depth_range_m=5.0, fallback=(0.005, 0.0)
    )

    assert result["fit_status"] == 1
    assert result["scale"] == pytest.approx(0.005)
    assert result["shift"] == pytest.approx(0.0)
    assert result["n_inliers"] == 3


def test_degenerate_fit_no_fallback_fails():
    """Degenerate fit with no fallback returns fit_status=2."""
    d_lidar = np.array([20.0, 20.1], dtype=np.float32)
    disp = np.array([10.0, 10.05], dtype=np.float32)

    result = ransac_disparity_fit(d_lidar, disp, min_depth_range_m=5.0, fallback=None)

    assert result["fit_status"] == 2


def test_empty_pairs_returns_failed():
    """Empty input returns fit_status=2."""
    result = ransac_disparity_fit(
        np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    )
    assert result["fit_status"] == 2


def test_rmse_is_over_inliers_only():
    """rmse_inliers_m / n_inliers describe the inliers, not every anchor."""
    d_lidar = np.linspace(3.0, 60.0, 100).astype(np.float32)
    disp = _disparity_for(d_lidar, 0.004, 0.0)
    d_lidar[:10] += 30.0  # 10 gross outliers

    result = ransac_disparity_fit(d_lidar, disp)

    assert result["n_inliers"] == 90
    assert result["rmse_inliers_m"] < 0.01


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
    disparity = np.ones((480, 640), dtype=np.float32)

    d_lidar, disp = build_anchor_pairs(
        sky_pt, disparity, K, cam_T_world, (640, 480), sky_mask_top_fraction=0.3
    )

    # 140 < 144 (30% of 480) → should be filtered.
    assert d_lidar.shape[0] == 0


def test_no_lidar_points_returns_empty():
    K = _make_K()
    d_lidar, disp = build_anchor_pairs(
        np.zeros((0, 3)),
        np.ones((480, 640), dtype=np.float32),
        K,
        np.eye(4),
        (640, 480),
    )
    assert d_lidar.shape[0] == 0


def test_none_lidar_points_returns_empty():
    K = _make_K()
    d_lidar, disp = build_anchor_pairs(
        None, np.ones((480, 640), dtype=np.float32), K, np.eye(4), (640, 480)
    )
    assert d_lidar.shape[0] == 0


def test_visible_point_produces_anchor_pair():
    """A point in the lower half of the image produces a valid anchor pair."""
    K = _make_K()
    cam_T_world = np.eye(4)
    # Point at (0, 1, 10) in world = camera frame. v = cy + fy*1/10 = 240 + 50 = 290.
    # 290 >= 144 (30% sky cutoff for H=480) → not filtered.
    pt = np.array([[0.0, 1.0, 10.0]])
    disparity = np.full((480, 640), 5.0, dtype=np.float32)

    d_lidar, disp = build_anchor_pairs(
        pt, disparity, K, cam_T_world, (640, 480), sky_mask_top_fraction=0.3
    )

    assert d_lidar.shape[0] == 1
    assert d_lidar[0] == pytest.approx(10.0, abs=0.1)  # camera-frame depth
    assert disp[0] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# disparity_to_depth
# ---------------------------------------------------------------------------


def test_disparity_to_depth_shape_dtype_and_values():
    disp = np.array([[1.0, 2.0], [4.0, 10.0]], dtype=np.float32)
    out = disparity_to_depth(disp, scale=0.01, shift=0.0)
    assert out.shape == (2, 2)
    assert out.dtype == np.float16
    np.testing.assert_allclose(
        out.astype(np.float32), [[100.0, 50.0], [25.0, 10.0]], rtol=1e-3
    )


def test_disparity_to_depth_clamps_far_and_zeroes_invalid():
    # sky: disparity ~0 → 1/shift = 1000 m → clamped; negative inverse → 0.
    disp = np.array([[0.0, -5.0]], dtype=np.float32)
    out = disparity_to_depth(disp, scale=0.01, shift=0.001, out_dtype="float32")
    assert out[0, 0] == pytest.approx(250.0)
    assert out[0, 1] == 0.0
