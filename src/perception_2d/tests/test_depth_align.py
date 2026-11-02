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


def test_depth_falls_as_disparity_rises():
    """DA-V2 disparity is *larger* for nearer points. A depth-affine model with
    positive slope can't fit that; the inverse-depth fit must, and must produce
    metric depth that matches."""
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


def test_no_fit_without_a_fallback():
    d_lidar = np.array([20.0, 20.1], dtype=np.float32)
    disp = np.array([10.0, 10.05], dtype=np.float32)
    result = ransac_disparity_fit(d_lidar, disp, min_depth_range_m=5.0, fallback=None)
    assert result["fit_status"] == 2

    empty = np.zeros(0, dtype=np.float32)
    assert ransac_disparity_fit(empty, empty)["fit_status"] == 2


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


def test_anchor_pairs_skip_the_sky():
    """With the camera at the world origin looking down +Z (fx = 500, cy = 240,
    H = 480), a point 2 m up at 10 m projects to v = 140, inside the top 30%
    (v < 144), and is dropped; one 1 m down projects to v = 290 and becomes an
    anchor at its camera-frame depth."""
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    points = np.array([[0.0, -2.0, 10.0], [0.0, 1.0, 10.0]])
    disparity = np.full((480, 640), 5.0, dtype=np.float32)

    d_lidar, disp = build_anchor_pairs(
        points, disparity, K, np.eye(4), (640, 480), sky_mask_top_fraction=0.3
    )

    assert d_lidar.tolist() == pytest.approx([10.0], abs=0.1)
    assert disp.tolist() == pytest.approx([5.0])


# ---------------------------------------------------------------------------
# disparity_to_depth
# ---------------------------------------------------------------------------


def test_disparity_to_depth_clamps_far_and_zeroes_invalid():
    # sky: disparity ~0 → 1/shift = 1000 m → clamped; negative inverse → 0.
    disp = np.array([[0.0, -5.0]], dtype=np.float32)
    out = disparity_to_depth(disp, scale=0.01, shift=0.001, out_dtype="float32")
    assert out[0, 0] == pytest.approx(250.0)
    assert out[0, 1] == 0.0
