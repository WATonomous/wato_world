"""Tests for the LiDAR-based scale alignment helper.

The DA V2 model load itself is not exercised here (it requires the
depth-anything-v2 package + a 1.3 GB checkpoint).  When the package is
present, `DepthAnythingV2Estimator.predict` is tested at integration time
in the running container — not in this unit-level suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from wato_perception_2d.depth_anything import (
    DepthAlignmentResult,
    align_depth_to_lidar,
)


def _make_camera() -> tuple[np.ndarray, np.ndarray]:
    """Pinhole camera at world origin looking down +z; identity extrinsic."""
    K = np.array([[100.0, 0.0, 320.0],
                  [0.0, 100.0, 240.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    cam_T_world = np.eye(4, dtype=np.float64)
    return K, cam_T_world


def _project(pts_world: np.ndarray, K: np.ndarray, cam_T_world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts_hom = np.concatenate([pts_world, np.ones((pts_world.shape[0], 1))], axis=1)
    pts_cam = (cam_T_world @ pts_hom.T).T[:, :3]
    z = pts_cam[:, 2]
    pix = (K @ pts_cam.T).T
    u = (pix[:, 0] / pix[:, 2]).astype(np.int32)
    v = (pix[:, 1] / pix[:, 2]).astype(np.int32)
    return u, v, z


def test_align_recovers_known_scale_and_shift():
    """Synthetic: depth_map = (z_lidar - true_shift) / true_scale at projected pixels.
    align_depth_to_lidar should recover (true_scale, true_shift) within tolerance.
    """
    rng = np.random.default_rng(42)
    K, cam_T_world = _make_camera()

    # 200 LiDAR points scattered in front of the camera.
    N = 200
    z_metric = rng.uniform(5.0, 50.0, size=N)
    x = rng.uniform(-5.0, 5.0, size=N)
    y = rng.uniform(-3.0, 3.0, size=N)
    pts_world = np.stack([x, y, z_metric], axis=1)

    u, v, _ = _project(pts_world, K, cam_T_world)
    in_image = (u >= 0) & (u < 640) & (v >= 0) & (v < 480)
    pts_world = pts_world[in_image]
    u = u[in_image]; v = v[in_image]
    z_metric = z_metric[in_image]
    assert pts_world.shape[0] >= 100, "set up should have >=100 in-frame points"

    true_scale, true_shift = 2.5, 1.0
    d_rel = (z_metric - true_shift) / true_scale

    depth_map = np.full((480, 640), np.nan, dtype=np.float32)
    depth_map[v, u] = d_rel.astype(np.float32)
    # Fill NaNs with a plausible value so the helper's int indexing doesn't trip;
    # the helper samples only at projected (u, v) anyway.
    depth_map = np.nan_to_num(depth_map, nan=1.0)

    result = align_depth_to_lidar(
        depth_relative=depth_map,
        lidar_world_pts=pts_world,
        K=K,
        cam_T_world=cam_T_world,
    )
    assert isinstance(result, DepthAlignmentResult)
    assert result.scale_method == "lidar_aligned"
    assert result.n_overlap_pts >= 50
    assert result.scale == pytest.approx(true_scale, abs=0.05)
    assert result.shift == pytest.approx(true_shift, abs=0.5)
    assert result.residual_rmse is not None and result.residual_rmse < 0.1


def test_align_returns_uncalibrated_with_too_few_overlap():
    """No LiDAR overlap → method='uncalibrated', scale=1, shift=0."""
    K, cam_T_world = _make_camera()
    pts_world = np.zeros((0, 3))
    depth = np.ones((100, 100), dtype=np.float32)

    result = align_depth_to_lidar(
        depth_relative=depth,
        lidar_world_pts=pts_world,
        K=K,
        cam_T_world=cam_T_world,
    )
    assert result.scale_method == "uncalibrated"
    assert result.scale == 1.0
    assert result.shift == 0.0
    assert result.n_overlap_pts == 0


def test_align_rejects_invalid_depth_shape():
    K, cam_T_world = _make_camera()
    with pytest.raises(ValueError, match="depth_relative must be"):
        align_depth_to_lidar(
            depth_relative=np.zeros((10, 10, 3), dtype=np.float32),
            lidar_world_pts=np.zeros((1, 3)),
            K=K,
            cam_T_world=cam_T_world,
        )


def test_align_ignores_points_behind_camera():
    """Points with negative cam-frame z are dropped before any RANSAC iteration."""
    K, cam_T_world = _make_camera()
    pts_world = np.array(
        [
            [0.0, 0.0, -5.0],   # behind camera
            [0.0, 0.0, -10.0],
        ]
    )
    depth = np.ones((480, 640), dtype=np.float32)
    result = align_depth_to_lidar(
        depth_relative=depth,
        lidar_world_pts=pts_world,
        K=K,
        cam_T_world=cam_T_world,
    )
    assert result.scale_method == "uncalibrated"
    assert result.n_overlap_pts == 0


def test_align_robust_to_outliers():
    """RANSAC should ignore a small fraction of outlier (z_lidar, d_rel) pairs."""
    rng = np.random.default_rng(7)
    K, cam_T_world = _make_camera()

    N = 200
    z_metric = rng.uniform(5.0, 50.0, size=N)
    x = rng.uniform(-5.0, 5.0, size=N)
    y = rng.uniform(-3.0, 3.0, size=N)
    pts_world = np.stack([x, y, z_metric], axis=1)
    u, v, _ = _project(pts_world, K, cam_T_world)
    in_image = (u >= 0) & (u < 640) & (v >= 0) & (v < 480)
    pts_world = pts_world[in_image]
    u = u[in_image]; v = v[in_image]
    z_metric = z_metric[in_image]

    true_scale, true_shift = 2.5, 1.0
    d_rel = (z_metric - true_shift) / true_scale

    # Inject 20% outliers — flip their d_rel sign so they're hugely wrong.
    n_outlier = max(1, len(d_rel) // 5)
    outlier_idx = rng.choice(len(d_rel), size=n_outlier, replace=False)
    d_rel = d_rel.copy()
    d_rel[outlier_idx] *= -1

    depth_map = np.ones((480, 640), dtype=np.float32)
    depth_map[v, u] = d_rel.astype(np.float32)

    result = align_depth_to_lidar(
        depth_relative=depth_map,
        lidar_world_pts=pts_world,
        K=K,
        cam_T_world=cam_T_world,
        max_iters=200,
    )
    assert result.scale_method == "lidar_aligned"
    assert result.scale == pytest.approx(true_scale, abs=0.15)
    assert result.shift == pytest.approx(true_shift, abs=1.0)
