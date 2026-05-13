"""Tests for cross_cam_merge.py — world-position lifting and Union-Find clustering."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image as PILImage

from wato_perception_2d.cross_cam_merge import (
    _estimate_depth_from_lidar,
    _lift_to_world,
    _mask_centroid_px,
    merge_cross_camera,
)
from wato_perception_2d.io import CalibrationInfo
from wato_perception_2d.tracker_2d import Masklet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_calib(fx: float = 500.0, cx: float = 320.0, cy: float = 240.0) -> CalibrationInfo:
    K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=np.float64)
    ego_T_cam = np.eye(4, dtype=np.float64)
    return CalibrationInfo(K=K, ego_T_cam=ego_T_cam)


def _masklet_with_png(
    tmp_path, masklet_id: str, cam_id: str, mask: np.ndarray
) -> Masklet:
    """Write the mask as a PNG and return a Masklet pointing to it."""
    d = tmp_path / masklet_id
    d.mkdir(exist_ok=True)
    png = str(d / "000000.png")
    PILImage.fromarray(mask.astype(np.uint8) * 255).save(png)
    return Masklet(
        masklet_id=masklet_id,
        bag_id="bag0",
        chunk_id="chunk0",
        cam_id=cam_id,
        cls="car",
        score=0.9,
        frames_present=[0],
        mask_paths=[png],
    )


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------

def test_mask_centroid_px_center():
    """Centroid of a centered square should be at its geometric center."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 40:60] = True  # 20×20 block centered at (49.5, 49.5)
    c = _mask_centroid_px(mask)
    assert c is not None
    np.testing.assert_allclose(c, [49.5, 49.5], atol=0.5)


def test_mask_centroid_px_empty_returns_none():
    mask = np.zeros((100, 100), dtype=bool)
    assert _mask_centroid_px(mask) is None


def test_lift_to_world_straight_ahead():
    """Pixel at principal point (cx, cy) at depth d → world point on optical axis."""
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    world_T_ego = np.eye(4, dtype=np.float64)
    ego_T_cam = np.eye(4, dtype=np.float64)
    pt = _lift_to_world(320.0, 240.0, 20.0, K, world_T_ego, ego_T_cam)
    # Pixel at principal point → zero lateral offset, depth = 20 m
    np.testing.assert_allclose(pt, [0, 0, 20], atol=1e-6)


def test_lift_to_world_known_offset():
    """Pixel shifted by exactly fx pixels right → 1 m lateral offset at 1 m depth."""
    fx = 500.0
    K = np.array([[fx, 0, 0], [0, fx, 0], [0, 0, 1]], dtype=np.float64)
    world_T_ego = np.eye(4, dtype=np.float64)
    ego_T_cam = np.eye(4, dtype=np.float64)
    pt = _lift_to_world(fx, 0.0, 1.0, K, world_T_ego, ego_T_cam)
    np.testing.assert_allclose(pt, [1.0, 0.0, 1.0], atol=1e-6)


def test_estimate_depth_from_lidar_no_points_returns_fallback():
    """No LiDAR points → fallback depth of 20 m."""
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    cam_T_world = np.eye(4, dtype=np.float64)
    depth = _estimate_depth_from_lidar(320, 240, None, K, cam_T_world)
    assert depth == pytest.approx(20.0)


def test_estimate_depth_from_lidar_close_point():
    """LiDAR point directly on the pixel → should return its camera-frame depth."""
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    cam_T_world = np.eye(4, dtype=np.float64)
    # A point at (0, 0, 10) in camera frame → world frame (same, cam_T_world=I)
    # Projects to principal point (320, 240) in image space.
    lidar_pts = np.array([[0.0, 0.0, 10.0]])
    depth = _estimate_depth_from_lidar(320, 240, lidar_pts, K, cam_T_world)
    assert depth == pytest.approx(10.0, abs=1.0)


# ---------------------------------------------------------------------------
# Integration tests for merge_cross_camera
# ---------------------------------------------------------------------------

def test_same_camera_masklets_never_merged(tmp_path):
    """
    Two masklets from the same camera — even at identical world positions —
    must never share a global_object_id (they cannot be cross-camera duplicates).
    """
    H, W = 480, 640
    mask = np.zeros((H, W), dtype=bool)
    mask[230:250, 310:330] = True  # centroid ≈ principal point

    mkl1 = _masklet_with_png(tmp_path, "m1", "cam_front", mask)
    mkl2 = _masklet_with_png(tmp_path, "m2", "cam_front", mask)

    calib = {"cam_front": _simple_calib()}
    world_T_ego = {"cam_front": np.eye(4, dtype=np.float64)}

    result = merge_cross_camera([mkl1, mkl2], calib, world_T_ego)

    assert result[0].global_object_id != result[1].global_object_id


def test_different_cameras_same_world_position_merged(tmp_path):
    """
    Two cameras with identical calibration, ego at world origin, and the same
    image centroid → identical back-projected world point → should be merged.
    """
    H, W = 480, 640
    # Centroid at exactly the principal point → back-projects to (0, 0, 20 m).
    mask = np.zeros((H, W), dtype=bool)
    mask[230:250, 310:330] = True  # center ≈ (320, 240)

    mkl1 = _masklet_with_png(tmp_path, "m1", "cam_front", mask)
    mkl2 = _masklet_with_png(tmp_path, "m2", "cam_back", mask)

    calib = {"cam_front": _simple_calib(), "cam_back": _simple_calib()}
    world_T_ego = {
        "cam_front": np.eye(4, dtype=np.float64),
        "cam_back": np.eye(4, dtype=np.float64),
    }

    result = merge_cross_camera([mkl1, mkl2], calib, world_T_ego, radius_m=1.5)

    assert result[0].global_object_id == result[1].global_object_id


def test_different_cameras_far_world_positions_not_merged(tmp_path):
    """
    Two cameras with centroids that map to world positions > 1.5 m apart
    should get different global_object_ids.
    """
    H, W = 480, 640
    # cam_front centroid ≈ (320, 240) → world ≈ (0, 0, 20)
    mask_center = np.zeros((H, W), dtype=bool)
    mask_center[230:250, 310:330] = True

    # cam_back centroid at top-left corner (u≈5, v≈5) → world ≈ (-12.6, -9.4, 20)
    mask_corner = np.zeros((H, W), dtype=bool)
    mask_corner[0:10, 0:10] = True

    mkl1 = _masklet_with_png(tmp_path, "m1", "cam_front", mask_center)
    mkl2 = _masklet_with_png(tmp_path, "m2", "cam_back", mask_corner)

    calib = {"cam_front": _simple_calib(), "cam_back": _simple_calib()}
    world_T_ego = {
        "cam_front": np.eye(4, dtype=np.float64),
        "cam_back": np.eye(4, dtype=np.float64),
    }

    result = merge_cross_camera([mkl1, mkl2], calib, world_T_ego, radius_m=1.5)

    assert result[0].global_object_id != result[1].global_object_id


def test_three_cameras_two_merged_one_separate(tmp_path):
    """
    cam_a and cam_b see the same object (merged); cam_c sees a distant one.
    Result: a and b share an ID; c gets its own.
    """
    H, W = 480, 640
    mask_center = np.zeros((H, W), dtype=bool)
    mask_center[230:250, 310:330] = True

    mask_far = np.zeros((H, W), dtype=bool)
    mask_far[0:10, 0:10] = True

    mkl_a = _masklet_with_png(tmp_path, "ma", "cam_a", mask_center)
    mkl_b = _masklet_with_png(tmp_path, "mb", "cam_b", mask_center)
    mkl_c = _masklet_with_png(tmp_path, "mc", "cam_c", mask_far)

    calib = {k: _simple_calib() for k in ("cam_a", "cam_b", "cam_c")}
    world_T_ego = {k: np.eye(4, dtype=np.float64) for k in ("cam_a", "cam_b", "cam_c")}

    result = merge_cross_camera([mkl_a, mkl_b, mkl_c], calib, world_T_ego, radius_m=1.5)

    id_a = next(m.global_object_id for m in result if m.cam_id == "cam_a")
    id_b = next(m.global_object_id for m in result if m.cam_id == "cam_b")
    id_c = next(m.global_object_id for m in result if m.cam_id == "cam_c")

    assert id_a == id_b, "cam_a and cam_b should merge"
    assert id_c != id_a, "cam_c should be separate"


def test_empty_masklets_returns_empty():
    result = merge_cross_camera([], {}, {})
    assert result == []


def test_masklet_without_mask_path_still_gets_id(tmp_path):
    """Masklet with no mask_paths should still get a global_object_id (just not merged)."""
    mkl = Masklet(
        masklet_id="m1", bag_id="bag0", chunk_id="chunk0",
        cam_id="cam_front", cls="car", score=0.9,
        frames_present=[0], mask_paths=[],
    )
    calib = {"cam_front": _simple_calib()}
    world_T_ego = {"cam_front": np.eye(4, dtype=np.float64)}

    result = merge_cross_camera([mkl], calib, world_T_ego)

    assert result[0].global_object_id is not None
