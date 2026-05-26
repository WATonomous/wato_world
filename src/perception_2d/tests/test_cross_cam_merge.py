"""Tests for cross_cam_merge.py — world-position lifting and Union-Find clustering."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image as PILImage

from wato_perception_2d.cross_cam_merge import (
    _load_depth_at,
    _mask_centroid_px,
    merge_cross_camera,
)
from wato_perception_2d.io import CalibrationInfo
from wato_perception_2d.tracker_2d import Masklet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_calib(
    fx: float = 500.0, cx: float = 320.0, cy: float = 240.0
) -> CalibrationInfo:
    K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=np.float64)
    ego_T_cam = np.eye(4, dtype=np.float64)
    return CalibrationInfo(K=K, ego_T_cam=ego_T_cam)


def _masklet_with_png(
    tmp_path, masklet_id: str, cam_id: str, mask: np.ndarray, frame_seq: int = 0
) -> Masklet:
    """Write the mask as a PNG and return a Masklet pointing to it."""
    d = tmp_path / masklet_id
    d.mkdir(exist_ok=True)
    png = str(d / f"{frame_seq:06d}.png")
    PILImage.fromarray(mask.astype(np.uint8) * 255).save(png)
    return Masklet(
        masklet_id=masklet_id,
        bag_id="bag0",
        chunk_id="chunk0",
        cam_id=cam_id,
        cls="car",
        score=0.9,
        frames_present=[frame_seq],
        mask_paths=[png],
    )


def _write_depth_artifact(monkeypatch, tmp_path, cam_id: str, depth_m: float) -> None:
    """Write a synthetic depth artifact so _load_depth_at can find it."""
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")
    from wato_common.artifact_store import depth_2d_path, local_path
    import os

    path = local_path(depth_2d_path("bag0", "chunk0", cam_id, 0))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    depth_arr = np.full((480, 640), depth_m, dtype=np.float16)
    np.savez_compressed(path, depth_m=depth_arr, fit_status=np.int32(0))


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


def test_load_depth_at_missing_returns_none(tmp_path, monkeypatch):
    """No depth artifact on disk → returns None."""
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")
    result = _load_depth_at("bag0", "chunk0", "cam_front", 0, 320.0, 240.0)
    assert result is None


def test_load_depth_at_reads_pixel_value(tmp_path, monkeypatch):
    """Depth at a specific pixel is returned correctly."""
    _write_depth_artifact(monkeypatch, tmp_path, "cam_front", 15.0)
    result = _load_depth_at("bag0", "chunk0", "cam_front", 0, 320.0, 240.0)
    assert result == pytest.approx(15.0, abs=0.1)


def test_load_depth_at_fit_failed_returns_none(tmp_path, monkeypatch):
    """fit_status == 2 (failed) → returns None."""
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")
    from wato_common.artifact_store import depth_2d_path, local_path
    import os

    path = local_path(depth_2d_path("bag0", "chunk0", "cam_front", 0))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    depth_arr = np.full((480, 640), 10.0, dtype=np.float16)
    np.savez_compressed(path, depth_m=depth_arr, fit_status=np.int32(2))

    result = _load_depth_at("bag0", "chunk0", "cam_front", 0, 320.0, 240.0)
    assert result is None


# ---------------------------------------------------------------------------
# Integration tests for merge_cross_camera
# ---------------------------------------------------------------------------


def test_same_camera_masklets_never_merged(tmp_path, monkeypatch):
    """
    Two masklets from the same camera — even at identical world positions —
    must never share a global_object_id (they cannot be cross-camera duplicates).
    """
    H, W = 480, 640
    mask = np.zeros((H, W), dtype=bool)
    mask[230:250, 310:330] = True

    mkl1 = _masklet_with_png(tmp_path, "m1", "cam_front", mask)
    mkl2 = _masklet_with_png(tmp_path, "m2", "cam_front", mask)

    calib = {"cam_front": _simple_calib()}
    world_T_ego = {"cam_front": np.eye(4, dtype=np.float64)}

    result = merge_cross_camera(
        [mkl1, mkl2], calib, world_T_ego, bag_id="bag0", chunk_id="chunk0"
    )
    assert result[0].global_object_id != result[1].global_object_id


def test_different_cameras_same_world_position_merged(tmp_path, monkeypatch):
    """
    Two cameras with identical calibration, ego at world origin, centroid at
    principal point, and depth 20 m → both back-project to (0, 0, 20) → merged.
    """
    H, W = 480, 640
    mask = np.zeros((H, W), dtype=bool)
    mask[230:250, 310:330] = True  # centroid ≈ (320, 240) = principal point

    _write_depth_artifact(monkeypatch, tmp_path, "cam_front", 20.0)
    _write_depth_artifact(monkeypatch, tmp_path, "cam_back", 20.0)

    mkl1 = _masklet_with_png(tmp_path, "m1", "cam_front", mask)
    mkl2 = _masklet_with_png(tmp_path, "m2", "cam_back", mask)

    calib = {"cam_front": _simple_calib(), "cam_back": _simple_calib()}
    world_T_ego = {
        "cam_front": np.eye(4, dtype=np.float64),
        "cam_back": np.eye(4, dtype=np.float64),
    }

    result = merge_cross_camera(
        [mkl1, mkl2], calib, world_T_ego, bag_id="bag0", chunk_id="chunk0",
        radius_m=1.5,
    )
    assert result[0].global_object_id == result[1].global_object_id


def test_different_cameras_far_world_positions_not_merged(tmp_path, monkeypatch):
    """
    cam_front centroid ≈ (320, 240) → world ≈ (0, 0, 20);
    cam_back centroid at (5, 5) → world ≈ (-12.6, -9.4, 20) → not merged.
    """
    H, W = 480, 640
    mask_center = np.zeros((H, W), dtype=bool)
    mask_center[230:250, 310:330] = True

    mask_corner = np.zeros((H, W), dtype=bool)
    mask_corner[0:10, 0:10] = True

    _write_depth_artifact(monkeypatch, tmp_path, "cam_front", 20.0)
    _write_depth_artifact(monkeypatch, tmp_path, "cam_back", 20.0)

    mkl1 = _masklet_with_png(tmp_path, "m1", "cam_front", mask_center)
    mkl2 = _masklet_with_png(tmp_path, "m2", "cam_back", mask_corner)

    calib = {"cam_front": _simple_calib(), "cam_back": _simple_calib()}
    world_T_ego = {
        "cam_front": np.eye(4, dtype=np.float64),
        "cam_back": np.eye(4, dtype=np.float64),
    }

    result = merge_cross_camera(
        [mkl1, mkl2], calib, world_T_ego, bag_id="bag0", chunk_id="chunk0",
        radius_m=1.5,
    )
    assert result[0].global_object_id != result[1].global_object_id


def test_three_cameras_two_merged_one_separate(tmp_path, monkeypatch):
    """cam_a and cam_b see the same object (merged); cam_c sees a distant one."""
    H, W = 480, 640
    mask_center = np.zeros((H, W), dtype=bool)
    mask_center[230:250, 310:330] = True

    mask_far = np.zeros((H, W), dtype=bool)
    mask_far[0:10, 0:10] = True

    for cam in ("cam_a", "cam_b", "cam_c"):
        _write_depth_artifact(monkeypatch, tmp_path, cam, 20.0)

    mkl_a = _masklet_with_png(tmp_path, "ma", "cam_a", mask_center)
    mkl_b = _masklet_with_png(tmp_path, "mb", "cam_b", mask_center)
    mkl_c = _masklet_with_png(tmp_path, "mc", "cam_c", mask_far)

    calib = {k: _simple_calib() for k in ("cam_a", "cam_b", "cam_c")}
    world_T_ego = {k: np.eye(4, dtype=np.float64) for k in ("cam_a", "cam_b", "cam_c")}

    result = merge_cross_camera(
        [mkl_a, mkl_b, mkl_c], calib, world_T_ego,
        bag_id="bag0", chunk_id="chunk0", radius_m=1.5,
    )

    id_a = next(m.global_object_id for m in result if m.cam_id == "cam_a")
    id_b = next(m.global_object_id for m in result if m.cam_id == "cam_b")
    id_c = next(m.global_object_id for m in result if m.cam_id == "cam_c")

    assert id_a == id_b, "cam_a and cam_b should merge"
    assert id_c != id_a, "cam_c should be separate"


def test_empty_masklets_returns_empty():
    result = merge_cross_camera([], {}, {}, bag_id="bag0", chunk_id="chunk0")
    assert result == []


def test_masklet_without_mask_path_still_gets_id(tmp_path, monkeypatch):
    """Masklet with no mask_paths should still get a global_object_id."""
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")
    mkl = Masklet(
        masklet_id="m1",
        bag_id="bag0",
        chunk_id="chunk0",
        cam_id="cam_front",
        cls="car",
        score=0.9,
        frames_present=[0],
        mask_paths=[],
    )
    calib = {"cam_front": _simple_calib()}
    world_T_ego = {"cam_front": np.eye(4, dtype=np.float64)}

    result = merge_cross_camera(
        [mkl], calib, world_T_ego, bag_id="bag0", chunk_id="chunk0"
    )
    assert result[0].global_object_id is not None
