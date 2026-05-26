"""Tests for phrase_dedup.py — synonym coarsening and NMS."""

from __future__ import annotations

import numpy as np
import pytest

from wato_perception_2d.phrase_dedup import (
    _exact_string_dedup,
    _mask_iou,
    coarsen_synonyms,
    nms_2d_fallback,
    nms_3d,
)
from wato_perception_2d.segmenter import SegmentedDetection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mask(H: int, W: int, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    m = np.zeros((H, W), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


def _det(
    phrase: str,
    mask: np.ndarray,
    sam3_score: float = 0.5,
    rough_box: tuple = (0, 0, 10, 10),
) -> SegmentedDetection:
    return SegmentedDetection(
        phrase=phrase,
        rough_box=rough_box,
        mask=mask,
        sam3_score=sam3_score,
        discovery_score=1.0,
    )


# ---------------------------------------------------------------------------
# _exact_string_dedup
# ---------------------------------------------------------------------------


def test_exact_dedup_identical_phrase_keeps_highest_score():
    m = _make_mask(100, 100, 10, 10, 50, 50)
    dets = [_det("car", m, 0.7), _det("car", m, 0.9), _det("car", m, 0.3)]
    result = _exact_string_dedup(dets)
    assert len(result) == 1
    assert result[0].sam3_score == pytest.approx(0.9)


def test_exact_dedup_different_phrases_all_kept():
    m = _make_mask(100, 100, 0, 0, 50, 50)
    dets = [_det("car", m), _det("truck", m), _det("pedestrian", m)]
    result = _exact_string_dedup(dets)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# coarsen_synonyms — without CLIP (exact-string fallback)
# ---------------------------------------------------------------------------


def test_coarsen_synonyms_empty():
    assert coarsen_synonyms([]) == []


def test_coarsen_synonyms_single():
    m = _make_mask(100, 100, 0, 0, 50, 50)
    dets = [_det("car", m)]
    assert len(coarsen_synonyms(dets)) == 1


def test_coarsen_synonyms_fallback_deduplicates_exact():
    """CLIP will be unavailable in test env; falls back to exact-string dedup."""
    m = _make_mask(100, 100, 0, 0, 50, 50)
    dets = [
        _det("car", m, sam3_score=0.6),
        _det("car", m, sam3_score=0.9),
        _det("truck", m, sam3_score=0.5),
    ]
    result = coarsen_synonyms(dets)
    # Both unique phrases kept; "car" deduped to highest score.
    phrases = {d.phrase for d in result}
    assert "car" in phrases
    assert "truck" in phrases
    car_det = next(d for d in result if d.phrase == "car")
    assert car_det.sam3_score == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# nms_2d_fallback
# ---------------------------------------------------------------------------


def test_nms_2d_same_region_keeps_highest_score():
    """Two fully overlapping masks → only the higher-scored survives."""
    m = _make_mask(100, 100, 10, 10, 80, 80)
    dets = [_det("car", m, 0.6), _det("vehicle", m, 0.9)]
    result = nms_2d_fallback(dets, iou_threshold=0.5)
    assert len(result) == 1
    assert result[0].sam3_score == pytest.approx(0.9)


def test_nms_2d_non_overlapping_both_kept():
    """Two spatially separated masks → both kept."""
    left = _make_mask(100, 100, 0, 0, 40, 40)
    right = _make_mask(100, 100, 60, 60, 100, 100)
    dets = [_det("car", left, 0.8), _det("truck", right, 0.7)]
    result = nms_2d_fallback(dets, iou_threshold=0.5)
    assert len(result) == 2


def test_nms_2d_partial_overlap_below_threshold_both_kept():
    m1 = _make_mask(100, 100, 0, 0, 60, 60)
    m2 = _make_mask(100, 100, 50, 50, 100, 100)
    iou = _mask_iou(m1, m2)
    assert iou < 0.5  # verify the test setup
    result = nms_2d_fallback([_det("a", m1, 0.8), _det("b", m2, 0.7)], iou_threshold=0.5)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# nms_3d — zero-depth fallback
# ---------------------------------------------------------------------------


def test_nms_3d_zero_depth_falls_back_to_2d():
    """When depth map is all zeros, nms_3d delegates to nms_2d_fallback."""
    m = _make_mask(100, 100, 10, 10, 80, 80)
    dets = [_det("car", m, 0.6), _det("vehicle", m, 0.9)]
    depth = np.zeros((100, 100), dtype=np.float32)
    K = np.eye(3)
    world_T_ego = np.eye(4)
    ego_T_cam = np.eye(4)
    result = nms_3d(dets, depth, K, world_T_ego, ego_T_cam, radius_m=1.0)
    assert len(result) == 1
    assert result[0].sam3_score == pytest.approx(0.9)


def test_nms_3d_spatially_distinct_both_kept():
    """Two detections with centroids > radius_m apart → both kept."""
    # Left detection at pixel (20, 50), right at (80, 50).
    left_mask = _make_mask(100, 100, 10, 40, 30, 60)
    right_mask = _make_mask(100, 100, 70, 40, 90, 60)

    # Depth map: give each centroid ~10m depth.
    depth = np.full((100, 100), 10.0, dtype=np.float32)

    # Simple camera: fx=fy=1, cx=cy=0 so world position ≈ (u*d, v*d, d).
    # Left centroid ≈ (20, 50, 10) world; right ≈ (80, 50, 10) world.
    # Distance ≈ 60 m >> radius_m=1.0 → both kept.
    K = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    world_T_ego = np.eye(4)
    ego_T_cam = np.eye(4)

    dets = [_det("car", left_mask, 0.8), _det("truck", right_mask, 0.7)]
    result = nms_3d(dets, depth, K, world_T_ego, ego_T_cam, radius_m=1.0)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# _mask_iou
# ---------------------------------------------------------------------------


def test_mask_iou_identical():
    m = _make_mask(50, 50, 5, 5, 40, 40)
    assert _mask_iou(m, m) == pytest.approx(1.0)


def test_mask_iou_no_overlap():
    a = _make_mask(50, 50, 0, 0, 20, 20)
    b = _make_mask(50, 50, 30, 30, 50, 50)
    assert _mask_iou(a, b) == pytest.approx(0.0)


def test_mask_iou_empty():
    empty = np.zeros((50, 50), dtype=bool)
    assert _mask_iou(empty, empty) == pytest.approx(0.0)
