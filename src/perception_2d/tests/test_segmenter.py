"""Tests for segmentation backends (wato_perception_2d.segmenter).

All paths are exercised without requiring sam2 or sam3 to be installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from wato_perception_2d.detector import Detection
from wato_perception_2d.segmenter import (
    SAM2Segmenter,
    SAM3Segmenter,
    SegmentedDetection,
    build_segmenter,
)


def _det(x1=10, y1=10, x2=40, y2=40, cls="car", score=0.9) -> Detection:
    return Detection(
        bbox_xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
        class_name=cls,
        score=score,
    )


def _image(h=64, w=64) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# BaseSegmenter._bbox_fill_fallback (shared by all backends)
# ---------------------------------------------------------------------------


def test_bbox_fill_fallback_shape():
    H, W = 64, 64
    det = _det(10, 10, 40, 40)
    results = SAM2Segmenter._bbox_fill_fallback([det], H, W)
    assert len(results) == 1
    sd = results[0]
    assert isinstance(sd, SegmentedDetection)
    assert sd.mask.shape == (H, W)
    assert sd.mask.dtype == bool


def test_bbox_fill_fallback_filled_inside():
    H, W = 64, 64
    det = _det(10, 10, 40, 40)
    results = SAM2Segmenter._bbox_fill_fallback([det], H, W)
    mask = results[0].mask
    assert mask[15, 15]   # inside
    assert not mask[5, 5]  # outside


def test_bbox_fill_fallback_clamps_to_image():
    H, W = 32, 32
    det = _det(-5, -5, 100, 100)  # bbox exceeds image bounds
    results = SAM2Segmenter._bbox_fill_fallback([det], H, W)
    assert results[0].mask.shape == (H, W)
    assert results[0].mask.all()  # entire image filled after clamping


# ---------------------------------------------------------------------------
# SAM2Segmenter
# ---------------------------------------------------------------------------


def test_sam2_segment_empty_returns_empty():
    seg = SAM2Segmenter()
    assert seg.segment(_image(), []) == []


def test_sam2_fallback_when_missing(monkeypatch):
    seg = SAM2Segmenter()
    monkeypatch.setattr(seg, "_load", lambda: False)
    results = seg.segment(_image(), [_det()])
    assert len(results) == 1
    assert isinstance(results[0].mask, np.ndarray)
    assert results[0].mask.dtype == bool


# ---------------------------------------------------------------------------
# SAM3Segmenter
# ---------------------------------------------------------------------------


def test_sam3_segment_empty_returns_empty():
    seg = SAM3Segmenter()
    assert seg.segment(_image(), []) == []


def test_sam3_fallback_when_missing(monkeypatch):
    """SAM3 is not installed — should fall back to bbox-fill without raising."""
    seg = SAM3Segmenter()
    # _load() will return False because sam3 is not installed
    results = seg.segment(_image(), [_det()])
    assert len(results) == 1
    assert results[0].mask.dtype == bool


# ---------------------------------------------------------------------------
# build_segmenter factory
# ---------------------------------------------------------------------------


def test_build_segmenter_returns_sam2_by_default():
    seg = build_segmenter("sam2", "sam2_hiera_large")
    assert isinstance(seg, SAM2Segmenter)


def test_build_segmenter_returns_sam3():
    seg = build_segmenter("sam3", "sam3_hiera_large")
    assert isinstance(seg, SAM3Segmenter)


def test_build_segmenter_unknown_backend_defaults_to_sam2():
    seg = build_segmenter("unknown_future_backend", "some_checkpoint")
    assert isinstance(seg, SAM2Segmenter)
