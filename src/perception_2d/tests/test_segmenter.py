"""Tests for SAM2Segmenter (wato_perception_2d.segmenter).

All paths are exercised without requiring sam2 to be installed.
"""

from __future__ import annotations

import numpy as np

from wato_perception_2d.detector import Detection
from wato_perception_2d.segmenter import SAM2Segmenter, SegmentedDetection


def _det(x1=10, y1=10, x2=40, y2=40, cls="car", score=0.9) -> Detection:
    return Detection(
        bbox_xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
        class_name=cls,
        score=score,
    )


def _image(h=64, w=64) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


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
    assert mask[15, 15]  # inside
    assert not mask[5, 5]  # outside


def test_bbox_fill_fallback_clamps_to_image():
    H, W = 32, 32
    det = _det(-5, -5, 100, 100)  # bbox exceeds image bounds
    results = SAM2Segmenter._bbox_fill_fallback([det], H, W)
    assert results[0].mask.shape == (H, W)
    assert results[0].mask.all()  # entire image filled after clamping


def test_segment_empty_returns_empty():
    seg = SAM2Segmenter()
    assert seg.segment(_image(), []) == []


def test_fallback_when_sam2_missing(monkeypatch):
    seg = SAM2Segmenter()
    monkeypatch.setattr(seg, "_load", lambda: False)
    results = seg.segment(_image(), [_det()])
    assert len(results) == 1
    assert isinstance(results[0].mask, np.ndarray)
    assert results[0].mask.dtype == bool
