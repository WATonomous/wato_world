"""Tests for DetectorEnsemble — IoU merging across multiple detectors.

Uses simple mock detectors that return predetermined Detection lists so the
ensemble's merge logic can be exercised without loading any real model.
"""

from __future__ import annotations

import numpy as np
import pytest

from wato_perception_2d.detector import (
    Detection,
    DetectorEnsemble,
)


class _MockDetector:
    """Returns a fixed list of detections regardless of input."""

    def __init__(self, name: str, detections: list[Detection]) -> None:
        self.name = name
        self._detections = detections

    def detect(
        self,
        image_rgb: np.ndarray,
        text_prompts: list[str],
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> list[Detection]:
        # Tag each detection with this mock's name (the real adapters do this too).
        out = []
        for d in self._detections:
            out.append(
                Detection(
                    bbox_xyxy=d.bbox_xyxy.copy(),
                    class_name=d.class_name,
                    score=d.score,
                    detector_name=self.name,
                )
            )
        return out


def _det(x1, y1, x2, y2, cls, score, name=None):
    return Detection(
        bbox_xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
        class_name=cls,
        score=score,
        detector_name=name,
    )


def _dummy_image():
    return np.zeros((100, 100, 3), dtype=np.uint8)


def test_ensemble_requires_at_least_one_detector():
    with pytest.raises(ValueError, match="at least one"):
        DetectorEnsemble([])


def test_single_detector_passes_through_unchanged():
    d = _MockDetector("d1", [_det(0, 0, 10, 10, "car", 0.9)])
    ens = DetectorEnsemble([d])
    out = ens.detect(_dummy_image(), ["car"])
    assert len(out) == 1
    assert out[0].class_name == "car"
    assert out[0].score == pytest.approx(0.9)


def test_two_detectors_overlapping_same_class_merges_to_one():
    d1 = _MockDetector("d1", [_det(0, 0, 10, 10, "car", 0.8)])
    d2 = _MockDetector("d2", [_det(0, 0, 11, 11, "car", 0.95)])  # ~IoU = 0.83
    ens = DetectorEnsemble([d1, d2], iou_threshold=0.6)
    out = ens.detect(_dummy_image(), ["car"])
    assert len(out) == 1
    # Higher-score box wins on greedy fusion.
    assert out[0].score == pytest.approx(0.95)
    # Provenance records both contributors.
    assert "d1" in out[0].detector_name
    assert "d2" in out[0].detector_name


def test_two_detectors_overlapping_different_class_kept_separately():
    """Same box geometry, different class → two outputs (each detector's class)."""
    d1 = _MockDetector("d1", [_det(0, 0, 10, 10, "car", 0.9)])
    d2 = _MockDetector("d2", [_det(0, 0, 10, 10, "truck", 0.9)])
    ens = DetectorEnsemble([d1, d2], iou_threshold=0.6)
    out = ens.detect(_dummy_image(), ["car", "truck"])
    classes = sorted(d.class_name for d in out)
    assert classes == ["car", "truck"]


def test_low_iou_overlap_not_merged():
    """Boxes with IoU < threshold are kept separately."""
    d1 = _MockDetector("d1", [_det(0, 0, 10, 10, "car", 0.9)])
    d2 = _MockDetector("d2", [_det(8, 8, 18, 18, "car", 0.9)])
    # IoU here is 4 / 196 ≈ 0.02 — well below default 0.6.
    ens = DetectorEnsemble([d1, d2])
    out = ens.detect(_dummy_image(), ["car"])
    assert len(out) == 2


def test_empty_detectors_produce_empty_output():
    d1 = _MockDetector("d1", [])
    d2 = _MockDetector("d2", [])
    ens = DetectorEnsemble([d1, d2])
    out = ens.detect(_dummy_image(), ["car"])
    assert out == []


def test_one_detector_empty_other_has_detection():
    d1 = _MockDetector("d1", [])
    d2 = _MockDetector("d2", [_det(0, 0, 10, 10, "car", 0.5)])
    ens = DetectorEnsemble([d1, d2])
    out = ens.detect(_dummy_image(), ["car"])
    assert len(out) == 1
    assert out[0].score == pytest.approx(0.5)


def test_three_detectors_partial_overlap():
    """d1+d2 overlap, d3 is in a different spot."""
    d1 = _MockDetector("d1", [_det(0, 0, 10, 10, "car", 0.8)])
    d2 = _MockDetector("d2", [_det(1, 1, 11, 11, "car", 0.7)])
    d3 = _MockDetector("d3", [_det(50, 50, 60, 60, "car", 0.85)])
    ens = DetectorEnsemble([d1, d2, d3], iou_threshold=0.5)
    out = ens.detect(_dummy_image(), ["car"])
    # d1+d2 → one merged detection; d3 standalone.
    assert len(out) == 2
    scores = sorted(d.score for d in out)
    assert scores == pytest.approx([0.8, 0.85])
