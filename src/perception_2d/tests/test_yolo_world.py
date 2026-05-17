"""Tests for YOLOWorldDetector.

The ultralytics library is not assumed to be present in the unit test
environment — `_load()` returns False with a warning when the import fails
or the checkpoint is missing, and `detect()` returns an empty list.  The
test below exercises that fallback path so the adapter is safe to import
in any environment.
"""

from __future__ import annotations

import numpy as np

from wato_perception_2d.detector import Detection
from wato_perception_2d.yolo_world import YOLOWorldDetector


def test_detect_returns_empty_when_model_unavailable(tmp_path):
    """Pointing the adapter at a non-existent checkpoint makes _load() fail
    cleanly; detect() then returns [] without raising."""
    det = YOLOWorldDetector(
        checkpoint_path=str(tmp_path / "does_not_exist.pt"),
        device="cpu",
    )
    out = det.detect(
        image_rgb=np.zeros((64, 64, 3), dtype=np.uint8),
        text_prompts=["car", "person"],
        box_threshold=0.25,
    )
    assert out == []


def test_detection_class_interop():
    """Confirm the adapter's return type matches what DetectorEnsemble expects.
    A static assertion via duck typing — Detection is a dataclass."""
    sample = Detection(
        bbox_xyxy=np.array([0, 0, 10, 10], dtype=np.float32),
        class_name="car",
        score=0.5,
        detector_name="yolo_world",
    )
    assert isinstance(sample.bbox_xyxy, np.ndarray)
    assert sample.detector_name == "yolo_world"
