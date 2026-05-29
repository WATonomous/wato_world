"""Tests for GroundingDINODetector (wato_perception_2d.detector).

All paths are exercised without requiring the transformers package to be
installed — the fallback / mock paths are tested instead.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from wato_perception_2d.detector import Detection, GroundingDINODetector


def _make_image(h: int = 64, w: int = 64) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Fallback behaviour when transformers is absent
# ---------------------------------------------------------------------------


def test_fallback_returns_empty(monkeypatch):
    """detect() returns [] and does not raise when transformers is missing."""
    # Force _load() to fail by making the import raise ImportError.
    detector = GroundingDINODetector()
    with patch.object(detector, "_load", return_value=False):
        result = detector.detect(_make_image(), ["car", "pedestrian"])
    assert result == []


def test_fallback_does_not_raise_on_empty_prompts(monkeypatch):
    detector = GroundingDINODetector()
    with patch.object(detector, "_load", return_value=False):
        result = detector.detect(_make_image(), [])
    assert result == []


# ---------------------------------------------------------------------------
# Successful detection path (mocked transformers)
# ---------------------------------------------------------------------------


def _mock_transformers():
    """Return a fake transformers module whose AutoProcessor and model work."""
    import torch

    box = torch.tensor([[10.0, 20.0, 50.0, 60.0]])
    score = torch.tensor([0.85])

    processor = MagicMock()
    processor.return_value = {"input_ids": torch.zeros(1, 5, dtype=torch.long)}
    processor.post_process_grounded_object_detection.return_value = [
        {"boxes": box, "scores": score, "labels": ["car"]}
    ]

    model_instance = MagicMock()
    model_instance.return_value = MagicMock()
    model_instance.to.return_value = model_instance

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoProcessor = MagicMock(return_value=processor)
    fake_transformers.AutoModelForZeroShotObjectDetection = MagicMock(
        return_value=model_instance
    )
    # Wire processor to be the singleton used in detect()
    fake_transformers.AutoProcessor.from_pretrained = MagicMock(return_value=processor)
    fake_transformers.AutoModelForZeroShotObjectDetection.from_pretrained = MagicMock(
        return_value=model_instance
    )
    return fake_transformers, processor


def test_detect_returns_detection_objects():
    """With mocked transformers, detect() returns Detection objects with correct types.

    Uses numpy arrays wrapped in MagicMock to avoid requiring torch at test time.
    """
    # Simulate torch tensor behaviour with numpy + MagicMock wrappers.
    box_np = np.array([[10.0, 20.0, 50.0, 60.0]], dtype=np.float32)
    score_np = np.array([0.85], dtype=np.float32)

    box_mock = MagicMock()
    box_mock.cpu.return_value.numpy.return_value = box_np

    score_mock = MagicMock()
    score_mock.cpu.return_value.numpy.return_value = score_np

    input_ids_mock = MagicMock()

    processor_mock = MagicMock()
    processor_mock.return_value = {"input_ids": input_ids_mock}
    processor_mock.post_process_grounded_object_detection.return_value = [
        {"boxes": box_mock, "scores": score_mock, "labels": ["car"]}
    ]

    model_mock = MagicMock()
    model_mock.to.return_value = model_mock
    model_mock.return_value = MagicMock()

    detector = GroundingDINODetector(model_id="fake/model", device="cpu")
    detector._processor = processor_mock
    detector._model = model_mock  # skip _load()

    # Patch the inline `import torch` inside detect() without requiring torch.
    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "torch":
            torch_mock = MagicMock()
            torch_mock.no_grad.return_value.__enter__ = MagicMock(return_value=None)
            torch_mock.no_grad.return_value.__exit__ = MagicMock(return_value=False)
            return torch_mock
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_fake_import):
        result = detector.detect(_make_image(), ["car"])

    assert len(result) == 1
    det = result[0]
    assert isinstance(det, Detection)
    assert det.bbox_xyxy.shape == (4,)
    assert det.bbox_xyxy.dtype == np.float32
    assert isinstance(det.score, float)
    assert isinstance(det.class_name, str)
    assert det.class_name == "car"
