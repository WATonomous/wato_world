"""Tests for sam3_tracker.py.

The `sam3` package is not installed in CI, so the contract under test is the
fail-loud behavior: SAM3Tracker.update() raises RuntimeError when SAM3 is
unavailable (there is no automatic IoU fallback — use tracker.backend "deva"
for the IoU tracker explicitly).  Cross-frame tracking itself is covered by the
Tracker2D suite (test_tracker.py), which is the "deva" backend.
"""

from __future__ import annotations

import numpy as np
import pytest

from wato_perception_2d.models.sam3_tracker import SAM3Tracker
from wato_perception_2d.models.segmenter import SegmentedDetection


def _tracker(tmp_path) -> SAM3Tracker:
    t = SAM3Tracker(
        bag_id="bag0",
        chunk_id="chunk0",
        cam_id="cam_front",
        masks_2d_base_dir=str(tmp_path),
    )
    t.reset()
    return t


def _sd() -> SegmentedDetection:
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:40, 10:40] = True
    return SegmentedDetection(
        phrase="car", rough_box=(10, 10, 40, 40), mask=mask, sam3_score=0.9
    )


def test_update_raises_without_sam3(tmp_path):
    """No sam3 installed → update() raises a clear RuntimeError, no fallback."""
    tracker = _tracker(tmp_path)
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="SAM 3.1 video predictor"):
        tracker.update(0, img, [_sd()])


def test_finalize_without_tracking_returns_empty(tmp_path):
    """finalize() on an unused tracker is model-free and returns no masklets."""
    assert _tracker(tmp_path).finalize() == []


def test_reset_is_model_free(tmp_path):
    """reset() must not touch SAM3 (so streams can be set up before loading)."""
    tracker = _tracker(tmp_path)
    tracker.reset()  # should not raise
    assert tracker.finalize() == []
