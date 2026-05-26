"""Tests for tracker_2d.py — IoU matching, gap closure, mask persistence."""

from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image as PILImage

from wato_perception_2d.segmenter import SegmentedDetection
from wato_perception_2d.tracker_2d import Tracker2D


def _mask(H: int, W: int, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    m = np.zeros((H, W), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


def _sd(
    mask: np.ndarray,
    x1: float = 0,
    y1: float = 0,
    x2: float = 10,
    y2: float = 10,
    cls: str = "car",
    score: float = 0.9,
) -> SegmentedDetection:
    return SegmentedDetection(
        phrase=cls,
        rough_box=(x1, y1, x2, y2),
        mask=mask,
        sam3_score=score,
        discovery_score=1.0,
    )


def test_single_detection_creates_one_masklet(tmp_path):
    """First detection → exactly one masklet."""
    tracker = Tracker2D("bag0", "chunk0", "cam_front", str(tmp_path))
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    m = _mask(100, 100, 10, 10, 40, 40)

    tracker.update(0, img, [_sd(m, 10, 10, 40, 40)])
    masklets = tracker.finalize()

    assert len(masklets) == 1
    mkl = masklets[0]
    assert mkl.cam_id == "cam_front"
    assert mkl.cls == "car"
    assert mkl.frames_present == [0]
    assert mkl.score == pytest.approx(0.9)


def test_same_mask_position_tracked_across_frames(tmp_path):
    """Same mask position across 3 frames → 1 masklet spanning all 3."""
    tracker = Tracker2D("bag0", "chunk0", "cam_front", str(tmp_path))
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    m = _mask(100, 100, 10, 10, 50, 50)
    sd = _sd(m, 10, 10, 50, 50)

    for seq in [0, 1, 2]:
        tracker.update(seq, img, [sd])
    masklets = tracker.finalize()

    assert len(masklets) == 1
    assert masklets[0].frames_present == [0, 1, 2]
    assert len(masklets[0].mask_paths) == 3


def test_non_overlapping_masks_are_separate_tracks(tmp_path):
    """Two masks with zero IoU → two separate masklets."""
    tracker = Tracker2D("bag0", "chunk0", "cam_front", str(tmp_path))
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    m_a = _mask(100, 100, 0, 0, 10, 10)    # top-left
    m_b = _mask(100, 100, 80, 80, 99, 99)  # bottom-right, no overlap

    tracker.update(0, img, [_sd(m_a), _sd(m_b)])
    masklets = tracker.finalize()

    assert len(masklets) == 2
    assert len({m.masklet_id for m in masklets}) == 2


def test_gap_closes_track_after_max_gap_frames(tmp_path):
    """
    Detection at frame 0, then 4 empty frames.
    MAX_GAP_FRAMES=3, so the track closes during the empty period.
    """
    tracker = Tracker2D("bag0", "chunk0", "cam_front", str(tmp_path))
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    m = _mask(100, 100, 10, 10, 40, 40)

    tracker.update(0, img, [_sd(m, 10, 10, 40, 40)])
    for seq in range(1, 5):  # 4 empty frames
        tracker.update(seq, img, [])

    masklets = tracker.finalize()

    assert len(masklets) == 1
    assert masklets[0].frames_present == [0]


def test_track_survives_within_max_gap(tmp_path):
    """Detection at 0, gap of 2 frames, detection at 3 → same tracklet."""
    tracker = Tracker2D("bag0", "chunk0", "cam_front", str(tmp_path))
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    m = _mask(100, 100, 10, 10, 40, 40)
    sd = _sd(m, 10, 10, 40, 40)

    tracker.update(0, img, [sd])
    tracker.update(1, img, [])
    tracker.update(2, img, [])
    tracker.update(3, img, [sd])
    masklets = tracker.finalize()

    assert len(masklets) == 1
    assert 0 in masklets[0].frames_present
    assert 3 in masklets[0].frames_present


def test_mask_png_written_and_readable(tmp_path):
    """Mask saved to PNG should round-trip through PIL without data loss."""
    tracker = Tracker2D("bag0", "chunk0", "cam_front", str(tmp_path))
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    m = _mask(100, 100, 20, 30, 60, 70)

    tracker.update(5, img, [_sd(m, 20, 30, 60, 70)])
    masklets = tracker.finalize()

    assert len(masklets[0].mask_paths) == 1
    path = masklets[0].mask_paths[0]
    assert os.path.exists(path)

    loaded = np.array(PILImage.open(path)).astype(bool)
    np.testing.assert_array_equal(loaded, m)


def test_masklet_id_in_path(tmp_path):
    """PNG path should contain the masklet_id as a directory component."""
    tracker = Tracker2D("bag0", "chunk0", "cam_front", str(tmp_path))
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    m = _mask(100, 100, 10, 10, 40, 40)

    tracker.update(0, img, [_sd(m, 10, 10, 40, 40)])
    masklets = tracker.finalize()

    mkl = masklets[0]
    assert mkl.masklet_id in mkl.mask_paths[0]


def test_two_cameras_independent_tracks(tmp_path):
    """Tracker instances are per-camera; cam_ids must not collide."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    m = _mask(100, 100, 10, 10, 40, 40)

    t1 = Tracker2D("bag0", "chunk0", "cam_front", str(tmp_path / "t1"))
    t2 = Tracker2D("bag0", "chunk0", "cam_back", str(tmp_path / "t2"))

    for t in (t1, t2):
        t.update(0, img, [_sd(m)])
        t.update(1, img, [_sd(m)])

    m1 = t1.finalize()
    m2 = t2.finalize()

    assert m1[0].cam_id == "cam_front"
    assert m2[0].cam_id == "cam_back"
