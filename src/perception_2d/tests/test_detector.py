"""Tests for the detector's pure helpers — thresholding/NMS/label-mapping logic,
exercised without loading any model.
"""

from __future__ import annotations

from wato_perception_2d.models.detector import (
    Detection,
    _box_iou,
    build_text_prompt,
    match_label_to_class,
    nms_per_class,
)


def test_box_iou_overlap_and_disjoint():
    assert _box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert _box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    # Half-overlap: intersection 50, union 150 → 1/3.
    assert _box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == 0.5 / 1.5


def test_build_text_prompt_dedups_and_formats():
    concepts = [("car", "car"), ("pickup truck", "truck"), ("car", "vehicle")]
    # lowercased, ' . '-joined, trailing ' .', duplicate "car" collapsed.
    assert build_text_prompt(concepts) == "car . pickup truck ."
    assert build_text_prompt([]) == ""


def test_match_label_to_class():
    concepts = [("car", "car"), ("pickup truck", "truck")]
    assert match_label_to_class("car", concepts) == "car"
    # substring either direction; longer prompt wins
    assert match_label_to_class("truck", concepts) == "truck"
    assert match_label_to_class("a pickup truck", concepts) == "truck"
    # off-taxonomy phrase → dropped
    assert match_label_to_class("traffic light", concepts) is None
    assert match_label_to_class("", concepts) is None


def test_nms_per_class_suppresses_same_class_only():
    # Two heavily-overlapping cars (keep the higher score) + a pedestrian that
    # overlaps a car (different class → never suppressed).
    dets = [
        Detection(box_xyxy=(0, 0, 10, 10), cls="car", score=0.9),
        Detection(box_xyxy=(1, 1, 11, 11), cls="car", score=0.8),
        Detection(box_xyxy=(0, 0, 10, 10), cls="pedestrian", score=0.7),
    ]
    kept = nms_per_class(dets, iou_threshold=0.5)
    assert len(kept) == 2
    cars = [d for d in kept if d.cls == "car"]
    assert len(cars) == 1 and cars[0].score == 0.9
    assert any(d.cls == "pedestrian" for d in kept)


def test_nms_keeps_non_overlapping_same_class():
    dets = [
        Detection(box_xyxy=(0, 0, 10, 10), cls="car", score=0.9),
        Detection(box_xyxy=(50, 50, 60, 60), cls="car", score=0.6),
    ]
    assert len(nms_per_class(dets, iou_threshold=0.5)) == 2
