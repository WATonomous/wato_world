"""Tests for the detector's pure helpers — thresholding/NMS/label-mapping logic,
exercised without loading any model.
"""

from __future__ import annotations

from wato_perception_2d.models.detector import (
    Detection,
    build_text_prompt,
    match_label_to_class,
    nms_per_class,
)


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
    # Two heavily-overlapping cars (keep the higher score), a distant car, and
    # a pedestrian that overlaps a car (different class → never suppressed).
    dets = [
        Detection(box_xyxy=(0, 0, 10, 10), cls="car", score=0.9),
        Detection(box_xyxy=(1, 1, 11, 11), cls="car", score=0.8),
        Detection(box_xyxy=(50, 50, 60, 60), cls="car", score=0.6),
        Detection(box_xyxy=(0, 0, 10, 10), cls="pedestrian", score=0.7),
    ]
    kept = nms_per_class(dets, iou_threshold=0.5)
    assert sorted((d.cls, d.score) for d in kept) == [
        ("car", 0.6),
        ("car", 0.9),
        ("pedestrian", 0.7),
    ]
