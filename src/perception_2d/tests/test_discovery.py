"""Tests for FixedClassDiscovery — the Florence-2 bypass (closed-set vocab)."""

from __future__ import annotations

import numpy as np

from wato_perception_2d.models.discovery import FixedClassDiscovery


def _img(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_one_proposal_per_class():
    classes = ["car", "truck", "person"]
    disco = FixedClassDiscovery(classes)
    proposals = disco.propose(_img())

    assert [p.phrase for p in proposals] == classes
    assert all(p.confidence == 1.0 for p in proposals)


def test_box_spans_full_image():
    disco = FixedClassDiscovery(["car"])
    proposals = disco.propose(_img(h=480, w=640))

    assert proposals[0].rough_box == (0.0, 0.0, 640.0, 480.0)


def test_empty_classes_yields_no_proposals():
    disco = FixedClassDiscovery([])
    assert disco.propose(_img()) == []


def test_blank_class_names_are_dropped():
    disco = FixedClassDiscovery(["car", "", "  ", "bus"])
    proposals = disco.propose(_img())
    assert [p.phrase for p in proposals] == ["car", "bus"]
