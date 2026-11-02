"""Tests for FixedClassDiscovery — the Florence-2 bypass (closed-set vocab)."""

from __future__ import annotations

import numpy as np

from wato_perception_2d.models.discovery import FixedClassDiscovery


def test_one_full_image_proposal_per_named_class():
    disco = FixedClassDiscovery(["car", "", "  ", "bus"])
    proposals = disco.propose(np.zeros((480, 640, 3), dtype=np.uint8))

    assert [p.phrase for p in proposals] == ["car", "bus"]
    assert all(p.confidence == 1.0 for p in proposals)
    assert proposals[0].rough_box == (0.0, 0.0, 640.0, 480.0)
