"""Duplicate suppression (CenterPoint's circle NMS) over one frame's boxes."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from wato_common.artifacts.proposals import Proposal
from wato_tracking.config import DuplicatesConfig, StrongConfig
from wato_tracking.detections import Detection, frames, suppress_duplicates


def _det(cx, cy=0.0, cls="car", strong=False, points=40, pid="p") -> Detection:
    p = Proposal(
        proposal_id=pid,
        sweep_id=0,
        frame_id=0,
        sweep_ids=(0,),
        reference_timestamp_ns=0,
        cx=cx,
        cy=cy,
        cz=0.8,
        w=1.8,
        l=4.4,
        h=1.6,
        heading=0.0,
        cls=cls,
        score=1.0,
        provenance="fused",
        lidar_point_count=points,
        member_proposal_ids=(),
    )
    return Detection(p, strong=strong, key=None, two_source=False, has_lidar_mos=False)


def _ids(dets):
    return [d.proposal.proposal_id for d in dets]


def test_a_weak_box_beside_a_strong_one_of_its_class_is_dropped():
    cfg = DuplicatesConfig()
    dets = [
        _det(0.9, pid="weak"),
        _det(0.0, strong=True, pid="strong"),
        _det(2.5, pid="neighbour"),  # 2.5 m: beyond the car radius of 2 m
    ]
    assert _ids(suppress_duplicates(dets, cfg)) == ["strong", "neighbour"]


def test_equal_strength_keeps_the_box_with_more_points():
    cfg = DuplicatesConfig()
    dets = [_det(0.0, points=10, pid="few"), _det(1.0, points=80, pid="many")]
    assert _ids(suppress_duplicates(dets, cfg)) == ["many"]


def test_other_groups_and_classless_boxes_are_never_suppressed():
    cfg = DuplicatesConfig()
    dets = [
        _det(0.0, strong=True, pid="car"),
        _det(0.3, cls="pedestrian", pid="ped"),  # no pedestrian group shipped
        _det(0.5, cls="truck", pid="truck"),  # its own group
        _det(0.4, cls=None, pid="mos"),
    ]
    assert _ids(suppress_duplicates(dets, cfg)) == ["car", "ped", "truck", "mos"]


def test_groups_share_one_radius_across_their_classes():
    cfg = DuplicatesConfig(
        groups=[{"classes": ["motorcycle", "bicycle"], "min_radius": 0.85}]
    )
    dets = [
        _det(0.0, cls="motorcycle", strong=True, pid="moto"),
        _det(0.9, cls="bicycle", pid="bike"),  # 0.81 m² ≤ 0.85
        _det(1.0, cls="bicycle", pid="far"),  # 1.0 m² > 0.85
    ]
    assert _ids(suppress_duplicates(dets, cfg)) == ["moto", "far"]


def test_a_class_in_two_groups_is_rejected():
    with pytest.raises(ValidationError):
        DuplicatesConfig(
            groups=[
                {"classes": ["car"], "min_radius": 4.0},
                {"classes": ["car", "truck"], "min_radius": 12.0},
            ]
        )


def test_frames_applies_suppression_only_when_enabled():
    rows = [
        _det(0.0, pid="fused-a").proposal,
        _det(1.0, pid="fused-b").proposal,
    ]
    on = frames(rows, StrongConfig(), DuplicatesConfig())
    off = frames(rows, StrongConfig(), DuplicatesConfig(enabled=False))
    assert len(on[0].dets) == 1 and len(off[0].dets) == 2
