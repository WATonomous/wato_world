"""Duplicate suppression (CenterPoint's circle NMS) over one frame's boxes."""

from __future__ import annotations

import pytest
from dataclasses import replace
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


def _member(pid, provenance, score=0.5, **kw) -> Proposal:
    return Proposal(
        proposal_id=pid,
        sweep_id=0,
        frame_id=0,
        sweep_ids=(0,),
        reference_timestamp_ns=0,
        cx=0.0,
        cy=0.0,
        cz=0.8,
        w=1.8,
        l=4.4,
        h=1.6,
        heading=0.0,
        cls="car",
        score=score,
        provenance=provenance,
        lidar_point_count=30,
        **kw,
    )


@pytest.mark.parametrize("score, strong", [(0.49, False), (0.5, True), (0.9, True)])
def test_a_confident_detector_member_makes_the_box_strong(score, strong):
    from wato_tracking.detections import detection

    det = _member("det-000000-0", "fusion_detector", score=score)
    fused = _member("fused-000000-0", "fused", member_proposal_ids=(det.proposal_id,))
    d = detection(fused, {det.proposal_id: det}, StrongConfig())
    assert d.strong is strong and d.detector_score == score
    assert d.key is None and not d.two_source


def test_detector_and_lift_agreeing_are_not_two_sources():
    from wato_tracking.detections import detection

    lift = _member(
        "lifted-m3-000000", "lifted", lift_confidence=0.1, lift_seed_point_count=5
    )
    det = _member("det-000000-0", "fusion_detector", score=0.2)
    fused = _member(
        "fused-000000-0",
        "fused",
        member_proposal_ids=(lift.proposal_id, det.proposal_id),
    )
    d = detection(fused, {p.proposal_id: p for p in (lift, det)}, StrongConfig())
    assert not d.two_source and not d.strong and d.key == "m3"


@pytest.mark.parametrize("score, can_birth", [(0.59, False), (0.6, True)])
def test_a_detector_only_box_starts_a_track_only_when_sure(score, can_birth):
    from wato_tracking.detections import detection

    det = _member("det-000000-0", "fusion_detector", score=score)
    fused = _member("fused-000000-0", "fused", member_proposal_ids=(det.proposal_id,))
    assert (
        detection(fused, {det.proposal_id: det}, StrongConfig()).can_birth is can_birth
    )


def test_other_evidence_lets_a_low_scored_detector_box_start_a_track():
    from wato_tracking.detections import detection

    lift = _member(
        "lifted-m3-000000", "lifted", lift_confidence=0.1, lift_seed_point_count=5
    )
    det = _member("det-000000-0", "fusion_detector", score=0.1)
    fused = _member(
        "fused-000000-0",
        "fused",
        member_proposal_ids=(lift.proposal_id, det.proposal_id),
    )
    assert detection(
        fused, {p.proposal_id: p for p in (lift, det)}, StrongConfig()
    ).can_birth


@pytest.mark.parametrize("source", ["lifted", "lidar_mos"])
def test_a_detector_cannot_promote_weak_independent_evidence(source):
    from wato_tracking.detections import detection

    independent = _member(
        "lifted-m3-000000" if source == "lifted" else "mos-0",
        source,
        score=0.1,
        lift_confidence=0.1,
    )
    det = _member("det-0", "fusion_detector", score=0.99)
    fused = _member(
        "fused-0",
        "fused",
        member_proposal_ids=(independent.proposal_id, det.proposal_id),
    )
    d = detection(fused, {p.proposal_id: p for p in (independent, det)}, StrongConfig())
    assert not d.strong
    assert not d.two_source


def test_detector_only_box_cannot_suppress_independent_geometry():
    primary = _det(0.0, points=5, pid="primary")
    supplemental = replace(
        _det(0.5, strong=True, points=500, pid="detector"), detector_score=0.99
    )
    assert _ids(suppress_duplicates([supplemental, primary], DuplicatesConfig())) == [
        "primary"
    ]


def test_detector_only_box_cannot_take_a_track_from_independent_evidence():
    from wato_tracking import tracker
    from wato_tracking.config import AssociationConfig
    from wato_tracking.detections import DetFrame

    initial = _det(0.0, pid="initial")
    primary = _det(0.8, pid="primary")
    supplemental = replace(_det(0.1, strong=True, pid="detector"), detector_score=0.99)
    tracks = tracker.run(
        [
            DetFrame(0, 0, (initial,)),
            DetFrame(1, 100_000_000, (primary, supplemental)),
        ],
        AssociationConfig(),
    )
    assert [o.det.proposal.proposal_id for o in tracks[0]] == ["initial", "primary"]
