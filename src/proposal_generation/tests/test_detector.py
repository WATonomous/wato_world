"""fusion_detector rows (detector.py) and how fusion takes them in."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wato_common.artifacts.detections_3d import Detection3D
from wato_common.artifacts.lidar_sweeps import Frame
from wato_common.schemas import Box3D
from wato_proposal_generation import detector, fusion
from wato_proposal_generation.boxes import SourceBox, proposal
from wato_proposal_generation.config import FusionConfig, FusionDetectorConfig
from wato_proposal_generation.frame_points import FramePoints

FRAME = Frame(
    frame_id=2, reference_sweep_id=4, sweep_ids=(4,), reference_timestamp_ns=40
)


def _points(xyz) -> FramePoints:
    n = len(xyz)
    return FramePoints(
        sweep_id=np.full(n, 4, np.int64),
        point_index=np.arange(n, dtype=np.int64),
        xyz=np.asarray(xyz, float),
        instance_id=np.array([""] * n),
        cls=np.array([""] * n),
        confidence=np.zeros(n, np.float32),
        cluster_id=np.full(n, -1, np.int64),
    )


def _det(k=0, **overrides) -> Detection3D:
    fields = dict(
        detection_id=f"det-000004-{k}",
        sweep_id=4,
        frame_id=2,
        reference_timestamp_ns=40,
        cx=10.0,
        cy=0.0,
        cz=1.0,
        w=2.0,
        l=4.0,
        h=1.6,
        heading=math.pi,
        vx=0.0,
        vy=0.0,
        class_id=0,
        cls="car",
        score=0.6,
        model_x=0.0,
        model_y=10.0,
        model_z=-0.8,
        model_w=2.0,
        model_l=4.0,
        model_h=1.6,
        model_rot=0.0,
        model_vx=0.0,
        model_vy=0.0,
    )
    fields.update(overrides)
    return Detection3D(**fields)


def test_kept_applies_score_class_and_the_cars_ego_zone():
    cfg = FusionDetectorConfig(enabled=True)
    assert detector.kept(_det(), cfg)
    assert not detector.kept(_det(score=0.05), cfg)
    assert not detector.kept(_det(class_id=4, cls="trailer"), cfg)
    assert not detector.kept(_det(model_x=1.0, model_y=1.0), cfg)


def test_rows_keep_the_models_box_in_the_proposal_convention():
    fp = _points([[10.0, 0.5, 1.0], [11.5, -0.5, 0.5], [20.0, 0.0, 1.0]])
    (box,) = detector.proposals(
        FRAME,
        [_det(w=4.0, l=2.0, heading=math.pi / 2), _det(1, score=0.01)],
        fp,
        FusionDetectorConfig(enabled=True),
    )
    p = box.proposal
    assert p.provenance == "fusion_detector" and p.proposal_id == "det-000004-0"
    assert (p.cx, p.cy, p.cz, p.h) == (10.0, 0.0, 1.0, 1.6)
    # l the longer side; heading undirected, along it: wider than long -> turned.
    assert (p.l, p.w) == (4.0, 2.0)
    assert p.heading == pytest.approx(0.0, abs=1e-12)
    assert box.members.tolist() == [True, True, False]
    assert p.lidar_point_count == 2 and p.cls == "car" and p.score == 0.6


def test_class_map_renames_and_restricts():
    cfg = FusionDetectorConfig(
        enabled=True, class_map={"truck": "truck", "bus": "truck"}
    )
    fp = _points([[10.0, 0.0, 1.0]])
    rows = detector.proposals(FRAME, [_det(class_id=3, cls="bus"), _det(1)], fp, cfg)
    assert [r.proposal.cls for r in rows] == ["truck"]


def _box(pid, members, provenance, cls=None, box=None, score=0.5):
    p = proposal(
        pid,
        FRAME,
        box or Box3D(0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0),
        cls=cls,
        score=score,
        provenance=provenance,
        lidar_point_count=int(np.sum(members)),
    )
    return SourceBox(p, np.asarray(members, bool))


XYZ = np.column_stack([np.arange(10.0), np.zeros(10), np.ones(10)])


def test_without_detector_boxes_fusion_is_unchanged():
    fp = _points(XYZ)
    m0 = _box("m0", [1, 1, 1, 1, 0, 0, 0, 0, 0, 0], "lidar_mos")
    f0 = _box("f0", [0, 1, 1, 1, 1, 0, 0, 0, 0, 0], "lifted", "car")
    cfg = FusionConfig(min_shared_points=2)
    assert fusion.fuse(FRAME, [m0], [f0], fp, None, cfg) == fusion.fuse(
        FRAME, [m0], [f0], fp, None, cfg, detector=[]
    )


def test_a_detector_box_joins_the_object_and_its_geometry_wins():
    fp = _points(XYZ)
    m0 = _box("m0", [1, 1, 1, 1, 0, 0, 0, 0, 0, 0], "lidar_mos")
    f0 = _box("f0", [0, 1, 1, 1, 1, 0, 0, 0, 0, 0], "lifted", "car", score=0.9)
    model_box = Box3D(2.5, 0.0, 0.9, 1.9, 4.6, 1.7, 0.1)
    d0 = _box(
        "det-000004-0",
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        "fusion_detector",
        "truck",
        model_box,
        score=0.6,
    )
    d1 = _box(
        "det-000004-1", [0, 0, 0, 0, 0, 0, 0, 0, 1, 1], "fusion_detector", "pedestrian"
    )
    rows = fusion.fuse(
        FRAME,
        [m0],
        [f0],
        fp,
        None,
        FusionConfig(min_shared_points=2),
        detector=[d0, d1],
    )
    assert [p.member_proposal_ids for p in rows] == [
        ("m0", "f0", "det-000004-0"),
        ("det-000004-1",),
    ]
    obj = rows[0]
    assert (obj.cx, obj.cy, obj.cz, obj.w, obj.l, obj.h, obj.heading) == (
        2.5,
        0.0,
        0.9,
        1.9,
        4.6,
        1.7,
        0.1,
    )
    # Lifting's class stands; the score averages the three members.
    assert obj.cls == "car" and obj.score == pytest.approx((0.5 + 0.9 + 0.6) / 3)
    assert obj.lidar_point_count == 6
    assert rows[1].cls == "pedestrian" and rows[1].provenance == "fused"


def test_unmatched_detector_boxes_can_be_left_out():
    fp = _points(XYZ)
    f0 = _box("f0", [0, 1, 1, 1, 1, 0, 0, 0, 0, 0], "lifted", None)
    d0 = _box("det-000004-0", [0, 1, 1, 1, 0, 0, 0, 0, 0, 0], "fusion_detector", "car")
    d1 = _box("det-000004-1", [0, 0, 0, 0, 0, 0, 0, 0, 1, 1], "fusion_detector", "car")
    rows = fusion.fuse(
        FRAME,
        [],
        [f0],
        fp,
        None,
        FusionConfig(min_shared_points=2),
        detector=[d0, d1],
        add_unmatched_detector=False,
    )
    assert [p.member_proposal_ids for p in rows] == [("f0", "det-000004-0")]
    # No lifted class: the detector's.
    assert rows[0].cls == "car"
