"""The sources and fusion on hand-built frames."""

from __future__ import annotations

import numpy as np
import pytest

from wato_common.artifacts.lidar_sweeps import CanonicalLidarError, Frame
from wato_common.artifacts.lifted_labels import LiftedInstance
from wato_common.artifacts.motion_proposals import MotionCluster
from wato_proposal_generation import fusion, lidar_mos, lifted
from wato_proposal_generation.config import (
    FusionConfig,
    LidarMosConfig,
    LiftedConfig,
)
from wato_proposal_generation.frame_points import FramePoints

FRAME = Frame(
    frame_id=0, reference_sweep_id=4, sweep_ids=(4, 5), reference_timestamp_ns=100
)


def _cluster(cluster_id=0, **overrides) -> MotionCluster:
    fields = dict(
        frame_id=0,
        reference_timestamp_ns=100,
        sweep_ids=(4, 5),
        cluster_id=cluster_id,
        track_hint_id=0,
        cx=0.0,
        cy=0.0,
        cz=1.0,
        w=1.0,
        l=1.0,
        h=1.0,
        heading=0.0,
        n_points=10,
        n_seed_points=10,
        source_bits=1,
        n_sources=2,
        frac_seg_dynamic=0.5,
        frac_persistent=0.1,
        track_life=5,
        net_displacement_m=3.0,
        motion_score=1.5,
        box_filled=True,
    )
    fields.update(overrides)
    return MotionCluster(**fields)


def _points(xyz, instance_id=None, cls=None, cluster_id=None, confidence=0.9):
    n = len(xyz)
    inst = np.array(instance_id if instance_id is not None else [""] * n)
    return FramePoints(
        sweep_id=np.full(n, 4, np.int64),
        point_index=np.arange(n, dtype=np.int64),
        xyz=np.asarray(xyz, float),
        instance_id=inst,
        cls=np.array(cls if cls is not None else [""] * n),
        confidence=np.where(inst != "", confidence, 0.0).astype(np.float32),
        cluster_id=np.array(
            cluster_id if cluster_id is not None else [-1] * n, np.int64
        ),
    )


INSTANCES = {
    "mA": LiftedInstance("mA", "car", ("mA",), ("CAM_A",), 10, 1),
    "mB": LiftedInstance("mB", "truck", ("mB",), ("CAM_B",), 10, 1),
}


def test_gate_keeps_only_moving_multi_source_transient_clusters():
    cfg = LidarMosConfig()
    kept = lidar_mos.gate(
        [
            _cluster(0),
            _cluster(1, motion_score=0.9),
            _cluster(2, n_sources=1),
            _cluster(3, frac_persistent=0.6),
        ],
        cfg,
    )
    assert [c.cluster_id for c in kept] == [0]


def test_by_frame_rejects_clusters_stamped_off_the_frames():
    assert lidar_mos.by_frame([_cluster()], [FRAME]) == {4: [_cluster()]}
    with pytest.raises(CanonicalLidarError):
        lidar_mos.by_frame([_cluster(reference_timestamp_ns=105)], [FRAME])
    with pytest.raises(CanonicalLidarError):
        lidar_mos.by_frame([_cluster(sweep_ids=(4, 6))], [FRAME])


def test_cluster_class_is_a_confidence_weighted_vote_with_a_floor():
    # 6 points in the cluster: 3 car (conf 0.9), 1 truck; 2 unlabelled.
    xyz = np.zeros((6, 3))
    fp = _points(
        xyz,
        instance_id=["mA", "mA", "mA", "mB", "", ""],
        cls=["car", "car", "car", "truck", "", ""],
        cluster_id=[0] * 6,
    )
    [box] = lidar_mos.proposals(
        FRAME, [_cluster()], fp, None, INSTANCES, LidarMosConfig()
    )
    p = box.proposal
    assert p.cls == "car" and p.supporting_cam_ids == ("CAM_A",)
    assert p.lidar_point_count == 6 and p.score == pytest.approx(0.75)
    assert (p.motion_score, p.n_sources, p.track_life) == (1.5, 2, 5)

    strict = LidarMosConfig(min_labeled_fraction=0.9)
    [box] = lidar_mos.proposals(FRAME, [_cluster()], fp, None, INSTANCES, strict)
    assert box.proposal.cls is None and box.proposal.supporting_masklet_ids == ()


def test_lifted_box_keeps_the_largest_group_and_skips_small_ones():
    rng = np.random.default_rng(0)
    car = rng.uniform([-2, -1, 0.3], [2, 1, 1.5], size=(30, 3)) + [10, 0, 0]
    # Default radius: 30 points in a car's volume chain at 2 m.
    bleed = np.array([[40.0, 40.0, 1.0]])
    tiny = rng.uniform(-0.2, 0.2, size=(3, 3)) + [0, 20, 1]
    xyz = np.concatenate([car, bleed, tiny])
    fp = _points(
        xyz,
        instance_id=["mA"] * 31 + ["mB"] * 3,
        cls=["car"] * 31 + ["truck"] * 3,
    )
    boxes = lifted.proposals(FRAME, fp, None, INSTANCES, LiftedConfig(min_points=5))
    assert [b.proposal.proposal_id for b in boxes] == ["lifted-mA-000004"]
    p = boxes[0].proposal
    assert p.lidar_point_count == 30 and not boxes[0].members[30]
    assert p.cx == pytest.approx(10.0, abs=0.5) and p.l <= 4.0
    assert p.cls == "car" and p.lift_confidence == pytest.approx(0.9)


def test_missing_lifted_instance_names_the_fix():
    fp = _points(np.zeros((6, 3)), instance_id=["mZ"] * 6, cls=["car"] * 6)
    with pytest.raises(KeyError, match="re-run semantic_lifting"):
        lifted.proposals(FRAME, fp, None, INSTANCES, LiftedConfig())


def _source_box(pid, members, provenance="lidar_mos", cls=None):
    from wato_proposal_generation.boxes import SourceBox, proposal
    from wato_common.schemas import Box3D

    p = proposal(
        pid,
        FRAME,
        Box3D(0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0),
        cls=cls,
        score=0.5,
        provenance=provenance,
        lidar_point_count=int(np.sum(members)),
    )
    return SourceBox(p, np.asarray(members, bool))


def test_match_is_one_to_one_by_most_shared_points():
    cfg = FusionConfig(min_shared_points=2, min_shared_fraction=0.3)
    m0 = _source_box("m0", [1, 1, 1, 1, 0, 0, 0, 0])
    m1 = _source_box("m1", [0, 0, 1, 1, 1, 1, 0, 0])
    f0 = _source_box("f0", [1, 1, 1, 1, 1, 1, 0, 0], "lifted", "car")
    f1 = _source_box("f1", [0, 0, 0, 0, 0, 0, 1, 1], "lifted", "car")
    # m0 and m1 both share 4 points with f0; m0 wins the tie on index; m1 then
    # shares nothing enough with f1.
    assert fusion.match([m0, m1], [f0, f1], cfg) == [(0, 0)]
    assert fusion.match([m0], [f1], cfg) == []


def test_fuse_merges_pairs_and_keeps_every_unmatched_box():
    xyz = np.column_stack([np.arange(8.0), np.zeros(8), np.ones(8)])
    fp = _points(xyz)
    m0 = _source_box("m0", [1, 1, 1, 1, 0, 0, 0, 0])
    m1 = _source_box("m1", [0, 0, 0, 0, 0, 0, 1, 0])
    f0 = _source_box("f0", [0, 1, 1, 1, 1, 0, 0, 0], "lifted", "car")
    rows = fusion.fuse(
        FRAME, [m0, m1], [f0], fp, None, FusionConfig(min_shared_points=2)
    )
    assert [p.member_proposal_ids for p in rows] == [("m0", "f0"), ("m1",)]
    pair, alone = rows
    assert pair.cls == "car" and pair.lidar_point_count == 5
    assert pair.cx == pytest.approx(2.0)
    assert alone.cls is None and alone.provenance == "fused"
    assert {p.proposal_id for p in rows} == {"fused-000004-0", "fused-000004-1"}
