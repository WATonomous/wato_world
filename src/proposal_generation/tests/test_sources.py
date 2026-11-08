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
    ScanRefinementConfig,
)
from wato_proposal_generation.frame_points import FramePoints, LiftedSeed

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


def _sal_config(**overrides):
    return LiftedConfig(support_method="scan_clusters", **overrides)


def test_gate_keeps_only_moving_multi_source_transient_clusters():
    cfg = LidarMosConfig()
    clusters = [
        _cluster(0),
        _cluster(1, motion_score=0.9),
        _cluster(2, n_sources=1),
        _cluster(3, frac_persistent=0.6),
    ]
    kept = lidar_mos.gate(clusters, lidar_mos.track_shapes(clusters), cfg)
    assert [c.cluster_id for c in kept] == [0]


def test_gate_drops_wall_shaped_tracks_by_their_median_box():
    # Track 1 is a 3 x 0.3 m strip in two of its three frames: wall-shaped,
    # although one frame's cluster looks like a car. Track 2 is a pedestrian.
    clusters = [
        _cluster(0, track_hint_id=1, l=3.0, w=0.3),
        _cluster(1, track_hint_id=1, l=3.2, w=0.3),
        _cluster(2, track_hint_id=1, l=4.0, w=1.8),
        _cluster(3, track_hint_id=2, l=0.6, w=0.5),
        # No track: judged on its own box.
        _cluster(4, track_hint_id=-1, l=5.0, w=1.0),
    ]
    shapes = lidar_mos.track_shapes(clusters)
    assert shapes[1].l == pytest.approx(3.2) and shapes[1].w == pytest.approx(0.3)
    kept = lidar_mos.gate(clusters, shapes, LidarMosConfig())
    assert [c.cluster_id for c in kept] == [3]
    loose = LidarMosConfig(max_aspect_ratio=float("inf"))
    assert len(lidar_mos.gate(clusters, shapes, loose)) == 5


def test_small_tracks_keep_a_box_only_when_a_lifted_label_classes_it():
    ped = _cluster(0, track_hint_id=7, l=0.6, w=0.5, h=1.4)  # 0.3 m², 0.42 m³
    shapes = lidar_mos.track_shapes([ped])
    unlabelled = _points(np.zeros((6, 3)), cluster_id=[0] * 6)
    assert (
        lidar_mos.proposals(
            FRAME, [ped], unlabelled, None, INSTANCES, shapes, LidarMosConfig()
        )
        == []
    )
    instances = {"mP": LiftedInstance("mP", "pedestrian", ("mP",), ("CAM_A",), 10, 1)}
    labelled = _points(
        np.zeros((6, 3)),
        instance_id=["mP"] * 6,
        cls=["pedestrian"] * 6,
        cluster_id=[0] * 6,
    )
    [box] = lidar_mos.proposals(
        FRAME, [ped], labelled, None, instances, shapes, LidarMosConfig()
    )
    assert box.proposal.cls == "pedestrian"
    off = LidarMosConfig(min_footprint_m2=0.0, min_volume_m3=0.0)
    assert (
        len(lidar_mos.proposals(FRAME, [ped], unlabelled, None, INSTANCES, shapes, off))
        == 1
    )


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
    shapes = lidar_mos.track_shapes([_cluster()])
    [box] = lidar_mos.proposals(
        FRAME, [_cluster()], fp, None, INSTANCES, shapes, LidarMosConfig()
    )
    p = box.proposal
    assert p.cls == "car" and p.supporting_cam_ids == ("CAM_A",)
    assert p.lidar_point_count == 6 and p.score == pytest.approx(0.75)
    assert (p.motion_score, p.n_sources, p.track_life) == (1.5, 2, 5)

    strict = LidarMosConfig(min_labeled_fraction=0.9)
    [box] = lidar_mos.proposals(
        FRAME, [_cluster()], fp, None, INSTANCES, shapes, strict
    )
    assert box.proposal.cls is None and box.proposal.supporting_masklet_ids == ()


def test_lifted_box_refines_support_and_skips_small_ones():
    rng = np.random.default_rng(0)
    car = rng.uniform([-2, -1, 0.3], [2, 1, 1.5], size=(30, 3)) + [10, 0, 0]
    # A sufficiently dense car cluster matches; the isolated bleed does not.
    bleed = np.array([[40.0, 40.0, 1.0]])
    tiny = rng.uniform(-0.2, 0.2, size=(3, 3)) + [0, 20, 1]
    xyz = np.concatenate([car, bleed, tiny])
    fp = _points(
        xyz,
        instance_id=["mA"] * 31 + ["mB"] * 3,
        cls=["car"] * 31 + ["truck"] * 3,
    )
    boxes = lifted.proposals(FRAME, fp, None, INSTANCES, _sal_config(min_points=5))
    assert [b.proposal.proposal_id for b in boxes] == ["lifted-mA-000004"]
    p = boxes[0].proposal
    assert p.lidar_point_count == 30 and not boxes[0].members[30]
    assert p.cx == pytest.approx(10.0, abs=0.5) and p.l <= 4.0
    assert p.cls == "car" and p.lift_confidence == pytest.approx(0.9)


def test_lifted_recovers_unlabelled_scan_returns_and_removes_bleed():
    x, y = np.meshgrid(np.arange(3) * 0.3 + 10, np.arange(3) * 0.3)
    surface = np.column_stack([x.ravel(), y.ravel(), np.ones(9)])
    xyz = np.vstack([surface, [30, 30, 1]])
    # Six object seeds plus an isolated mislabeled return. Three object
    # returns have neither an image label nor a motion-cluster assignment.
    fp = _points(xyz, instance_id=["mA"] * 6 + [""] * 3 + ["mA"])
    [box] = lifted.proposals(FRAME, fp, None, INSTANCES, _sal_config())
    assert box.members.tolist() == [True] * 9 + [False]
    assert box.proposal.lidar_point_count == 9
    assert box.proposal.lift_seed_point_count == 6
    assert box.proposal.lift_confidence == pytest.approx(0.9)


def test_lifted_preserves_original_seed_confidence_after_replacement():
    xyz = np.column_stack([np.arange(9) * 0.05, np.zeros(9), np.ones(9)])
    xyz = np.vstack([xyz, [30, 30, 1]])
    fp = _points(
        xyz,
        instance_id=["mA"] * 6 + [""] * 3 + ["mA"],
        confidence=[0.9] * 6 + [0] * 3 + [0.1],
    )
    [box] = lifted.proposals(FRAME, fp, None, INSTANCES, _sal_config())
    assert box.proposal.lift_confidence == pytest.approx((6 * 0.9 + 0.1) / 7)
    assert fp.instance_id.tolist() == ["mA"] * 6 + [""] * 3 + ["mA"]
    assert np.all(fp.confidence[6:9] == 0)


def test_refinement_runs_before_the_final_five_point_floor():
    xyz = np.column_stack([np.arange(5) * 0.05, np.zeros(5), np.ones(5)])
    fp = _points(xyz, instance_id=["mA"] * 3 + [""] * 2)
    [box] = lifted.proposals(FRAME, fp, None, INSTANCES, _sal_config())
    assert box.proposal.lidar_point_count == 5
    assert box.proposal.lift_seed_point_count == 3


def test_legacy_method_is_separate_from_sals_original_support_fallback():
    xyz = np.column_stack(
        [
            np.r_[np.arange(6) * 0.01, 5 + np.arange(6) * 0.01, 20],
            np.zeros(13),
            np.ones(13),
        ]
    )
    fp = _points(xyz, instance_id=["mA"] * 13)
    [sal] = lifted.proposals(FRAME, fp, None, INSTANCES, _sal_config())
    [legacy] = lifted.proposals(FRAME, fp, None, INSTANCES, LiftedConfig())
    assert sal.proposal.lidar_point_count == 13
    assert legacy.proposal.lidar_point_count == 6


def test_a_contaminated_seed_can_match_background_without_a_semantic_veto():
    # SAL's geometric overlap does not know that the larger component is a wall.
    rng = np.random.default_rng(1)
    ped = rng.uniform([-0.25, -0.25, 0.3], [0.25, 0.25, 1.7], size=(20, 3))
    wall = rng.uniform([1.5, -1.0, 0.3], [1.6, 1.0, 2.0], size=(40, 3))
    xyz = np.concatenate([ped, wall])
    instances = {"mP": LiftedInstance("mP", "pedestrian", ("mP",), ("CAM_A",), 10, 1)}
    fp = _points(xyz, instance_id=["mP"] * 60, cls=["pedestrian"] * 60)
    [box] = lifted.proposals(FRAME, fp, None, instances, _sal_config())
    assert box.proposal.lidar_point_count == 40
    assert box.members.tolist() == [False] * 20 + [True] * 40


def test_lifted_height_cap_leaves_out_points_above_it():
    rng = np.random.default_rng(2)
    car = rng.uniform([-2, -1, 0.0], [2, 1, 1.5], size=(40, 3))
    pole = np.column_stack([np.full(10, 1.5), np.zeros(10), np.linspace(1.6, 6.0, 10)])
    fp = _points(np.concatenate([car, pole]), instance_id=["mA"] * 50, cls=["car"] * 50)
    [tall] = lifted.proposals(FRAME, fp, None, INSTANCES, _sal_config())
    assert tall.proposal.h > 5.0
    capped = _sal_config(max_height_m={"car": 3.0})
    [box] = lifted.proposals(FRAME, fp, None, INSTANCES, capped)
    assert box.proposal.h <= 3.0 and box.proposal.lidar_point_count < 50
    # A class without a cap is untouched.
    assert _sal_config(max_height_m={"truck": 3.0}).max_height_m.get("car") is None


def test_lifted_box_inside_a_same_class_box_with_more_points_is_dropped():
    rng = np.random.default_rng(3)
    front = rng.uniform([-2, -1, 0.3], [2, 1, 1.5], size=(40, 3)) + [10, 0, 0]
    hidden = rng.uniform([-0.5, -0.5, 0.3], [0.5, 0.5, 1.0], size=(8, 3)) + [11, 0, 0]
    beside = rng.uniform([-2, -1, 0.3], [2, 1, 1.5], size=(30, 3)) + [10, 5, 0]
    instances = {
        "mA": LiftedInstance("mA", "car", ("mA",), ("CAM_A",), 10, 1),
        "mH": LiftedInstance("mH", "car", ("mH",), ("CAM_A",), 10, 1),
        "mS": LiftedInstance("mS", "car", ("mS",), ("CAM_A",), 10, 1),
    }
    fp = _points(
        np.concatenate([front, hidden, beside]),
        instance_id=["mA"] * 40 + ["mH"] * 8 + ["mS"] * 30,
        cls=["car"] * 78,
    )
    # Exercise the existing box filter independently of scan replacement.
    original = ScanRefinementConfig(enabled=False)
    boxes = lifted.proposals(
        FRAME, fp, None, instances, _sal_config(refinement=original)
    )
    assert [b.proposal.proposal_id for b in boxes] == [
        "lifted-mA-000004",
        "lifted-mS-000004",
    ]
    off = _sal_config(
        refinement=original, max_nested_fraction=1.1, fragment_min_fraction=1.1
    )
    assert len(lifted.proposals(FRAME, fp, None, instances, off)) == 3


def test_a_few_points_against_a_bigger_objects_edge_are_its_fragment():
    # A car's rear face, a 0.1 m grid; a mask beside it claims 8 of the
    # face's edge returns (a thin column), another class's mask included.
    gy, gz = np.meshgrid(np.arange(-0.9, 0.91, 0.1), np.arange(0.3, 1.51, 0.1))
    face = np.column_stack([np.full(gy.size, 10.0), gy.ravel(), gz.ravel()])
    column = np.column_stack(
        [np.full(8, 10.05), np.full(8, 1.05), np.linspace(0.4, 1.4, 8)]
    )
    rng = np.random.default_rng(4)
    walker = rng.uniform([-0.25, -0.25, 0.2], [0.25, 0.25, 1.7], size=(30, 3))
    walker += [10.0, 2.5, 0.0]  # 1.3 m clear of the car
    instances = {
        "mC": LiftedInstance("mC", "car", ("mC",), ("CAM_A",), 10, 1),
        "mF": LiftedInstance("mF", "pedestrian", ("mF",), ("CAM_A",), 10, 1),
        "mW": LiftedInstance("mW", "pedestrian", ("mW",), ("CAM_A",), 10, 1),
    }
    n = len(face)
    fp = _points(
        np.concatenate([face, column, walker]),
        instance_id=["mC"] * n + ["mF"] * 8 + ["mW"] * 30,
        cls=["car"] * n + ["pedestrian"] * 38,
    )
    original = ScanRefinementConfig(enabled=False)
    boxes = lifted.proposals(
        FRAME, fp, None, instances, _sal_config(refinement=original)
    )
    assert [b.proposal.proposal_id for b in boxes] == [
        "lifted-mC-000004",
        "lifted-mW-000004",
    ]
    off = _sal_config(refinement=original, fragment_min_fraction=1.1)
    assert len(lifted.proposals(FRAME, fp, None, instances, off)) == 3


def test_two_walkers_side_by_side_are_not_fragments_of_each_other():
    rng = np.random.default_rng(5)
    a = rng.uniform([-0.25, -0.25, 0.2], [0.25, 0.25, 1.7], size=(30, 3))
    b = rng.uniform([-0.25, -0.25, 0.2], [0.25, 0.25, 1.7], size=(20, 3))
    instances = {
        k: LiftedInstance(k, "pedestrian", (k,), ("CAM_A",), 10, 1)
        for k in ("mA", "mB")
    }
    fp = _points(
        np.concatenate([a + [10, 0, 0], b + [10, 0.6, 0]]),
        instance_id=["mA"] * 30 + ["mB"] * 20,
        cls=["pedestrian"] * 50,
    )
    cfg = _sal_config(refinement=ScanRefinementConfig(enabled=False))
    assert len(lifted.proposals(FRAME, fp, None, instances, cfg)) == 2


def test_missing_lifted_instance_names_the_fix():
    fp = _points(np.zeros((6, 3)), instance_id=["mZ"] * 6, cls=["car"] * 6)
    with pytest.raises(KeyError, match="re-run semantic_lifting"):
        lifted.proposals(FRAME, fp, None, INSTANCES, _sal_config())


def test_partial_overlap_is_owned_once_and_boxes_refit_to_final_support():
    from dataclasses import replace

    from wato_common.tracking import fit_bev_box

    xyz = np.column_stack([np.arange(10) * 0.1, np.zeros(10), np.ones(10)])
    fp = replace(
        _points(xyz),
        lifted_seeds={
            "mA": LiftedSeed(np.arange(6), np.full(6, 0.9)),
            "mB": LiftedSeed(np.arange(4, 10), np.full(6, 0.4)),
        },
    )
    cfg = _sal_config(
        refinement=ScanRefinementConfig(enabled=False),
        min_points=3,
        max_nested_fraction=1.1,
        fragment_min_fraction=1.1,
    )
    a, b = lifted.proposals(FRAME, fp, None, INSTANCES, cfg)
    assert np.flatnonzero(a.members).tolist() == list(range(6))
    assert np.flatnonzero(b.members).tolist() == list(range(6, 10))
    assert not (a.members & b.members).any()
    assert b.proposal.cx == pytest.approx(fit_bev_box(xyz[6:]).cx)
    assert b.proposal.lift_seed_point_count == 4
    assert b.proposal.lift_confidence == pytest.approx(0.4)


def test_motion_semantic_view_uses_surviving_camera_seeds_only():
    # Geometry expands six camera seeds to nine returns; a rejected competing
    # provisional identity labels the whole object in the old unique view.
    from dataclasses import replace

    xyz = np.column_stack([np.arange(9) * 0.05, np.zeros(9), np.ones(9)])
    fp = replace(
        _points(xyz, instance_id=["mB"] * 9, cls=["truck"] * 9),
        lifted_seeds={"mA": LiftedSeed(np.arange(6), np.full(6, 0.7))},
    )
    boxes = lifted.proposals(FRAME, fp, None, INSTANCES, _sal_config())
    view = lifted.semantic_view(fp, boxes)
    assert view.instance_id.tolist() == ["mA"] * 6 + [""] * 3
    assert view.cls.tolist() == ["car"] * 6 + [""] * 3
    assert view.confidence.tolist() == pytest.approx([0.7] * 6 + [0] * 3)
    assert fp.instance_id.tolist() == ["mB"] * 9


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


@pytest.mark.parametrize("gap_m", [0.05, 0.1, 0.3])
def test_a_pedestrian_beside_a_car_is_not_its_fragment(gap_m):
    """A person standing at a car's corner has ~8x fewer points than it, but
    a body's depth, not a sliver of the car's edge: most of its returns are
    further than fragment_radius_m from the car's."""
    gy, gz = np.meshgrid(np.arange(-0.9, 0.91, 0.1), np.arange(0.3, 1.51, 0.1))
    face = np.column_stack([np.full(gy.size, 10.0), gy.ravel(), gz.ravel()])
    rng = np.random.default_rng(6)
    person = rng.uniform([-0.25, 0.0, 0.2], [0.25, 0.5, 1.7], size=(30, 3))
    person += [10.0, 0.9 + gap_m, 0.0]
    instances = {
        "mC": LiftedInstance("mC", "car", ("mC",), ("CAM_A",), 10, 1),
        "mP": LiftedInstance("mP", "pedestrian", ("mP",), ("CAM_A",), 10, 1),
    }
    n = len(face)
    fp = _points(
        np.concatenate([face, person]),
        instance_id=["mC"] * n + ["mP"] * 30,
        cls=["car"] * n + ["pedestrian"] * 30,
    )
    assert n >= 5 * 30  # big enough for the fragment rule to consider it
    cfg = _sal_config(refinement=ScanRefinementConfig(enabled=False))
    boxes = lifted.proposals(FRAME, fp, None, instances, cfg)
    assert sorted(b.proposal.cls for b in boxes) == ["car", "pedestrian"]
    # Full SAL replacement can absorb neighboring returns into the car; the
    # existing fragment filter then suppresses the pedestrian. Keep this
    # observed limitation visible rather than imposing an unpublished veto.
    with_refinement = lifted.proposals(FRAME, fp, None, instances, _sal_config())
    expected = ["car"] if gap_m <= 0.1 else ["car", "pedestrian"]
    assert sorted(b.proposal.cls for b in with_refinement) == expected
