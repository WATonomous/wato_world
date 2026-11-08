"""Rejected 3D hypotheses cannot suppress another object's retained support."""

import numpy as np

from wato_common.artifacts.lidar_sweeps import Frame
from wato_common.artifacts.lifted_labels import LiftedInstance
from wato_proposal_generation import lifted
from wato_proposal_generation.config import LiftedConfig, ScanRefinementConfig
from wato_proposal_generation.frame_points import FramePoints, LiftedSeed
from wato_proposal_generation.scan_refinement import RefinedMask


def _proposals(xyz, supports, classes, **config):
    n = len(xyz)
    seeds = {
        iid: LiftedSeed(np.asarray(indices, np.int64), np.ones(len(indices)))
        for iid, indices in supports.items()
    }
    fp = FramePoints(
        sweep_id=np.zeros(n, np.int64),
        point_index=np.arange(n),
        xyz=np.asarray(xyz),
        instance_id=np.full(n, ""),
        cls=np.full(n, ""),
        confidence=np.zeros(n),
        cluster_id=np.full(n, -1),
        lifted_seeds=seeds,
    )
    instances = {
        iid: LiftedInstance(iid, classes[iid], (iid,), ("CAM",), len(indices), 1)
        for iid, indices in supports.items()
    }
    # Supply geometry directly to isolate gating from the clustering algorithm.
    refinements = {
        iid: RefinedMask(s.point_indices, s.point_indices, None, None, 1.0, "disabled")
        for iid, s in seeds.items()
    }
    rows = lifted.proposals(
        Frame(0, 0, (0,), 0),
        fp,
        None,
        instances,
        LiftedConfig(
            support_method="scan_clusters",
            refinement=ScanRefinementConfig(enabled=False),
            **config,
        ),
        refinements,
    )
    return {min(row.proposal.supporting_masklet_ids): row for row in rows}


def test_nested_rejection_does_not_first_discard_another_objects_fragment():
    rng = np.random.default_rng(14)
    inner = rng.uniform([-1, -1, 0.3], [1, 1, 1.5], size=(25, 3))
    angles = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    outer = np.column_stack([10 * np.cos(angles), 10 * np.sin(angles), np.ones(30)])
    # B claims all of A. B is rejected inside C's large box, whose actual
    # returns are far from A. B must never get to suppress A as a fragment.
    boxes = _proposals(
        np.concatenate([inner, outer]),
        {"A": np.arange(5), "B": np.arange(25), "C": np.arange(25, 55)},
        {"A": "pedestrian", "B": "car", "C": "car"},
    )
    assert set(boxes) == {"A", "C"}
    assert np.flatnonzero(boxes["A"].members).tolist() == list(range(5))
    assert boxes["A"].proposal.lift_seed_point_count == 5
    assert not np.any(boxes["A"].members & boxes["C"].members)


def test_fragment_rejection_does_not_first_discard_another_objects_nested_box():
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    edge = np.column_stack([np.cos(angles), np.sin(angles), np.ones(6)])
    small = np.array(
        [
            [-0.05, -0.05, 1],
            [-0.05, 0.05, 1],
            [0.05, -0.05, 1],
            [0.05, 0.05, 1],
            [0, 0, 1],
        ]
    )
    big = np.repeat(edge, 5, axis=0)
    # B's six points coincide with C's 30 returns but have distinct row keys.
    # B is C's fragment. A lies inside B's box and far from either edge set.
    boxes = _proposals(
        np.concatenate([small, edge, big]),
        {"A": np.arange(5), "B": np.arange(5, 11), "C": np.arange(11, 41)},
        {"A": "truck", "B": "truck", "C": "car"},
    )
    assert set(boxes) == {"A", "C"}
    assert np.flatnonzero(boxes["A"].members).tolist() == list(range(5))


def test_nested_gate_uses_refitted_exclusive_support():
    xyz = np.column_stack([np.arange(9), np.zeros(9), np.ones(9)])
    # The original B box overlaps A by enough to trigger the nested gate.
    # Once A's two contested points are excluded, B's final box is disjoint.
    boxes = _proposals(
        xyz,
        {"A": np.arange(6), "B": np.arange(4, 9)},
        {"A": "car", "B": "car"},
        min_points=3,
        fragment_min_fraction=2,
    )
    assert set(boxes) == {"A", "B"}
    assert np.flatnonzero(boxes["B"].members).tolist() == [6, 7, 8]
    assert boxes["B"].proposal.cx == 7
    assert boxes["B"].proposal.l == 2
    assert boxes["B"].proposal.lift_seed_point_count == 3
