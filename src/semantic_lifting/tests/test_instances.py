"""Tests for instances.py — merging one object's masklets across cameras."""

from __future__ import annotations

import numpy as np

from wato_common.artifacts.masklets import MaskletRecord
from wato_semantic_lifting.instances import CoClaims, merge_masklets
from wato_semantic_lifting.voting import Votes


def _rec(masklet_id: str, cam: str, cls="car", frames=(1, 2, 3)) -> MaskletRecord:
    return MaskletRecord(masklet_id, cam, cls, 0.5, tuple(frames))


def _votes(*rows: tuple[int, int, int]) -> Votes:
    """rows of (point, camera, label)."""
    a = np.array(rows, dtype=np.int64).reshape(-1, 3)
    return Votes(a[:, 0], a[:, 1], a[:, 2], np.ones(len(a)))


def _co(*sweeps: Votes | tuple[Votes, np.ndarray]) -> CoClaims:
    """Sweeps' votes, each optionally with its points' xyz (default: every
    point at the origin)."""
    co = CoClaims()
    for i, sweep in enumerate(sweeps):
        v, xyz = sweep if isinstance(sweep, tuple) else (sweep, None)
        if xyz is None:
            xyz = np.zeros((int(v.point_idx.max()) + 1, 3))
        co.add(i, v, xyz)
    return co


def _merge(records, co, n=2, frac=0.5, sep=3.0):
    return merge_masklets(
        records,
        co,
        min_shared_points=n,
        min_shared_fraction=frac,
        max_separation_m=sep,
    ).tolist()


def test_co_claims_count_shared_points_and_the_smaller_claim():
    # Labels 0 (cam 0) and 1 (cam 1) share points 0, 1; label 1 also has 2-4.
    v = _votes((0, 0, 0), (1, 0, 0), (0, 1, 1), (1, 1, 1), (2, 1, 1), (3, 1, 1))
    co = _co(v, v)
    assert co.shared == {(0, 1): 4}
    assert co.smaller_claim == {(0, 1): 4}  # min(2, 4) per sweep, twice


def test_masklets_claiming_the_same_points_merge_under_the_smallest_id():
    records = [_rec("b", "A"), _rec("a", "B"), _rec("c", "C")]
    co = _co(_votes((0, 0, 0), (1, 0, 0), (0, 1, 1), (1, 1, 1)))
    # 0 and 1 merge, represented by masklet "a" (index 1); 2 stays alone.
    assert _merge(records, co) == [1, 1, 2]


def test_weak_or_incidental_overlap_does_not_merge():
    records = [_rec("big", "A"), _rec("car", "B")]
    # A road-sized mask touching a car: 1 of the car's 4 points is shared.
    v = _votes(
        *[(p, 0, 0) for p in range(100)], *[(p, 1, 1) for p in (0, 200, 201, 202)]
    )
    co = _co(v, v, v)
    assert _merge(records, co, n=2, frac=0.5) == [0, 1]  # shared 3, fraction 0.25
    assert _merge(records, co, n=4, frac=0.2) == [0, 1]  # too few points
    assert _merge(records, co, n=2, frac=0.2) == [0, 0]


def test_different_classes_never_merge():
    records = [_rec("a", "A", cls="car"), _rec("b", "B", cls="truck")]
    co = _co(_votes((0, 0, 0), (1, 0, 0), (0, 1, 1), (1, 1, 1)))
    assert _merge(records, co) == [0, 1]


def test_two_masklets_seen_together_by_one_camera_stay_apart():
    """a and c are both in camera A's frame 2: two objects. Each overlaps b
    (camera B); only the stronger link is kept."""
    records = [
        _rec("a", "A", frames=(1, 2)),
        _rec("b", "B"),
        _rec("c", "A", frames=(2, 3)),
    ]
    strong = _votes(*[(p, 0, 0) for p in range(5)], *[(p, 1, 1) for p in range(5)])
    weak = _votes(*[(p, 0, 2) for p in range(3)], *[(p, 1, 1) for p in range(3)])
    assert _merge(records, _co(strong, weak)) == [0, 0, 2]


def test_one_camera_masklets_at_different_times_can_join_through_another():
    """SAM2 lost the car in camera A and restarted it (a, then c); camera B
    tracked it throughout (b): one object."""
    records = [
        _rec("a", "A", frames=(1, 2)),
        _rec("b", "B"),
        _rec("c", "A", frames=(3, 4)),
    ]
    v1 = _votes((0, 0, 0), (1, 0, 0), (0, 1, 1), (1, 1, 1))
    v2 = _votes((0, 0, 2), (1, 0, 2), (0, 1, 1), (1, 1, 1))
    assert _merge(records, _co(v1, v2)) == [0, 0, 0]


def test_co_claims_record_where_each_masklet_is_per_sweep():
    xyz = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 4.0, 0.0]])
    co = _co((_votes((0, 0, 0), (1, 0, 0), (2, 1, 1)), xyz))
    assert co.positions[0] == {0: (1.0, 0.0)}  # median of its points' x, y
    assert co.positions[1] == {0: (10.0, 4.0)}


def test_a_drifted_masklet_does_not_chain_two_objects():
    """Camera A's masklet m covers car X, then drifts onto car Y. It shares
    points with b (camera B, on X) and with c (camera C, on Y), but b and c
    are 20 m apart in the same sweep: two objects, so only one link is kept."""
    records = [_rec("m", "A"), _rec("b", "B"), _rec("c", "C")]
    x = np.zeros((4, 3))
    y = np.full((4, 3), 20.0)
    on_x = _votes(*[(p, 0, 0) for p in range(4)], *[(p, 1, 1) for p in range(4)])
    on_y = _votes(*[(p, 0, 0) for p in range(3)], *[(p, 2, 2) for p in range(3)])
    both = (_votes((0, 1, 1), (1, 2, 2)), np.vstack([x[:1], y[:1]]))
    co = _co((on_x, x), (on_y, y), both)
    assert _merge(records, co) == [1, 1, 2]  # m+b (stronger), c alone
    assert _merge(records, co, sep=100.0) == [1, 1, 1]
