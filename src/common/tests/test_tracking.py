"""Tests for wato_common.tracking (box fit, IoU, Kalman, association)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wato_common.schemas import Box3D
from wato_common.tracking import (
    GIOU_3D,
    AssociationGates,
    BoxKalmanFilter,
    MultiObjectTracker,
    associate,
    bev_inside_fraction,
    box_corners_3d,
    cost_terms,
    fit_bev_box,
    giou_3d,
    giou_3d_cannot_exceed,
    iou_3d,
    iou_bev,
    points_in_box,
    wrap_heading_half_pi,
)


def _box(cx=0.0, cy=0.0, cz=0.0, l=4.0, w=2.0, h=1.5, heading=0.0) -> Box3D:  # noqa: E741
    return Box3D(cx=cx, cy=cy, cz=cz, w=w, l=l, h=h, heading=heading)


def _cuboid_points(box: Box3D, n: int = 400, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    local = rng.uniform(-0.5, 0.5, size=(n, 3)) * np.array([box.l, box.w, box.h])
    c, s = math.cos(box.heading), math.sin(box.heading)
    xy = local[:, :2] @ np.array([[c, s], [-s, c]])
    return np.column_stack([xy + [box.cx, box.cy], local[:, 2] + box.cz])


# --------------------------------------------------------------------------- #
# Boxes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("heading", [0.0, 0.3, -1.2, math.pi / 2])
def test_fit_bev_box_recovers_oriented_cuboid(heading):
    truth = _box(cx=10.0, cy=-3.0, cz=1.0, heading=wrap_heading_half_pi(heading))
    fit = fit_bev_box(_cuboid_points(truth, n=2000))
    assert fit.l >= fit.w
    assert fit.l == pytest.approx(truth.l, abs=0.1)
    assert fit.w == pytest.approx(truth.w, abs=0.1)
    assert fit.cx == pytest.approx(truth.cx, abs=0.1)
    assert fit.cy == pytest.approx(truth.cy, abs=0.1)
    dh = wrap_heading_half_pi(fit.heading - truth.heading)
    assert abs(dh) < 0.05


def test_fit_bev_box_degenerate_inputs_have_floored_extents():
    one = fit_bev_box(np.array([[1.0, 2.0, 3.0]]))
    assert one.l > 0 and one.w > 0 and one.h > 0
    line = fit_bev_box(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0]]))
    assert line.l == pytest.approx(math.sqrt(8), abs=1e-6)
    assert abs(wrap_heading_half_pi(line.heading - math.pi / 4)) < 1e-6


def test_box_corners_3d_span_the_box():
    box = _box(cx=10.0, cy=-3.0, cz=1.0, heading=0.4)
    corners = box_corners_3d(box)
    assert corners.shape == (8, 3)
    np.testing.assert_allclose(corners[:4, 2], 0.25)
    np.testing.assert_allclose(corners[4:, 2], 1.75)
    np.testing.assert_allclose(corners[:4, :2], corners[4:, :2])
    np.testing.assert_allclose(corners.mean(axis=0), [10.0, -3.0, 1.0])
    assert points_in_box(corners, box, margin_m=1e-9).all()


def test_iou_identical_disjoint_and_half_overlap():
    a = _box()
    assert iou_3d(a, a) == pytest.approx(1.0)
    assert iou_3d(a, _box(cx=100.0)) == 0.0
    # Shift by half the length along heading → intersection = half volume.
    half = iou_3d(a, _box(cx=2.0))
    assert half == pytest.approx(0.5 / 1.5, rel=1e-6)


def test_iou_rotated_boxes_symmetric():
    a = _box(heading=0.2)
    b = _box(cx=0.5, cy=0.3, heading=-0.4)
    assert iou_3d(a, b) == pytest.approx(iou_3d(b, a), rel=1e-9)
    assert 0.0 < iou_3d(a, b) < 1.0


def test_bev_inside_fraction_is_the_share_of_the_first_box():
    big = _box()  # 4 x 2
    small = _box(cx=2.0, l=2.0, w=1.0)  # 2 x 1, half of it past big's end
    assert bev_inside_fraction(small, big) == pytest.approx(0.5)
    assert bev_inside_fraction(big, small) == pytest.approx(1.0 / 8.0)
    assert bev_inside_fraction(small, _box(cx=50.0)) == 0.0
    assert bev_inside_fraction(big, big) == pytest.approx(1.0)


def test_points_in_box_respects_heading():
    b = _box(heading=math.pi / 2)  # long axis along y
    pts = np.array([[0.0, 1.9, 0.0], [1.9, 0.0, 0.0]])
    assert points_in_box(pts, b).tolist() == [True, False]


# --------------------------------------------------------------------------- #
# Kalman
# --------------------------------------------------------------------------- #


def test_kalman_learns_constant_velocity():
    kf = BoxKalmanFilter(_box())
    for k in range(1, 20):
        kf.predict(0.1)
        kf.update(_box(cx=1.0 * k))  # 10 m/s along x
    assert kf.velocity[0] == pytest.approx(10.0, abs=0.5)
    pred = kf.predict(0.1)
    assert pred.cx == pytest.approx(20.0, abs=0.3)


def test_kalman_heading_update_wraps_undirected():
    kf = BoxKalmanFilter(_box(heading=math.pi / 2 - 0.01))
    kf.predict(0.1)
    # Same physical orientation measured on the other side of the wrap.
    out = kf.update(_box(heading=-math.pi / 2 + 0.01))
    assert abs(wrap_heading_half_pi(out.heading - math.pi / 2)) < 0.05


# --------------------------------------------------------------------------- #
# Association
# --------------------------------------------------------------------------- #


def test_cost_terms_match_chen_definitions():
    a = _box()
    b = _box(cx=2.0, l=4.0, w=2.0, h=0.75)
    c_d, c_o, c_v = cost_terms(a, b)
    assert c_d == pytest.approx(math.hypot(2.0, 0.0))
    assert c_v == pytest.approx(0.5)
    assert 0.0 < c_o < 1.0


def test_associate_gates_reject_distant_pair():
    dets = [_box(cx=0.0), _box(cx=50.0)]
    preds = [_box(cx=0.3)]
    matches, ud, up = associate(dets, preds)
    assert matches == [(0, 0)]
    assert ud == [1] and up == []


def test_associate_infeasible_pair_never_displaces_feasible():
    # det0: c_d = 2.1 m > T_d (infeasible) but lowest raw cost (~2.79).
    # det1: every term inside its gate, raw cost ~3.42. Solving on raw cost
    # and gating afterwards would pick det0, reject it, and match nothing.
    preds = [_box()]
    dets = [_box(cx=2.1), _box(cx=1.9, h=0.5)]
    c0 = sum(cost_terms(dets[0], preds[0]))
    c1 = sum(cost_terms(dets[1], preds[0]))
    assert c0 < c1
    matches, ud, _ = associate(dets, preds, AssociationGates())
    assert matches == [(1, 0)]
    assert ud == [0]


def test_tracker_follows_mover_and_reidentifies_after_gap():
    trk = MultiObjectTracker(n_old=3)
    ids = []
    for k in range(12):
        if k in (5, 6):  # two missed frames
            trk.step(0.1 * k, [])
            continue
        ids.append(trk.step(0.1 * k, [_box(cx=0.8 * k)])[0])
    assert len(set(ids)) == 1
    (track,) = trk.all_tracks()
    assert len(track.history) == 10


def test_tracker_retires_after_n_old_and_births_new_id():
    trk = MultiObjectTracker(n_old=1)
    first = trk.step(0.0, [_box()])[0]
    trk.step(0.1, [])
    trk.step(0.2, [])  # misses=2 > n_old → retired
    second = trk.step(0.3, [_box()])[0]
    assert first != second
    assert len(trk.all_tracks()) == 2


def test_strong_detections_are_matched_before_weak_ones():
    trk = MultiObjectTracker()
    trk.step(0.0, [_box(cx=0.0)])
    # Frame 2: a weak box fits the track better than the strong one, but both
    # are within the gates; strong-first gives the track to the strong box.
    weak, strong = _box(cx=0.1), _box(cx=0.8)
    ids = trk.step(0.1, [weak, strong], strong=[False, True])
    assert ids[1] == 0 and ids[0] == 1
    # Without the flags, the nearer (weak) box wins.
    trk2 = MultiObjectTracker()
    trk2.step(0.0, [_box(cx=0.0)])
    assert trk2.step(0.1, [weak, strong])[0] == 0


def test_a_det_that_may_not_birth_only_extends_tracks():
    """ByteTrack: a low-score box can continue a track, never start one."""
    trk = MultiObjectTracker()
    assert trk.step(0.0, [_box(cx=0.0), _box(cx=20.0)], can_birth=[True, False]) == [
        0,
        -1,
    ]
    # Matched, it extends the track whatever can_birth says.
    assert trk.step(0.1, [_box(cx=0.1)], can_birth=[False]) == [0]
    assert len(trk.all_tracks()) == 1
    # Unset, every unmatched det starts a track as before.
    assert trk.step(0.2, [_box(cx=0.2), _box(cx=20.0)]) == [0, 1]


def test_a_shared_key_breaks_a_near_tie_but_never_opens_a_gate():
    trk = MultiObjectTracker(key_bonus=1.0)
    trk.step(0.0, [_box(cx=0.0), _box(cx=3.0)], keys=["a", "b"])
    # Two boxes between the tracks; keys pull each to its own track although
    # geometry alone would swap them.
    ids = trk.step(0.1, [_box(cx=1.6), _box(cx=1.4)], keys=["a", "b"])
    assert ids == [0, 1]
    # A key never makes an out-of-gate pair match (T_d = 2 m).
    far = trk.step(0.2, [_box(cx=30.0)], keys=["a"])
    assert far == [2]


def test_a_shared_key_can_open_the_shape_gates_but_not_the_distance_gate():
    # A one-sided box whose visible face changed: same object, IoU ~ 0.
    a = _box(cx=0.0, l=4.0, w=0.3)
    b = _box(cx=0.0, cy=0.9, l=0.3, w=1.8)
    closed = MultiObjectTracker()
    closed.step(0.0, [a], keys=["k"])
    assert closed.step(0.1, [b], keys=["k"]) == [1]
    opened = MultiObjectTracker(key_opens_shape_gates=True)
    opened.step(0.0, [a], keys=["k"])
    assert opened.step(0.1, [b], keys=["k"]) == [0]
    # Without the shared key the shape gates still hold.
    assert opened.step(0.2, [_box(cx=0.0, cy=0.9, l=0.3, w=1.8)], keys=["other"]) == [1]
    assert opened.step(0.3, [_box(cx=5.0)], keys=["k"]) == [2]


# --------------------------------------------------------------------------- #
# GIoU cost, pruning, never-retiring tracks
# --------------------------------------------------------------------------- #


def test_giou_equals_iou_for_a_box_inside_another_and_falls_with_distance():
    car = _box(l=4.5, w=1.8)
    sliver = _box(cx=1.0, l=1.0, w=0.2)
    assert giou_3d(car, car) == pytest.approx(1.0)
    # The hull is the car's box, so GIoU = IoU = 0.2 / 8.1 (Chen's IoU gate
    # 0.05 and volume gate 0.3 both reject this pair).
    assert giou_3d(car, sliver) == pytest.approx(iou_3d(car, sliver))
    assert giou_3d(car, sliver) == pytest.approx(0.2 / 8.1, abs=1e-6)
    far = [giou_3d(car, _box(cx=x)) for x in (5.0, 8.0, 20.0)]
    assert far[0] < 0 and far[0] > far[1] > far[2] > -1.0


def test_iou_bev_ignores_height():
    a, b = _box(), _box(cx=2.0, cz=10.0)
    assert iou_3d(a, b) == 0.0
    assert iou_bev(a, b) == pytest.approx(4.0 / 12.0)


def test_giou_pruning_never_drops_a_pair_that_passes():
    rng = np.random.default_rng(0)

    def rand(n):
        return [
            _box(
                cx=rng.uniform(-15, 15),
                cy=rng.uniform(-15, 15),
                cz=rng.uniform(0, 2),
                l=rng.uniform(0.1, 6),
                w=rng.uniform(0.1, 2.5),
                h=rng.uniform(0.1, 3),
                heading=rng.uniform(-math.pi / 2, math.pi / 2),
            )
            for _ in range(n)
        ]

    dets, preds = rand(60), rand(60)
    for g in (-0.5, -0.31, 0.0):
        pruned = giou_3d_cannot_exceed(dets, preds, g)
        assert pruned.any()
        for i, j in zip(*np.nonzero(pruned)):
            assert giou_3d(dets[i], preds[j]) < g


def test_giou_cost_keeps_a_sliver_chen_splits_off():
    track = MultiObjectTracker(AssociationGates(cost=GIOU_3D))
    chen = MultiObjectTracker()
    for trk in (track, chen):
        trk.step(0.0, [_box(l=4.5, w=1.8)])
    sliver = _box(cx=1.0, l=1.0, w=0.2)
    assert track.step(0.1, [sliver]) == [0]
    assert chen.step(0.1, [sliver]) == [1]
    # GIoU < −0.5: a new track.
    assert track.step(0.2, [_box(cx=12.0)]) == [1]


def test_unknown_cost_is_rejected():
    with pytest.raises(ValueError):
        AssociationGates(cost="mahalanobis")


def test_a_track_that_never_retires_reidentifies_after_a_long_gap():
    immortal = MultiObjectTracker(n_old=None)
    mortal = MultiObjectTracker(n_old=5)
    for trk in (immortal, mortal):
        trk.step(0.0, [_box()])
        for k in range(1, 40):
            trk.step(0.1 * k, [])
    assert immortal.step(4.0, [_box()]) == [0]
    assert mortal.step(4.0, [_box()]) == [1]


def test_history_records_the_filter_velocity():
    trk = MultiObjectTracker()
    for k in range(10):
        trk.step(0.1 * k, [_box(cx=1.0 * k)])
    (track,) = trk.all_tracks()
    assert track.history[0].velocity == (0.0, 0.0, 0.0)
    assert track.history[-1].velocity[0] == pytest.approx(10.0, abs=1.0)
