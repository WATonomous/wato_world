"""Tests for wato_common.tracking (box fit, IoU, Kalman, association)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wato_common.schemas import Box3D
from wato_common.tracking import (
    AssociationGates,
    BoxKalmanFilter,
    MultiObjectTracker,
    associate,
    cost_terms,
    fit_bev_box,
    iou_3d,
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
