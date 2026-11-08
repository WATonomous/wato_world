"""wato_common.eval.motion: an object's ground-truth parked / moving state."""

from __future__ import annotations

import numpy as np

from wato_common.artifacts.gt_boxes import MOVING, PARKED
from wato_common.eval.motion import motion_state

S = 1_000_000_000


def test_label_jitter_is_parked():
    t = np.arange(10) * S // 2
    xy = np.column_stack([0.05 * np.sin(np.arange(10)), np.zeros(10)])
    assert motion_state(t, xy) == PARKED


def test_steady_walk_is_moving():
    t = np.arange(10) * S // 2
    xy = np.column_stack([1.4 * np.arange(10) * 0.5, np.zeros(10)])
    assert motion_state(t, xy) == MOVING


def test_out_and_back_is_moving_although_it_ends_where_it_started():
    t = np.arange(3) * S
    xy = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 0.0]])
    assert motion_state(t, xy) == MOVING


def test_order_does_not_matter():
    t = np.array([2, 0, 1]) * S
    xy = np.array([[2.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    assert motion_state(t, xy) == MOVING


def test_a_single_label_is_unknown():
    assert motion_state(np.array([0]), np.zeros((1, 2))) is None


def test_instant_speeds_follow_input_order():
    from wato_common.eval.motion import instant_speeds

    t = np.array([1, 0, 2]) * S
    xy = np.array([[2.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    v = instant_speeds(t, xy)
    # sorted: x = 0, 2, 2 at t = 0, 1, 2 → speeds 2, 1, 0
    assert np.allclose(v, [1.0, 2.0, 0.0])
    assert instant_speeds(np.array([0]), np.zeros((1, 2))).tolist() == [0.0]
