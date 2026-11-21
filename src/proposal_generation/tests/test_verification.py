"""verification.verify: SAL's scan-cluster match plus FGR's frustum
containment, per scale, any scale passing."""

import numpy as np
import pytest

from wato_proposal_generation.config import VerificationConfig
from wato_proposal_generation.scan_refinement import Scale, build_bank
from wato_proposal_generation.verification import (
    CONTAINMENT_BELOW,
    IOU_NOT_ABOVE,
    NO_OVERLAP,
    VERIFIED,
    verify,
)

CFG = VerificationConfig(enabled=True)


def _scale(labels, radius=0.5) -> Scale:
    labels = np.asarray(labels, np.int64)
    return Scale(radius, labels, np.bincount(labels[labels >= 0]))


def _rows(*ranges) -> np.ndarray:
    return np.concatenate([np.arange(a, b) for a, b in ranges]).astype(np.int64)


def test_a_segment_inside_the_frustum_matching_the_seed_verifies():
    # Rows 0-9 are one cluster; the seed holds 7 of them, the frustum all 10
    # and a background row (10) that belongs to no cluster.
    scale = _scale([0] * 10 + [-1])
    (v,) = verify(
        {"car": _rows((0, 7))}, {"car": _rows((0, 11))}, [scale], 11, CFG
    ).values()
    assert (v.verified, v.status) == (True, VERIFIED)
    assert v.iou == pytest.approx(0.7) and v.containment == 1.0
    assert (v.cluster_points, v.frustum_points) == (10, 11)


def test_a_segment_reaching_beyond_the_frustum_fails_containment():
    # A hedge strip: 10 rows, the mask's rectangle covers 7 of them, which the
    # car mask claims. IoU 0.7 passes SAL, but 30% of the strip lies outside
    # the frustum: it belongs to something else (FGR).
    scale = _scale([0] * 10)
    v = verify({"car": _rows((0, 7))}, {"car": _rows((0, 7))}, [scale], 10, CFG)["car"]
    assert (v.verified, v.status) == (False, CONTAINMENT_BELOW)
    assert v.containment == pytest.approx(0.7)


def test_a_seed_on_a_small_part_of_a_large_segment_fails_iou():
    # Tree-line returns behind a far car: the seed is 5 of a 40-row cluster.
    scale = _scale([0] * 40)
    v = verify({"car": _rows((0, 5))}, {"car": _rows((0, 40))}, [scale], 40, CFG)["car"]
    assert (v.verified, v.status) == (False, IOU_NOT_ABOVE)
    assert v.iou == pytest.approx(5 / 40)


def test_iou_must_be_strictly_above_and_containment_at_least_the_threshold():
    scale = _scale([0] * 10)
    at_half = verify({"a": _rows((0, 5))}, {"a": _rows((0, 10))}, [scale], 10, CFG)
    assert at_half["a"].status == IOU_NOT_ABOVE  # IoU 0.5 is not above 0.5
    exact = verify({"a": _rows((0, 8))}, {"a": _rows((0, 8))}, [scale], 10, CFG)
    assert exact["a"].verified and exact["a"].containment == pytest.approx(0.8)


def test_a_finer_scale_verifies_what_a_coarse_one_merges_with_the_hedge():
    # Coarse: car rows 0-9 and hedge rows 10-29 are one cluster (IoU 1/3).
    # Fine: the car alone. FGR keeps the best component over its radii.
    coarse = _scale([0] * 30, radius=1.2)
    fine = _scale([0] * 10 + [1] * 20, radius=0.3)
    v = verify(
        {"car": _rows((0, 10))}, {"car": _rows((0, 12))}, [coarse, fine], 30, CFG
    )
    assert v["car"].verified and v["car"].radius_m == 0.3


def test_of_several_passing_scales_the_best_iou_one_is_named():
    # Coarse: the car (rows 0-9) and a neighbour's two rows (10-11) inside the
    # frustum, IoU 10/12, containment 1. Fine: the car alone, IoU 1. Both
    # pass; SAL takes the best IoU across its ensemble, not the first scale.
    coarse = _scale([0] * 12, radius=1.2)
    fine = _scale([0] * 10 + [-1] * 2, radius=0.3)
    v = verify(
        {"car": _rows((0, 10))}, {"car": _rows((0, 12))}, [coarse, fine], 12, CFG
    )["car"]
    assert v.verified and (v.radius_m, v.cluster_points) == (0.3, 10)
    assert v.iou == pytest.approx(1.0)


def test_equal_iou_passing_scales_name_the_first():
    a = _scale([0] * 10, radius=1.2)
    b = _scale([0] * 10, radius=0.3)
    v = verify({"car": _rows((0, 10))}, {"car": _rows((0, 10))}, [a, b], 10, CFG)
    assert v["car"].radius_m == 1.2


def test_a_failed_instance_reports_its_best_iou_scale():
    seed = _rows((0, 5), (10, 15))
    a = _scale([0] * 10 + [-1] * 10, radius=1.0)  # 5 of a 10-row cluster: IoU 1/3
    b = _scale([0] * 20, radius=0.5)  # all 10 in a 20-row cluster: IoU 0.5
    v = verify({"x": seed}, {"x": _rows((0, 20))}, [a, b], 20, CFG)["x"]
    assert not v.verified
    assert (v.radius_m, v.iou) == (0.5, pytest.approx(0.5))


def test_a_seed_of_noise_has_no_overlap():
    scale = _scale([-1] * 5 + [0] * 10)
    v = verify({"a": _rows((0, 5))}, {"a": _rows((0, 15))}, [scale], 15, CFG)["a"]
    assert (v.verified, v.status, v.cluster_id, v.containment) == (
        False,
        NO_OVERLAP,
        None,
        None,
    )


def test_without_a_bank_nothing_verifies():
    v = verify({"a": _rows((0, 5))}, {"a": _rows((0, 5))}, [], 5, CFG)["a"]
    assert v.status == NO_OVERLAP


def test_a_missing_frustum_contains_nothing():
    scale = _scale([0] * 10)
    v = verify({"a": _rows((0, 9))}, {}, [scale], 10, CFG)["a"]
    assert v.status == CONTAINMENT_BELOW and v.containment == 0.0


def _grid(cx, cy, lx, ly, lz, step=0.15):
    xs, ys, zs = (np.arange(-side / 2, side / 2 + 1e-9, step) for side in (lx, ly, lz))
    g = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)
    return g + [cx, cy, 0.0]


def test_a_dbscan_bank_separates_a_car_from_a_hedge_beside_it():
    """The six SAL radii over a real point layout: a car 1.1 m behind a hedge
    strip, merged with it only at the coarsest radius. The car's mask (rows of
    the car) verifies at a finer scale; a mask whose seed is the hedge's
    middle, inside a car-sized frustum, does not."""
    car = _grid(10.0, 0.0, 4.0, 1.8, 1.2)  # x 8.0 to 12.0
    hedge = _grid(6.6, 0.0, 0.6, 20.0, 1.0)  # x 6.3 to 6.9
    xyz = np.vstack([car, hedge])
    n_car = len(car)
    bank = build_bank(xyz, (1.2488, 0.8136, 0.6952, 0.594, 0.4353, 0.3221), 5)
    car_rows = np.arange(n_car)
    hedge_rows = np.arange(n_car, len(xyz))
    in_car_box = hedge_rows[np.abs(hedge[:, 1]) < 1.0]
    frustum = np.concatenate([car_rows, in_car_box])
    verdicts = verify(
        {"real": car_rows, "on_hedge": in_car_box},
        {"real": frustum, "on_hedge": frustum},
        bank,
        len(xyz),
        CFG,
    )
    assert verdicts["real"].verified
    assert not verdicts["on_hedge"].verified
