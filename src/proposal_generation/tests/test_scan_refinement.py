"""Point-set behavior of SAL matching and standard DBSCAN density expansion."""

import numpy as np
import pytest

from wato_proposal_generation.config import ScanRefinementConfig
from wato_proposal_generation.scan_refinement import refine as refine_seeds


def refine(xyz, instance_id, cfg):
    """Existing exclusive seed fixtures are a subset of independent claims."""
    return refine_seeds(
        xyz,
        {
            str(iid): np.flatnonzero(instance_id == iid)
            for iid in np.unique(instance_id[instance_id != ""])
        },
        cfg,
    )


def _line(x):
    return np.column_stack([x, np.zeros(len(x)), np.ones(len(x))])


@pytest.mark.parametrize("n_seed, replaced", [(3, False), (5, False), (6, True)])
def test_strict_iou_gate_keeps_original_support_at_or_below_half(n_seed, replaced):
    xyz = _line(np.arange(10) * 0.01)
    ids = np.array(["i"] * n_seed + [""] * (10 - n_seed))
    result = refine(xyz, ids, ScanRefinementConfig())["i"]
    assert result.iou == pytest.approx(n_seed / 10)
    assert result.status == ("replaced" if replaced else "iou_not_above_threshold")
    assert result.point_indices.tolist() == list(range(10 if replaced else n_seed))


def test_best_iou_can_choose_a_finer_scale_over_a_larger_merged_cluster():
    xyz = _line(np.r_[np.arange(8) * 0.01, 0.8 + np.arange(8) * 0.01])
    ids = np.array(["i"] * 6 + [""] * 10)
    result = refine(xyz, ids, ScanRefinementConfig(radii_m=(1.0, 0.15)))["i"]
    assert result.radius_m == 0.15 and result.iou == pytest.approx(6 / 8)
    assert result.point_indices.tolist() == list(range(8))


def test_sparse_noncore_chain_does_not_connect_two_dense_objects():
    xyz = _line(
        np.r_[np.arange(6) * 0.01, 4 + np.arange(6) * 0.01, [1.0, 1.75, 2.5, 3.25]]
    )
    ids = np.array(["left"] * 6 + ["right"] * 6 + [""] * 4)
    masks = refine(xyz, ids, ScanRefinementConfig(radii_m=(1.0,), min_samples=5))
    assert masks["left"].point_indices.tolist() == list(range(6)) + [12, 13]
    assert masks["right"].point_indices.tolist() == list(range(6, 12)) + [14, 15]
    assert not np.intersect1d(
        masks["left"].point_indices, masks["right"].point_indices
    ).size


def test_density_connected_chain_can_still_merge_objects():
    xyz = _line(np.arange(12) * 0.5)
    ids = np.array(["i"] * 8 + [""] * 4)
    result = refine(xyz, ids, ScanRefinementConfig(radii_m=(0.6,), min_samples=2))["i"]
    assert result.point_indices.tolist() == list(range(12))


def test_border_tie_goes_to_first_core_component_in_input_order():
    xyz = _line(np.r_[np.arange(8) * 0.01 - 0.06, np.arange(8) * 0.01 + 1.99, 1.0])
    ids = np.array(["left"] * 8 + ["right"] * 8 + [""])
    cfg = ScanRefinementConfig(radii_m=(1.0,), min_samples=8)
    masks = refine(xyz, ids, cfg)
    assert 16 in masks["left"].point_indices
    assert 16 not in masks["right"].point_indices
    order = np.r_[np.arange(8, 16), np.arange(8), 16]
    masks = refine(xyz[order], ids[order], cfg)
    assert 16 in masks["right"].point_indices
    assert 16 not in masks["left"].point_indices


def test_independent_matching_does_not_veto_cross_scale_overlaps():
    # At the fine scale only four B plus one A form a cluster. A's other
    # seeds are noise. The coarse cluster contains all six A and four B seeds.
    xyz = _line(np.r_[np.arange(4) * 0.01, 0.04, [0.5, 0.8, 1.1, 1.4, 1.7]])
    ids = np.array(["B"] * 4 + ["A"] * 6)
    masks = refine(xyz, ids, ScanRefinementConfig(radii_m=(2.0, 0.1)))
    assert masks["A"].point_indices.tolist() == list(range(10))
    assert masks["B"].point_indices.tolist() == list(range(5))
    assert masks["A"].iou == pytest.approx(0.6)
    assert masks["B"].iou == pytest.approx(0.8)


def test_coincident_returns_are_distinct_indices_and_count_self():
    masks = refine(
        np.zeros((5, 3)),
        np.array(["i"] * 3 + [""] * 2),
        ScanRefinementConfig(min_samples=5),
    )
    assert masks["i"].point_indices.tolist() == list(range(5))
    assert masks["i"].iou == pytest.approx(0.6)


def test_noise_and_disabled_refinement_keep_every_original_point():
    xyz = _line([0.0, 5.0, 10.0])
    ids = np.array(["i"] * 3)
    for cfg, status in (
        (ScanRefinementConfig(), "no_overlap"),
        (ScanRefinementConfig(enabled=False), "disabled"),
    ):
        mask = refine(xyz, ids, cfg)["i"]
        assert mask.point_indices.tolist() == [0, 1, 2]
        assert mask.status == status and mask.radius_m is None
    assert refine(np.empty((0, 3)), np.array([], dtype=str), cfg) == {}
    assert refine(xyz, np.array([""] * 3), cfg) == {}


def test_failed_match_keeps_all_original_components_and_noise():
    xyz = _line(np.r_[np.arange(6) * 0.01, 5 + np.arange(6) * 0.01, 20.0])
    mask = refine(xyz, np.array(["i"] * 13), ScanRefinementConfig())["i"]
    assert mask.iou == pytest.approx(6 / 13)
    assert mask.point_indices.tolist() == list(range(13))


def test_equal_iou_retains_published_scale_order():
    xyz = _line(np.arange(6) * 0.01)
    mask = refine(xyz, np.array(["i"] * 6), ScanRefinementConfig())["i"]
    assert mask.radius_m == 1.2488


def test_competing_original_seeds_can_both_refine_against_the_same_bank():
    xyz = _line(np.arange(10) * 0.01)
    masks = refine_seeds(
        xyz,
        {"left": np.arange(6), "right": np.arange(4, 10)},
        ScanRefinementConfig(),
    )
    for mask in masks.values():
        assert mask.status == "replaced"
        assert mask.iou == pytest.approx(0.6)
        assert mask.point_indices.tolist() == list(range(10))
