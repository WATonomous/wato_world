"""Late flattening preserves recoverable support and enforces unique owners."""

import numpy as np
import pytest

from wato_proposal_generation.overlap import resolve_support


def test_duplicate_smaller_hypothesis_is_suppressed_after_refinement():
    result = resolve_support(
        {"duplicate": np.arange(2, 7), "object": np.arange(10)}, min_points=1
    )
    assert list(result) == ["object"]
    assert result["object"].tolist() == list(range(10))


def test_partial_overlap_keeps_both_objects_with_unique_point_owners():
    result = resolve_support(
        {"larger": np.arange(6), "smaller": np.arange(4, 9)}, min_points=3
    )
    assert result["larger"].tolist() == list(range(6))
    assert result["smaller"].tolist() == [6, 7, 8]
    assert np.intersect1d(*result.values()).size == 0


def test_rejected_small_residual_neither_claims_nor_suppresses_later_support():
    # B has two unowned points, below the floor. C needs those same points.
    # B's IoM with C is 1, so keeping B in the NMS set would also lose C.
    result = resolve_support(
        {
            "A": np.arange(10),
            "B": np.array([0, 1, 2, 3, 4, 5, 6, 10, 11]),
            "C": np.array([6, 10, 11, 12]),
        },
        min_points=3,
        max_iom=0.8,
    )
    assert list(result) == ["A", "C"]
    assert result["C"].tolist() == [10, 11, 12]


def test_iom_threshold_is_strict():
    supports = {"A": np.arange(10), "B": np.array([0, 1, 10, 11])}
    at = resolve_support(supports, min_points=1)
    above = resolve_support(supports, min_points=1, max_iom=0.49)
    assert at["B"].tolist() == [10, 11]
    assert list(above) == ["A"]


def test_union_coverage_does_not_suppress_a_residual_above_the_point_floor():
    # Neither pair exceeds 0.5 IoM; the union covers 4/5 of C. Pairwise NMS
    # retains C's fifth point, rather than treating separate objects as one.
    supports = {
        "A": np.arange(6),
        "B": np.arange(6, 12),
        "C": np.array([0, 1, 6, 7, 12]),
    }
    result = resolve_support(supports, min_points=1)
    assert result["C"].tolist() == [12]


def test_threshold_one_disables_suppression_but_keeps_exclusive_assignment():
    supports = {"A": np.arange(6), "B": np.arange(3, 7)}
    assert list(resolve_support(supports, min_points=1)) == ["A"]
    result = resolve_support(supports, min_points=1, max_iom=1)
    assert result["B"].tolist() == [6]


def test_threshold_zero_suppresses_any_overlap_but_keeps_disjoint_support():
    result = resolve_support(
        {"A": np.arange(6), "B": np.arange(5, 8), "C": np.arange(8, 10)},
        min_points=1,
        max_iom=0,
    )
    assert list(result) == ["A", "C"]


def test_rejected_geometry_neither_owns_points_nor_suppresses_duplicates():
    calls = []

    def accept(iid, remaining, accepted):
        calls.append((iid, remaining.tolist(), list(accepted)))
        return iid != "B"

    result = resolve_support(
        {"A": np.arange(10), "B": np.arange(10, 19), "C": np.arange(10, 15)},
        min_points=3,
        accept=accept,
    )
    assert list(result) == ["A", "C"]
    assert result["C"].tolist() == list(range(10, 15))
    assert calls == [
        ("A", list(range(10)), []),
        ("B", list(range(10, 19)), ["A"]),
        ("C", list(range(10, 15)), ["A"]),
    ]


def test_geometry_predicate_sees_final_support_and_only_accepted_hypotheses():
    seen = []

    def accept(iid, remaining, accepted):
        seen.append(
            (iid, remaining.tolist(), {k: v.tolist() for k, v in accepted.items()})
        )
        return True

    result = resolve_support(
        {"A": np.arange(6), "B": np.arange(4, 9)},
        min_points=3,
        accept=accept,
    )
    assert result["B"].tolist() == [6, 7, 8]
    assert seen[-1] == ("B", [6, 7, 8], {"A": list(range(6))})


def test_ties_and_input_permutations_are_deterministic_without_mutation():
    supports = {
        "B": np.array([6, 5, 4, 3, 2]),
        "A": np.array([4, 3, 2, 1, 0]),
    }
    before = {iid: indices.copy() for iid, indices in supports.items()}
    first = resolve_support(supports, min_points=1, max_iom=0.8)
    second = resolve_support(
        dict(reversed(list(supports.items()))), min_points=1, max_iom=0.8
    )
    assert list(first) == list(second) == ["A", "B"]
    assert first["A"].tolist() == [0, 1, 2, 3, 4]
    assert first["B"].tolist() == [5, 6]
    for iid in supports:
        np.testing.assert_array_equal(first[iid], second[iid])
        np.testing.assert_array_equal(supports[iid], before[iid])


def test_repeated_indices_do_not_inflate_priority_or_support():
    result = resolve_support(
        {"B": np.array([0, 0, 0, 0]), "A": np.array([0, 1])}, min_points=1
    )
    assert list(result) == ["A"]
    assert result["A"].tolist() == [0, 1]


def test_empty_and_below_floor_supports_have_no_owners():
    assert resolve_support({}, min_points=1) == {}
    assert resolve_support({"A": np.array([])}, min_points=1) == {}
    assert resolve_support({"A": np.arange(2)}, min_points=3) == {}


@pytest.mark.parametrize("min_points", [0, -1, 1.5, True])
def test_invalid_point_floor_is_rejected(min_points):
    with pytest.raises(ValueError, match="min_points"):
        resolve_support({}, min_points=min_points)


@pytest.mark.parametrize("max_iom", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_iom_threshold_is_rejected(max_iom):
    with pytest.raises(ValueError, match="max_iom"):
        resolve_support({}, min_points=1, max_iom=max_iom)


@pytest.mark.parametrize("indices", [np.array([-1]), np.array([1.5]), np.zeros((1, 1))])
def test_invalid_support_indices_are_rejected(indices):
    with pytest.raises(ValueError, match="nonnegative integer"):
        resolve_support({"A": indices}, min_points=1)
