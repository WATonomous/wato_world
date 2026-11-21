"""The retained hypothesis relations permit overlap but reject corruption."""

from dataclasses import replace

import numpy as np
import pytest

from wato_common.artifact_store import lifted_candidates_path, local_path
from wato_common.artifacts import ArtifactFormatError, lifted_candidates


@pytest.fixture(autouse=True)
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")


def _claims():
    # Point 0 claimed by "a" and "b"; point 4 by "a". The frustum also holds
    # point 2, inside a's rectangle but hidden (no claim), and point 3 in b's.
    return lifted_candidates.SweepCandidates(
        np.array([0, 0, 4], np.int32),
        np.array(["a", "b", "a"]),
        np.array([0.8, 0.6, 1.0], np.float32),
        np.array([0, 2, 4, 0, 3], np.int32),
        np.array(["a", "a", "a", "b", "b"]),
    )


FIELDS = (
    "point_indices",
    "instance_ids",
    "confidences",
    "frustum_point_indices",
    "frustum_instance_ids",
)


def test_competing_claims_and_frustum_round_trip_without_pickle():
    lifted_candidates.write_sweep("b", "0000", 42, _claims())
    got = lifted_candidates.read_sweep("b", "0000", 42, n_points=5)
    for field in FIELDS:
        np.testing.assert_array_equal(getattr(got, field), getattr(_claims(), field))
    assert got.instance_ids.dtype.kind == "U"
    assert got.frustum_instance_ids.dtype.kind == "U"
    assert local_path(lifted_candidates_path("b", "0000", 42)).endswith(
        "lifted_candidates/000042.npz"
    )


def test_empty_relation_round_trips():
    lifted_candidates.write_sweep(
        "b", "0000", 0, lifted_candidates.SweepCandidates.empty()
    )
    got = lifted_candidates.read_sweep("b", "0000", 0, n_points=0)
    assert got.point_indices.size == 0 and got.frustum_point_indices.size == 0


def test_a_frustum_without_claims_round_trips():
    # Every return in the rectangle hidden behind an occluder: no candidates.
    claims = replace(
        lifted_candidates.SweepCandidates.empty(),
        frustum_point_indices=np.array([1, 2], np.int32),
        frustum_instance_ids=np.array(["a", "a"]),
    )
    lifted_candidates.write_sweep("b", "0000", 0, claims)
    got = lifted_candidates.read_sweep("b", "0000", 0, n_points=3)
    np.testing.assert_array_equal(got.frustum_point_indices, [1, 2])


def test_ids_of_different_widths_compare_as_pairs():
    claims = lifted_candidates.SweepCandidates(
        np.array([1], np.int32),
        np.array(["abc"]),
        np.array([1.0], np.float32),
        np.array([1, 1], np.int32),
        np.array(["abc", "a-much-longer-id"]),
    )
    lifted_candidates.write_sweep("b", "0000", 0, claims)


@pytest.mark.parametrize(
    "claims, message",
    [
        (replace(_claims(), instance_ids=np.array(["a", "a", "b"])), "duplicate"),
        (replace(_claims(), point_indices=np.array([-1, 0, 4])), "outside"),
        (replace(_claims(), instance_ids=np.array(["", "b", "a"])), "nonempty"),
        (replace(_claims(), confidences=np.array([np.nan, 0.5, 1.0])), "finite"),
        (replace(_claims(), confidences=np.array([0.0, 1.1, 1.0])), "within"),
        (replace(_claims(), confidences=np.zeros(2)), "shape"),
        (
            replace(
                _claims(), frustum_instance_ids=np.array(["a", "a", "a", "a", "b"])
            ),
            "duplicate",
        ),
        (
            replace(_claims(), frustum_point_indices=np.array([0, 2, 4, 0, -3])),
            "outside",
        ),
        (
            replace(_claims(), frustum_instance_ids=np.array(["a", "a", "a", "", "b"])),
            "nonempty",
        ),
        (replace(_claims(), frustum_instance_ids=np.array(["a", "b"])), "shape"),
        # b claims point 0, but 0 is not in b's frustum.
        (
            replace(
                _claims(),
                frustum_point_indices=np.array([0, 2, 4, 3], np.int32),
                frustum_instance_ids=np.array(["a", "a", "a", "b"]),
            ),
            "outside their instance's frustum",
        ),
    ],
)
def test_invalid_relations_are_rejected(claims, message):
    with pytest.raises(ValueError, match=message):
        lifted_candidates.write_sweep("b", "0000", 0, claims)


def test_reader_checks_world_indices_and_missing_legacy_output():
    with pytest.raises(ArtifactFormatError, match="re-run semantic_lifting"):
        lifted_candidates.read_sweep("b", "0000", 0)
    lifted_candidates.write_sweep("b", "0000", 0, _claims())
    with pytest.raises(ArtifactFormatError, match="outside"):
        lifted_candidates.read_sweep("b", "0000", 0, n_points=4)


def test_reader_rejects_output_written_before_the_frustum():
    lifted_candidates.write_sweep("b", "0000", 0, _claims())
    path = local_path(lifted_candidates_path("b", "0000", 0))
    np.savez_compressed(
        path,
        point_indices=_claims().point_indices,
        instance_ids=_claims().instance_ids,
        confidences=_claims().confidences,
    )
    with pytest.raises(ArtifactFormatError, match="frustum.*re-run semantic_lifting"):
        lifted_candidates.read_sweep("b", "0000", 0)


@pytest.mark.parametrize("field", ["instance_ids", "frustum_instance_ids"])
def test_reader_rejects_pickled_strings(field):
    lifted_candidates.write_sweep("b", "0000", 0, _claims())
    path = local_path(lifted_candidates_path("b", "0000", 0))
    arrays = {f: getattr(_claims(), f) for f in FIELDS}
    arrays[field] = arrays[field].astype(object)
    np.savez_compressed(path, **arrays)
    with pytest.raises(ArtifactFormatError, match="pickle.*re-run semantic_lifting"):
        lifted_candidates.read_sweep("b", "0000", 0)
