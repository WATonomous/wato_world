"""The retained hypothesis relation permits overlap but rejects corruption."""

from dataclasses import replace

import numpy as np
import pytest

from wato_common.artifact_store import lifted_candidates_path, local_path
from wato_common.artifacts import ArtifactFormatError, lifted_candidates


@pytest.fixture(autouse=True)
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")


def _claims():
    return lifted_candidates.SweepCandidates(
        np.array([0, 0, 4], np.int32),
        np.array(["a", "b", "a"]),
        np.array([0.8, 0.6, 1.0], np.float32),
    )


def test_competing_claims_round_trip_without_pickle():
    lifted_candidates.write_sweep("b", "0000", 42, _claims())
    got = lifted_candidates.read_sweep("b", "0000", 42, n_points=5)
    for field in ("point_indices", "instance_ids", "confidences"):
        np.testing.assert_array_equal(getattr(got, field), getattr(_claims(), field))
    assert got.instance_ids.dtype.kind == "U"
    assert local_path(lifted_candidates_path("b", "0000", 42)).endswith(
        "lifted_candidates/000042.npz"
    )


def test_empty_relation_round_trips():
    lifted_candidates.write_sweep(
        "b", "0000", 0, lifted_candidates.SweepCandidates.empty()
    )
    assert (
        lifted_candidates.read_sweep("b", "0000", 0, n_points=0).point_indices.size == 0
    )


@pytest.mark.parametrize(
    "claims, message",
    [
        (replace(_claims(), instance_ids=np.array(["a", "a", "b"])), "duplicate"),
        (replace(_claims(), point_indices=np.array([-1, 0, 4])), "outside"),
        (replace(_claims(), instance_ids=np.array(["", "b", "a"])), "nonempty"),
        (replace(_claims(), confidences=np.array([np.nan, 0.5, 1.0])), "finite"),
        (replace(_claims(), confidences=np.array([0.0, 1.1, 1.0])), "within"),
        (replace(_claims(), confidences=np.zeros(2)), "shape"),
    ],
)
def test_invalid_candidate_relation_is_rejected(claims, message):
    with pytest.raises(ValueError, match=message):
        lifted_candidates.write_sweep("b", "0000", 0, claims)


def test_reader_checks_world_indices_and_missing_legacy_output():
    with pytest.raises(ArtifactFormatError, match="re-run semantic_lifting"):
        lifted_candidates.read_sweep("b", "0000", 0)
    lifted_candidates.write_sweep("b", "0000", 0, _claims())
    with pytest.raises(ArtifactFormatError, match="outside"):
        lifted_candidates.read_sweep("b", "0000", 0, n_points=4)


def test_reader_rejects_pickled_strings():
    lifted_candidates.write_sweep("b", "0000", 0, _claims())
    path = local_path(lifted_candidates_path("b", "0000", 0))
    np.savez_compressed(
        path,
        point_indices=_claims().point_indices,
        instance_ids=np.array(["a", "b", "a"], dtype=object),
        confidences=_claims().confidences,
    )
    with pytest.raises(ArtifactFormatError, match="re-run semantic_lifting"):
        lifted_candidates.read_sweep("b", "0000", 0)
