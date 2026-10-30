"""wato_common.artifacts.lifted_labels: what semantic_lifting writes,
consumers read back."""

from __future__ import annotations

import os

import numpy as np
import pytest

from wato_common.artifact_store import lifted_labels_path, local_path
from wato_common.artifacts import ArtifactFormatError, lifted_labels
from wato_common.schemas import LiftedStatsRow

BAG, CHUNK = "b", "0000"


@pytest.fixture(autouse=True)
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")


def _labels() -> lifted_labels.SweepLabels:
    return lifted_labels.SweepLabels(
        point_indices=np.array([0, 4, 9], np.int32),
        instance_ids=np.array(["obj3", "m0", "m0"]),
        classes=np.array(["pedestrian", "vehicle", "vehicle"]),
        confidences=np.array([0.9, 0.6, 0.55], np.float32),
        n_supporting_cameras=np.array([2, 1, 1], np.int32),
        n_disagreeing_cameras=np.array([0, 1, 0], np.int32),
    )


def test_sweep_round_trip_without_pickle():
    lifted_labels.write_sweep(BAG, CHUNK, 42, _labels())
    path = local_path(lifted_labels_path(BAG, CHUNK, 42))
    assert os.path.basename(path) == "000042.npz"  # same padding as every sweep file
    got = lifted_labels.read_sweep(BAG, CHUNK, 42, n_points=10)
    for name in ("point_indices", "instance_ids", "classes", "confidences",
                 "n_supporting_cameras", "n_disagreeing_cameras"):  # fmt: skip
        np.testing.assert_array_equal(getattr(got, name), getattr(_labels(), name))
    assert got.classes.dtype.kind == "U"  # loadable with allow_pickle=False


def test_empty_sweep_round_trips():
    lifted_labels.write_sweep(BAG, CHUNK, 1, lifted_labels.SweepLabels.empty())
    got = lifted_labels.read_sweep(BAG, CHUNK, 1, n_points=0)
    assert got.point_indices.shape == (0,) and got.classes.shape == (0,)


def test_indices_beyond_the_sweep_are_rejected():
    lifted_labels.write_sweep(BAG, CHUNK, 2, _labels())
    with pytest.raises(ArtifactFormatError, match="outside"):
        lifted_labels.read_sweep(BAG, CHUNK, 2, n_points=5)


def test_pickled_files_from_before_string_arrays_are_rejected():
    path = local_path(lifted_labels_path(BAG, CHUNK, 3))
    lifted_labels.write_sweep(BAG, CHUNK, 3, _labels())  # creates the directory
    old = {k: getattr(_labels(), k) for k in ("point_indices", "confidences",
           "n_supporting_cameras", "n_disagreeing_cameras")}  # fmt: skip
    np.savez_compressed(
        path,
        instance_ids=np.array(["m0"] * 3, dtype=object),
        classes=np.array(["vehicle"] * 3, dtype=object),
        **old,
    )
    with pytest.raises(ArtifactFormatError, match="re-run semantic_lifting"):
        lifted_labels.read_sweep(BAG, CHUNK, 3)


def test_stats_round_trip_with_int_sweep_id():
    row = LiftedStatsRow(
        sweep_id=42,
        bag_id=BAG,
        chunk_id=CHUNK,
        n_points_total=10,
        n_points_labeled=3,
        n_points_in_any_mask=4,
        n_points_failed_visibility=1,
        n_points_disagreement=1,
        mean_confidence_labeled=0.68,
        n_cameras_used=2,
    )
    assert not lifted_labels.exists(BAG, CHUNK)
    lifted_labels.write_stats(BAG, CHUNK, [row])
    assert lifted_labels.exists(BAG, CHUNK)
    assert lifted_labels.read_stats(BAG, CHUNK) == [row]


def test_instances_round_trip_with_sorted_lists():
    inst = lifted_labels.LiftedInstance(
        instance_id="a1",
        cls="car",
        masklet_ids=("c3", "a1"),
        cam_ids=("PANO_SE", "PANO_EE", "PANO_SE"),
        n_points_labeled=120,
        n_sweeps=14,
    )
    lifted_labels.write_instances(BAG, CHUNK, [inst])
    (got,) = lifted_labels.read_instances(BAG, CHUNK)
    assert got.masklet_ids == ("a1", "c3")
    assert got.cam_ids == ("PANO_EE", "PANO_SE")
    assert (got.instance_id, got.cls, got.n_points_labeled, got.n_sweeps) == (
        "a1",
        "car",
        120,
        14,
    )
    lifted_labels.write_instances(BAG, CHUNK, [])
    assert lifted_labels.read_instances(BAG, CHUNK) == []
