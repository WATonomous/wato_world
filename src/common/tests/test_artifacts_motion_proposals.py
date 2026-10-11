"""wato_common.artifacts.motion_proposals: what lidar_preprocessing Step F
writes, consumers read back."""

from __future__ import annotations

import numpy as np
import pytest

from wato_common.artifact_store import local_path, motion_proposals_path
from wato_common.artifacts import ArtifactFormatError, motion_proposals as mp
from wato_common.schemas import MotionClusterRow

BAG, CHUNK = "b", "0000"


@pytest.fixture(autouse=True)
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")
    (tmp_path / "raw" / BAG / "chunks" / CHUNK / "lidar_proc").mkdir(parents=True)


def test_sweep_round_trip():
    bits = np.array([mp.AW_DYNAMIC | mp.UNMAPPED, 0, mp.BOX_FILL], dtype=np.uint8)
    cid = np.array([4, -1, 4], dtype=np.int32)
    mp.write_sweep(BAG, CHUNK, 7, bits, cid)
    got = mp.read_sweep(BAG, CHUNK, 7, n_points=3)
    np.testing.assert_array_equal(got.source_bits, bits)
    np.testing.assert_array_equal(got.cluster_id, cid)
    assert got.source_bits.dtype == np.uint8 and got.cluster_id.dtype == np.int32

    mp.remove_sweep(BAG, CHUNK, 7)
    with pytest.raises(FileNotFoundError):
        mp.read_sweep(BAG, CHUNK, 7)


def test_sweep_misalignment_is_caught():
    with pytest.raises(ValueError, match="same"):
        mp.write_sweep(BAG, CHUNK, 1, np.zeros(3, np.uint8), np.zeros(2, np.int32))
    mp.write_sweep(BAG, CHUNK, 1, np.zeros(3, np.uint8), np.zeros(3, np.int32))
    with pytest.raises(ArtifactFormatError, match="world sweep has 4"):
        mp.read_sweep(BAG, CHUNK, 1, n_points=4)


def test_sweep_with_foreign_dtypes_is_rejected():
    np.savez_compressed(
        local_path(motion_proposals_path(BAG, CHUNK, 2)),
        source_bits=np.zeros(3, np.int64),
        cluster_id=np.zeros(3, np.int32),
    )
    with pytest.raises(ArtifactFormatError, match="uint8/int32"):
        mp.read_sweep(BAG, CHUNK, 2)


def _cluster(**overrides) -> mp.MotionCluster:
    fields = dict(
        frame_id=3,
        reference_timestamp_ns=1_000,
        sweep_ids=(5, 6),
        cluster_id=4,
        track_hint_id=-1,
        cx=1.0,
        cy=2.0,
        cz=0.5,
        w=1.8,
        l=4.2,
        h=1.5,
        heading=0.1,
        n_points=120,
        n_seed_points=80,
        source_bits=int(mp.AW_DYNAMIC | mp.SEG_DYNAMIC),
        n_sources=2,
        frac_seg_dynamic=0.7,
        frac_persistent=0.05,
        track_life=9,
        net_displacement_m=8.0,
        motion_score=1.9,
        box_filled=True,
    )
    fields.update(overrides)
    return mp.MotionCluster(**fields)


def test_clusters_round_trip():
    clusters = [_cluster(), _cluster(cluster_id=5, sweep_ids=(), box_filled=False)]
    mp.write_clusters(BAG, CHUNK, clusters)
    assert mp.read_clusters(BAG, CHUNK) == clusters
    mp.write_clusters(BAG, CHUNK, [])
    assert mp.read_clusters(BAG, CHUNK) == []


def test_cluster_record_tracks_the_row_schema():
    """A column added to MotionClusterRow must be added to MotionCluster too."""
    record = {f for f in mp.MotionCluster.__dataclass_fields__}
    assert record == set(MotionClusterRow.model_fields) - {"bag_id", "chunk_id"}


def test_bit_roles_and_decode():
    assert int(mp.SEED_BITS) & int(mp.ATTACH_BITS) == 0
    assert int(mp.BOX_FILL) & (int(mp.SEED_BITS) | int(mp.ATTACH_BITS)) == 0
    decoded = mp.decode_bits(np.array([mp.MF_MOS | mp.UNMAPPED, 0], np.uint8))
    assert decoded["MF_MOS"].tolist() == [True, False]
    assert decoded["UNMAPPED"].tolist() == [True, False]
    assert not decoded["BOX_FILL"].any()
    assert set(decoded) == set(mp.BIT_NAMES.values())
