"""wato_common.artifacts.lidar_sweeps: raw sweeps (ingest → lidar_preprocessing)
and world sweeps + dynamic masks (lidar_preprocessing → consumers)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from wato_common.artifact_store import (
    dynamic_mask_path,
    lidar_dir,
    lidar_proc_dir,
    lidar_proc_index_path,
    local_path,
)
from wato_common.artifacts import ArtifactFormatError, lidar_sweeps as ls
from wato_common.io.parquet_io import write_table
from wato_common.schemas import PROCESSED_SWEEPS_SCHEMA, ProcessedSweepMeta

BAG, CHUNK = "b", "0000"


@pytest.fixture(autouse=True)
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")
    os.makedirs(local_path(lidar_dir(BAG, CHUNK, "lidar_cc")))
    os.makedirs(local_path(lidar_proc_dir(BAG, CHUNK)))


def test_raw_round_trip():
    raw = ls.RawSweep(
        x=np.array([1.0, np.nan, 3.0]),
        y=np.zeros(3),
        z=np.ones(3),
        intensity=np.array([5, 6, 7]),
        ring=np.array([0, 1, 31]),
        t_offset_us=np.array([0.0, 0.5, 1.0]),
    )
    uri = ls.write_raw(BAG, CHUNK, "lidar_cc", 12, raw)
    got = ls.read_raw(uri)
    assert got.x.dtype == np.float32 and got.ring.dtype == np.uint16
    assert got.t_offset_us.dtype == np.float64
    np.testing.assert_array_equal(got.x, raw.x.astype(np.float32))  # NaN kept
    assert got.n_points == 3

    bare = ls.read_raw(
        ls.write_raw(BAG, CHUNK, "lidar_cc", 13, ls.RawSweep(*([np.ones(2)] * 3)))
    )
    assert bare.intensity is None and bare.ring is None and bare.t_offset_us is None


def _world(n=4, **kw) -> ls.WorldSweep:
    return ls.WorldSweep(
        x=np.arange(n, dtype=np.float64),
        y=np.arange(n, dtype=np.float64) + 10,
        z=np.zeros(n),
        origin=np.array([1.0, 2.0, 0.5]),
        **kw,
    )


def test_world_round_trip():
    ground = np.array([True, False, False, True])
    uri = ls.write_world(
        BAG, CHUNK, 7, _world(ground_mask=ground, intensity=np.ones(4, np.float32))
    )
    got = ls.read_world(uri)
    assert got.xyz.shape == (4, 3) and got.xyz.dtype == np.float64
    np.testing.assert_array_equal(got.xyz[:, 1], np.arange(4) + 10)
    np.testing.assert_array_equal(got.ground_mask, ground)
    np.testing.assert_array_equal(got.origin, [1.0, 2.0, 0.5])
    assert got.ring is None


def test_world_arrays_must_share_n():
    with pytest.raises(ValueError, match="ground_mask"):
        ls.write_world(BAG, CHUNK, 7, _world(ground_mask=np.zeros(3, bool)))


def test_dynamic_mask_must_be_bool_and_aligned():
    with pytest.raises(ValueError, match="1-D bool"):
        ls.write_dynamic_mask(BAG, CHUNK, 7, np.zeros(4, np.uint8))
    uri = ls.write_dynamic_mask(BAG, CHUNK, 7, np.array([True, False, False]))
    np.testing.assert_array_equal(
        ls.read_dynamic_mask(uri, n_points=3), [True, False, False]
    )
    with pytest.raises(ArtifactFormatError, match="world sweep has 4"):
        ls.read_dynamic_mask(uri, n_points=4)


def _index(rows: list[dict]) -> None:
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(BAG, CHUNK))


def _meta(sweep_id: int, world_uri: str, valid: bool = True) -> dict:
    return ProcessedSweepMeta(
        bag_id=BAG,
        chunk_id=CHUNK,
        sweep_id=sweep_id,
        lidar_id="lidar_cc",
        reference_timestamp_ns=sweep_id * 100,
        n_points_total=4,
        n_points_static=3,
        n_points_dynamic=1,
        world_path=world_uri if valid else "",
        dynamic_mask_path=dynamic_mask_path(BAG, CHUNK, sweep_id) if valid else "",
        has_intensity=False,
        deskewed=True,
        valid=valid,
    ).model_dump()


def test_processed_sweeps_index_world_and_mask():
    w = ls.write_world(BAG, CHUNK, 1, _world())
    ls.write_dynamic_mask(BAG, CHUNK, 1, np.array([False, True, False, False]))
    _index([_meta(0, "", valid=False), _meta(1, w)])

    sweeps = ls.ProcessedSweeps.load(BAG, CHUNK)
    assert [r.sweep_id for r in sweeps.rows] == [1]  # invalid rows dropped
    assert sweeps.get(0) is None
    world, dynamic = sweeps.world_and_dynamic_mask(1)
    static = world.xyz[~dynamic]
    assert static.shape == (3, 3) and 1.0 not in static[:, 0]


def test_misaligned_dynamic_mask_is_caught():
    w = ls.write_world(BAG, CHUNK, 2, _world())
    ls.write_dynamic_mask(BAG, CHUNK, 2, np.zeros(3, bool))  # world has 4 points
    _index([_meta(2, w)])
    with pytest.raises(ArtifactFormatError, match="world sweep has 4"):
        ls.ProcessedSweeps.load(BAG, CHUNK).world_and_dynamic_mask(2)


def test_empty_sweep_has_an_empty_mask():
    w = ls.write_world(BAG, CHUNK, 3, _world(n=0))
    ls.write_dynamic_mask(BAG, CHUNK, 3, np.zeros(0, bool))
    _index([_meta(3, w)])
    world, dynamic = ls.ProcessedSweeps.load(BAG, CHUNK).world_and_dynamic_mask(3)
    assert world.n_points == 0 and dynamic.shape == (0,)
