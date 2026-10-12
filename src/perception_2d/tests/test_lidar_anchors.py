"""Depth anchors: the static points perception_2d reads from lidar_preprocessing
are the complement of the dynamic mask, read through
wato_common.artifacts.lidar_sweeps."""

from __future__ import annotations

import os

import numpy as np
import pytest

from wato_common.artifact_store import (
    lidar_proc_dir,
    lidar_proc_index_path,
    local_path,
)
from wato_common.artifacts import lidar_sweeps
from wato_common.io.parquet_io import write_table
from wato_common.schemas import PROCESSED_SWEEPS_SCHEMA, ProcessedSweepMeta
from wato_perception_2d import io

BAG, CHUNK = "b", "0000"


@pytest.fixture(autouse=True)
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")
    os.makedirs(local_path(lidar_proc_dir(BAG, CHUNK)))
    yield
    io.clear_lidar_caches(BAG, CHUNK)


def _stage(sweep_id: int, dynamic: np.ndarray, n: int = 4) -> None:
    world = lidar_sweeps.write_world(
        BAG,
        CHUNK,
        sweep_id,
        lidar_sweeps.WorldSweep(
            x=np.arange(n, dtype=np.float64), y=np.zeros(n), z=np.zeros(n)
        ),
    )
    mask = lidar_sweeps.write_dynamic_mask(BAG, CHUNK, sweep_id, dynamic)
    write_table(
        [
            ProcessedSweepMeta(
                bag_id=BAG,
                chunk_id=CHUNK,
                sweep_id=sweep_id,
                lidar_id="lidar_cc",
                reference_timestamp_ns=0,
                n_points_total=n,
                n_points_static=int((~dynamic).sum()),
                n_points_dynamic=int(dynamic.sum()),
                world_path=world,
                dynamic_mask_path=mask,
                has_intensity=False,
                deskewed=True,
            ).model_dump()
        ],
        PROCESSED_SWEEPS_SCHEMA,
        lidar_proc_index_path(BAG, CHUNK),
    )


def test_static_anchors_exclude_dynamic_points():
    _stage(5, np.array([False, True, False, True]))
    static = io.load_static_lidar_points(BAG, CHUNK, 5)
    assert static[:, 0].tolist() == [0.0, 2.0]
    assert io.load_static_lidar_points(BAG, CHUNK, 6) is None  # not indexed
