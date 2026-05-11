"""Tests for reduce.py (Step D — global static map reduce)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from wato_common.artifact_store import (
    chunks_index_path,
    global_static_map_path,
    local_path,
    static_map_path,
)
from wato_common.io.parquet_io import write_table
from wato_common.schemas import CHUNK_SCHEMA
from wato_lidar_preprocessing.config import ComponentConfig
from wato_lidar_preprocessing.reduce import _voxel_snap_downsample, reduce_static_map


@pytest.fixture()
def tmp_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", str(tmp_path))
    return tmp_path


def _write_chunk_index(bag_id: str, chunk_ids: list[str]):
    rows = [
        {
            "bag_id": bag_id, "chunk_id": cid,
            "t_start_ns": i * 30_000_000_000,
            "t_end_ns": (i + 1) * 30_000_000_000,
            "t_overlap_start_ns": i * 30_000_000_000,
            "t_overlap_end_ns": (i + 1) * 30_000_000_000,
        }
        for i, cid in enumerate(chunk_ids)
    ]
    write_table(rows, CHUNK_SCHEMA, chunks_index_path(bag_id))


def _write_static_map(bag_id: str, chunk_id: str, xyz: np.ndarray):
    path = local_path(static_map_path(bag_id, chunk_id))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, xyz=xyz, origin=np.zeros(3), voxel_size=np.float32(0.15))


def test_reduces_two_chunks(tmp_env):
    bag_id = "bag0"
    chunk_ids = ["chunk0", "chunk1"]
    _write_chunk_index(bag_id, chunk_ids)

    # Dense grid of points, voxel downsample should reduce count.
    rng = np.random.default_rng(0)
    xyz0 = rng.uniform(0, 10, size=(500, 3))
    xyz1 = rng.uniform(10, 20, size=(500, 3))
    _write_static_map(bag_id, "chunk0", xyz0)
    _write_static_map(bag_id, "chunk1", xyz1)

    cfg = ComponentConfig(global_map_voxel_size_m=0.5)
    out = reduce_static_map(bag_id, cfg)

    assert os.path.exists(local_path(global_static_map_path(bag_id)))
    data = np.load(local_path(out))
    assert data["xyz"].ndim == 2
    assert data["xyz"].shape[1] == 3
    # Downsampling must not increase point count.
    assert data["xyz"].shape[0] <= 1000


def test_skips_missing_chunks_gracefully(tmp_env):
    bag_id = "bag1"
    _write_chunk_index(bag_id, ["chunk0", "chunk1"])
    # Only write chunk0's static map; chunk1 is missing.
    xyz0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    _write_static_map(bag_id, "chunk0", xyz0)

    cfg = ComponentConfig()
    out = reduce_static_map(bag_id, cfg)
    data = np.load(local_path(out))
    assert data["xyz"].shape[0] > 0


def test_all_chunks_missing_writes_empty(tmp_env):
    bag_id = "bag2"
    _write_chunk_index(bag_id, ["chunk0"])
    # Don't write any static maps.
    cfg = ComponentConfig()
    out = reduce_static_map(bag_id, cfg)
    data = np.load(local_path(out))
    assert data["xyz"].shape == (0, 3)


def test_voxel_snap_one_point_per_voxel():
    """Many points within one voxel collapse to a single output point."""
    # 10 points all inside the [0, 0.5)^3 voxel at voxel_size=0.5.
    xyz = np.column_stack([
        np.linspace(0.0, 0.49, 10),
        np.linspace(0.0, 0.49, 10),
        np.linspace(0.0, 0.49, 10),
    ])
    out = _voxel_snap_downsample(xyz, voxel_size=0.5)
    assert out.shape == (1, 3)
    # Snapped to the cell centre (0.25, 0.25, 0.25).
    np.testing.assert_allclose(out[0], [0.25, 0.25, 0.25])


def test_voxel_snap_distinct_voxels_preserved():
    """Two points in distinct voxels stay distinct after snap."""
    xyz = np.array([[0.1, 0.0, 0.0], [10.0, 0.0, 0.0]])
    out = _voxel_snap_downsample(xyz, voxel_size=1.0)
    assert out.shape == (2, 3)
    # Sort by x to make order deterministic regardless of np.unique ordering.
    out_sorted = out[np.argsort(out[:, 0])]
    np.testing.assert_allclose(out_sorted[0], [0.5, 0.5, 0.5])
    np.testing.assert_allclose(out_sorted[1], [10.5, 0.5, 0.5])


def test_voxel_snap_negative_coordinates():
    """Negative-coordinate inputs (the typical SLAM map case) are handled."""
    xyz = np.array([[-5.0, -5.0, -5.0], [-5.1, -5.1, -5.1]])
    out = _voxel_snap_downsample(xyz, voxel_size=1.0)
    # Both points map to floor(-5/1) = -5 and floor(-5.1/1) = -6 → 2 voxels.
    assert out.shape == (2, 3)


def test_voxel_snap_empty():
    out = _voxel_snap_downsample(np.empty((0, 3)), voxel_size=0.5)
    assert out.shape == (0, 3)
