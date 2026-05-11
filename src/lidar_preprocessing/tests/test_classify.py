"""Tests for classify.py (Step B — voxel static/dynamic decomposition)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from wato_common.artifact_store import (
    dynamic_mask_path,
    lidar_proc_index_path,
    lidar_world_path,
    local_path,
    static_map_path,
)
from wato_common.io.parquet_io import write_table
from wato_common.schemas import PROCESSED_SWEEPS_SCHEMA
from wato_lidar_preprocessing.classify import process_chunk
from wato_lidar_preprocessing.config import ComponentConfig


@pytest.fixture()
def tmp_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", str(tmp_path))
    return tmp_path


def _write_world_sweep(bag_id: str, chunk_id: str, sweep_id: int, xyz: np.ndarray):
    path = local_path(lidar_world_path(bag_id, chunk_id, sweep_id))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2])


def _proc_row(
    bag_id: str, chunk_id: str, sweep_id: int, xyz: np.ndarray,
    *, has_intensity: bool = False,
) -> dict:
    """Build a parquet row that mirrors what real deskew would write."""
    n = xyz.shape[0]
    return {
        "bag_id": bag_id, "chunk_id": chunk_id,
        "sweep_id": sweep_id, "lidar_id": "LIDAR_TOP",
        "reference_timestamp_ns": sweep_id * 100_000_000,
        "n_points_total": n, "n_points_static": 0, "n_points_dynamic": 0,
        "world_path": lidar_world_path(bag_id, chunk_id, sweep_id),
        "dynamic_mask_path": "",
        "has_intensity": has_intensity, "deskewed": True,
        "world_xmin": float(xyz[:, 0].min()) if n else None,
        "world_xmax": float(xyz[:, 0].max()) if n else None,
        "world_ymin": float(xyz[:, 1].min()) if n else None,
        "world_ymax": float(xyz[:, 1].max()) if n else None,
        "world_zmin": float(xyz[:, 2].min()) if n else None,
        "world_zmax": float(xyz[:, 2].max()) if n else None,
    }


def _write_proc_index(bag_id: str, chunk_id: str, sweep_ids: list[int],
                      xyz_per_sweep: list[np.ndarray] | None = None):
    if xyz_per_sweep is None:
        xyz_per_sweep = [np.empty((0, 3))] * len(sweep_ids)
    rows = [
        _proc_row(bag_id, chunk_id, sid, xyz)
        for sid, xyz in zip(sweep_ids, xyz_per_sweep)
    ]
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))


def test_all_sweeps_present_classified_static(tmp_env):
    """Points seen in every sweep → static; dynamic mask all False."""
    bag_id, chunk_id = "bag0", "chunk0"
    n_sweeps = 10
    # Same 3 points across all sweeps (definitely static).
    static_xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    for i in range(n_sweeps):
        _write_world_sweep(bag_id, chunk_id, i, static_xyz)
    _write_proc_index(
        bag_id, chunk_id, list(range(n_sweeps)),
        xyz_per_sweep=[static_xyz] * n_sweeps,
    )

    cfg = ComponentConfig(static_sweep_fraction=0.3, static_sweep_min=2)
    result = process_chunk(cfg, bag_id, chunk_id)

    assert result.n_dynamic == 0
    assert result.n_static == n_sweeps * 3

    for i in range(n_sweeps):
        mask = np.load(local_path(dynamic_mask_path(bag_id, chunk_id, i)))
        assert not mask.any(), f"sweep {i}: expected all static"

    static_data = np.load(local_path(static_map_path(bag_id, chunk_id)))
    assert static_data["xyz"].shape[0] == n_sweeps * 3


def test_single_sweep_classified_dynamic(tmp_env):
    """Points seen in only 1 out of 10 sweeps → dynamic."""
    bag_id, chunk_id = "bag1", "chunk0"
    n_sweeps = 10
    # Background static points present in all sweeps.
    static_xyz = np.array([[100.0, 0.0, 0.0], [101.0, 0.0, 0.0]])
    # Dynamic points present only in sweep 5.
    dynamic_xyz = np.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]])

    xyz_per_sweep: list[np.ndarray] = []
    for i in range(n_sweeps):
        if i == 5:
            xyz = np.concatenate([static_xyz, dynamic_xyz])
        else:
            xyz = static_xyz
        _write_world_sweep(bag_id, chunk_id, i, xyz)
        xyz_per_sweep.append(xyz)
    _write_proc_index(bag_id, chunk_id, list(range(n_sweeps)), xyz_per_sweep=xyz_per_sweep)

    cfg = ComponentConfig(static_sweep_fraction=0.3, static_sweep_min=2)
    process_chunk(cfg, bag_id, chunk_id)

    mask_5 = np.load(local_path(dynamic_mask_path(bag_id, chunk_id, 5)))
    # First 2 points are static (background), last 2 are dynamic.
    assert not mask_5[0] and not mask_5[1], "background should be static"
    assert mask_5[2] and mask_5[3], "one-off points should be dynamic"


def test_intensity_backfilled_when_first_sweep_lacks_it(tmp_env):
    """Sweep 0 has no intensity, sweep 1 does → output intensity aligns with xyz."""
    bag_id, chunk_id = "bag_mixed_int", "chunk0"
    n_sweeps = 5
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    # All sweeps share the same xyz so they're all classified static.
    for i in range(n_sweeps):
        path = local_path(lidar_world_path(bag_id, chunk_id, i))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        kwargs = {"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2]}
        # Sweeps 0,1 have no intensity; 2,3,4 do.
        if i >= 2:
            kwargs["intensity"] = np.array([0.5, 0.7], dtype=np.float32)
        np.savez_compressed(path, **kwargs)

    rows = [
        _proc_row(bag_id, chunk_id, sid, pts, has_intensity=sid >= 2)
        for sid in range(n_sweeps)
    ]
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    cfg = ComponentConfig(static_sweep_fraction=0.3, static_sweep_min=2)
    process_chunk(cfg, bag_id, chunk_id)

    static = np.load(local_path(static_map_path(bag_id, chunk_id)))
    assert "intensity" in static
    assert static["intensity"].shape[0] == static["xyz"].shape[0], (
        f"intensity ({static['intensity'].shape[0]}) and xyz ({static['xyz'].shape[0]}) "
        "must align"
    )
    # First 4 entries (sweeps 0–1, 2 points each) should be zero-padded.
    assert (static["intensity"][:4] == 0).all()
    # Sweeps 2–4 contributed real intensity values.
    assert (static["intensity"][4:] > 0).all()


def test_empty_proc_index_writes_sentinel(tmp_env):
    """No proc_index rows → write empty static_map.npz, don't crash."""
    bag_id, chunk_id = "bag_empty", "chunk0"
    write_table([], PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    cfg = ComponentConfig()
    result = process_chunk(cfg, bag_id, chunk_id)
    assert result.n_static == 0
    assert result.n_dynamic == 0
    static = np.load(local_path(static_map_path(bag_id, chunk_id)))
    assert static["xyz"].shape == (0, 3)


def test_static_map_written(tmp_env):
    bag_id, chunk_id = "bag2", "chunk0"
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    for i in range(5):
        _write_world_sweep(bag_id, chunk_id, i, xyz)
    _write_proc_index(bag_id, chunk_id, list(range(5)), xyz_per_sweep=[xyz] * 5)

    cfg = ComponentConfig(static_sweep_fraction=0.1, static_sweep_min=1)
    process_chunk(cfg, bag_id, chunk_id)

    assert os.path.exists(local_path(static_map_path(bag_id, chunk_id)))
    data = np.load(local_path(static_map_path(bag_id, chunk_id)))
    assert "xyz" in data
    assert data["xyz"].shape[1] == 3
