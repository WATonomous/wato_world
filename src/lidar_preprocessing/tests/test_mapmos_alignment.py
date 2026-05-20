"""Alignment + edge-case tests for the MapMOS sidecar contract.

Plan non-negotiables exercised: #2 length invariant, #6 missing -> graceful
fallback, #7 valid=False sweeps, #20 empty != None distinction.

The "ground-strip + full-length reconstruction" assertion (non-negotiable
#3) is exercised at the orchestration level once Step 3 lands real
inference. The stub returns full-length zeros directly so there's
nothing to reconstruct; tests for the reconstruction path are deferred
to the Step 3 PR with a fixture-based mock of `run_sweep_inference`.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from wato_common.artifact_store import (
    lidar_proc_index_path,
    lidar_world_path,
    local_path,
    mapmos_logit_path,
)
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import PROCESSED_SWEEPS_SCHEMA
from wato_lidar_preprocessing.config import ComponentConfig, MapMOSParams
from wato_lidar_preprocessing.mapmos.io import read_logits, write_logits
from wato_lidar_preprocessing.mapmos.pipeline import process_chunk as mapmos_process_chunk


@pytest.fixture()
def tmp_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", str(tmp_path))
    return tmp_path


def _write_world_sweep(bag_id, chunk_id, sweep_id, xyz, ground_mask=None):
    path = local_path(lidar_world_path(bag_id, chunk_id, sweep_id))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    kwargs = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    if ground_mask is not None:
        kwargs["ground_mask"] = ground_mask
    np.savez_compressed(path, **kwargs)


def _row(bag_id, chunk_id, sweep_id, xyz, valid=True, parquet_n=None):
    """Build a meta row. `parquet_n` overrides the n_points_total column."""
    n = xyz.shape[0]
    return {
        "bag_id": bag_id,
        "chunk_id": chunk_id,
        "sweep_id": sweep_id,
        "lidar_id": "LIDAR_TOP",
        "reference_timestamp_ns": sweep_id * 100_000_000,
        "n_points_total": parquet_n if parquet_n is not None else n,
        "n_points_static": 0,
        "n_points_dynamic": 0,
        "n_points_ground": 0,
        "world_path": lidar_world_path(bag_id, chunk_id, sweep_id),
        "dynamic_mask_path": "",
        "mapmos_logit_path": None,
        "has_intensity": False,
        "deskewed": True,
        "valid": valid,
        "drop_reason": None,
        "world_xmin": float(xyz[:, 0].min()) if n else None,
        "world_xmax": float(xyz[:, 0].max()) if n else None,
        "world_ymin": float(xyz[:, 1].min()) if n else None,
        "world_ymax": float(xyz[:, 1].max()) if n else None,
        "world_zmin": float(xyz[:, 2].min()) if n else None,
        "world_zmax": float(xyz[:, 2].max()) if n else None,
        "frame_id": None,
    }


def _cfg_enabled() -> ComponentConfig:
    return ComponentConfig(mapmos=MapMOSParams(enabled=True))


# ---------------------------------------------------------------------------
# read_logits semantics
# ---------------------------------------------------------------------------


def test_read_logits_missing_returns_none(tmp_env):
    """Plan non-negotiable #20: missing file -> None (NOT empty array)."""
    uri = mapmos_logit_path("bag0", "c0", 0)
    assert read_logits(uri) is None


def test_read_logits_empty_vs_missing(tmp_env):
    """A zero-length array is distinct from None.

    Branch on `is None`, NOT on `len(...) == 0` (plan non-negotiable #20).
    """
    bag_id, chunk_id = "bag0", "c0"
    uri = mapmos_logit_path(bag_id, chunk_id, 0)
    os.makedirs(os.path.dirname(local_path(uri)), exist_ok=True)
    write_logits(uri, np.empty(0, dtype=np.float32))
    out = read_logits(uri)
    assert out is not None
    assert out.shape == (0,)
    assert out.dtype == np.float32


def test_write_logits_rejects_wrong_dtype(tmp_env):
    """Defensive dtype check at the write boundary."""
    uri = mapmos_logit_path("bag0", "c0", 0)
    os.makedirs(os.path.dirname(local_path(uri)), exist_ok=True)
    with pytest.raises(ValueError):
        write_logits(uri, np.zeros(5, dtype=np.float64))


# ---------------------------------------------------------------------------
# Stub-pipeline alignment + valid=False handling
# ---------------------------------------------------------------------------


def test_stub_writes_length_aligned_sidecar(tmp_env):
    bag_id, chunk_id = "bag0", "c0"
    n_points = 7
    xyz = np.zeros((n_points, 3), dtype=np.float64)
    _write_world_sweep(bag_id, chunk_id, 0, xyz)
    rows = [_row(bag_id, chunk_id, 0, xyz)]
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    res = mapmos_process_chunk(_cfg_enabled(), bag_id, chunk_id, prev_chunk_id=None)
    assert res.n_sweeps_processed == 1

    out = np.load(local_path(mapmos_logit_path(bag_id, chunk_id, 0)))
    assert out.shape == (n_points,)
    assert out.dtype == np.float32
    assert np.all(out == 0.0)


def test_invalid_sweep_writes_no_sidecar(tmp_env):
    """One valid=False sweep in a chunk -> no sidecar for it; chunk still completes."""
    bag_id, chunk_id = "bag0", "c0"
    xyz_a = np.zeros((3, 3), dtype=np.float64)
    xyz_b = np.zeros((3, 3), dtype=np.float64)
    _write_world_sweep(bag_id, chunk_id, 0, xyz_a)
    _write_world_sweep(bag_id, chunk_id, 1, xyz_b)
    rows = [
        _row(bag_id, chunk_id, 0, xyz_a, valid=True),
        _row(bag_id, chunk_id, 1, xyz_b, valid=False),
    ]
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    res = mapmos_process_chunk(_cfg_enabled(), bag_id, chunk_id, prev_chunk_id=None)
    assert res.n_sweeps_processed == 1
    assert res.n_sweeps_skipped == 1

    # Valid sweep has a sidecar; invalid sweep does NOT.
    assert os.path.exists(local_path(mapmos_logit_path(bag_id, chunk_id, 0)))
    assert not os.path.exists(local_path(mapmos_logit_path(bag_id, chunk_id, 1)))


def test_zero_point_sweep_writes_empty_sidecar(tmp_env):
    """Legitimate zero-point sweep -> empty length-0 sidecar (NOT None).

    classify will see an empty-but-present sidecar and treat it as
    length-aligned (no fallback warning).
    """
    bag_id, chunk_id = "bag0", "c0"
    xyz = np.zeros((0, 3), dtype=np.float64)
    _write_world_sweep(bag_id, chunk_id, 0, xyz)
    rows = [_row(bag_id, chunk_id, 0, xyz)]
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    res = mapmos_process_chunk(_cfg_enabled(), bag_id, chunk_id, prev_chunk_id=None)
    assert res.n_sweeps_processed == 1

    out = read_logits(mapmos_logit_path(bag_id, chunk_id, 0))
    assert out is not None
    assert out.shape == (0,)


def test_disabled_is_noop(tmp_env):
    """cfg.mapmos.enabled=False -> no sidecars written, no parquet rewrite."""
    bag_id, chunk_id = "bag0", "c0"
    xyz = np.zeros((3, 3), dtype=np.float64)
    _write_world_sweep(bag_id, chunk_id, 0, xyz)
    rows = [_row(bag_id, chunk_id, 0, xyz)]
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    cfg = ComponentConfig(mapmos=MapMOSParams(enabled=False))
    res = mapmos_process_chunk(cfg, bag_id, chunk_id, prev_chunk_id=None)
    assert res.n_sweeps_processed == 0
    assert not os.path.exists(local_path(mapmos_logit_path(bag_id, chunk_id, 0)))


def test_length_invariant_uses_xyz_not_parquet(tmp_env):
    """Length invariant must compare against xyz NPZ, not n_points_total column.

    Plan non-negotiable #14: deskew may filter points after writing the
    parquet row, so the NPZ is the source of truth. We mock a row where
    parquet says 100 but the NPZ has 3 points -> stub writes length 3
    (matching NPZ) and the pipeline succeeds.
    """
    bag_id, chunk_id = "bag0", "c0"
    xyz = np.zeros((3, 3), dtype=np.float64)
    _write_world_sweep(bag_id, chunk_id, 0, xyz)
    rows = [_row(bag_id, chunk_id, 0, xyz, parquet_n=100)]  # lying parquet
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    res = mapmos_process_chunk(_cfg_enabled(), bag_id, chunk_id, prev_chunk_id=None)
    assert res.n_sweeps_processed == 1
    out = read_logits(mapmos_logit_path(bag_id, chunk_id, 0))
    assert out is not None
    # Length matches xyz (3), NOT parquet's 100.
    assert out.shape == (3,)


def test_pipeline_populates_mapmos_logit_path_column(tmp_env):
    """After mapmos.process_chunk, lidar_proc_index parquet has the column set."""
    bag_id, chunk_id = "bag0", "c0"
    xyz = np.zeros((3, 3), dtype=np.float64)
    _write_world_sweep(bag_id, chunk_id, 0, xyz)
    rows = [_row(bag_id, chunk_id, 0, xyz)]
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    mapmos_process_chunk(_cfg_enabled(), bag_id, chunk_id, prev_chunk_id=None)

    out = read_rows(lidar_proc_index_path(bag_id, chunk_id))
    assert out[0]["mapmos_logit_path"] == mapmos_logit_path(bag_id, chunk_id, 0)
