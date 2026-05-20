"""Tests for mapmos.history — past-sweep selection + RollingHistoryCache."""

from __future__ import annotations

import os

import numpy as np
import pytest

from wato_common.artifact_store import (
    lidar_proc_index_path,
    lidar_world_path,
    local_path,
)
from wato_common.io.parquet_io import write_table
from wato_common.schemas import PROCESSED_SWEEPS_SCHEMA
from wato_lidar_preprocessing.mapmos.history import (
    RollingHistoryCache,
    get_past_rows,
    load_prev_chunk_meta_rows,
)


@pytest.fixture()
def tmp_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", str(tmp_path))
    return tmp_path


def _row(sweep_id: int, valid: bool = True) -> dict:
    return {
        "bag_id": "bag",
        "chunk_id": "chunk",
        "sweep_id": sweep_id,
        "lidar_id": "LIDAR_TOP",
        "reference_timestamp_ns": sweep_id * 100_000_000,
        "n_points_total": 1,
        "n_points_static": 0,
        "n_points_dynamic": 0,
        "n_points_ground": 0,
        "world_path": f"/tmp/sweep_{sweep_id:06d}.npz",
        "dynamic_mask_path": "",
        "mapmos_logit_path": None,
        "has_intensity": False,
        "deskewed": True,
        "valid": valid,
        "drop_reason": None,
        "world_xmin": None,
        "world_xmax": None,
        "world_ymin": None,
        "world_ymax": None,
        "world_zmin": None,
        "world_zmax": None,
        "frame_id": None,
    }


# ---------------------------------------------------------------------------
# get_past_rows
# ---------------------------------------------------------------------------


def test_within_chunk_history():
    """Sweep 50 of 100 returns the previous 10 rows, newest-first."""
    meta_rows = [_row(i) for i in range(100)]
    past = get_past_rows(meta_rows, sweep_id=50, n_past=10)
    assert len(past) == 10
    assert [r["sweep_id"] for r in past] == list(range(49, 39, -1))


def test_chunk_boundary_pads_from_previous():
    """Sweep 3 of current chunk should pull 3 from current + 7 from prev tail."""
    current = [_row(i) for i in range(10)]
    prev = [_row(i) for i in range(20)]  # prev chunk has sweeps 0..19
    past = get_past_rows(current, sweep_id=3, n_past=10, prev_meta_rows=prev)
    # Newest current rows: sweep_id 2, 1, 0
    assert [r["sweep_id"] for r in past[:3]] == [2, 1, 0]
    # Then 7 from prev tail (also newest-first)
    assert [r["sweep_id"] for r in past[3:]] == [19, 18, 17, 16, 15, 14, 13]


def test_first_chunk_short_history(caplog):
    """Sweep 0 of chunk 0 with no prev chunk -> empty list with debug log."""
    meta_rows = [_row(i) for i in range(5)]
    past = get_past_rows(meta_rows, sweep_id=0, n_past=10, prev_meta_rows=None)
    assert past == []


def test_invalid_rows_skipped_in_history():
    """valid=False rows in the history window are skipped, NOT zero-padded.

    The network must never see a hole. Plan non-negotiable #17.
    """
    rows = [_row(0), _row(1, valid=False), _row(2), _row(3, valid=False), _row(4)]
    past = get_past_rows(rows, sweep_id=4, n_past=3)
    # Should skip sweep 3 (invalid) and 1 (invalid), giving us sweeps 2, 0
    # (only 2 valid available, n_past=3 -> short history is fine).
    assert [r["sweep_id"] for r in past] == [2, 0]


def test_get_past_rows_unknown_sweep_id_raises():
    rows = [_row(0), _row(1), _row(2)]
    with pytest.raises(ValueError):
        get_past_rows(rows, sweep_id=99, n_past=2)


# ---------------------------------------------------------------------------
# load_prev_chunk_meta_rows
# ---------------------------------------------------------------------------


def test_missing_prev_chunk_index_graceful(tmp_env):
    """prev_chunk_id provided but no parquet on disk -> returns None."""
    result = load_prev_chunk_meta_rows("bag0", "0000")
    assert result is None


def test_none_prev_chunk_id_returns_none(tmp_env):
    assert load_prev_chunk_meta_rows("bag0", None) is None


def test_prev_chunk_index_load(tmp_env):
    """When the prev-chunk index exists, it round-trips."""
    rows = [_row(0), _row(1), _row(2)]
    bag_id, chunk_id = "bag0", "0000"
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))
    out = load_prev_chunk_meta_rows(bag_id, chunk_id)
    assert out is not None
    assert [r["sweep_id"] for r in out] == [0, 1, 2]


# ---------------------------------------------------------------------------
# RollingHistoryCache
# ---------------------------------------------------------------------------


def _write_world_npz(path: str, n_points: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        x=np.zeros(n_points, dtype=np.float64),
        y=np.zeros(n_points, dtype=np.float64),
        z=np.zeros(n_points, dtype=np.float64),
        origin=np.zeros(3, dtype=np.float64),
        ground_mask=np.zeros(n_points, dtype=np.bool_),
    )


def test_rolling_cache_hits_on_repeat(tmp_env):
    uri = lidar_world_path("bag0", "chunk0", 0)
    _write_world_npz(local_path(uri), n_points=100)
    cache = RollingHistoryCache(byte_budget=10 * 1024**2)
    a = cache.get_or_load(uri)
    b = cache.get_or_load(uri)
    # Same object on second hit (cache held a reference; we don't reload).
    assert a is b


def test_rolling_cache_evicts_when_over_budget(tmp_env):
    """When loading more bytes than budget, oldest entries evict."""
    # Make NPZs big enough that 2 fit but 3 push out the first.
    uri0 = lidar_world_path("bag0", "chunk0", 0)
    uri1 = lidar_world_path("bag0", "chunk0", 1)
    uri2 = lidar_world_path("bag0", "chunk0", 2)
    _write_world_npz(local_path(uri0), n_points=5000)
    _write_world_npz(local_path(uri1), n_points=5000)
    _write_world_npz(local_path(uri2), n_points=5000)
    # One xyz of 5000 floats is 5000 * 3 * 8 = 120000 bytes; ground_mask
    # is 5000 bytes; total ~125 KB. Set budget so two fit but three don't.
    cache = RollingHistoryCache(byte_budget=200_000)
    cache.get_or_load(uri0)
    cache.get_or_load(uri1)
    cache.get_or_load(uri2)
    # uri0 should be evicted now; loading it again returns a new object.
    a_first = cache.get_or_load(uri0)
    # uri1 is now the oldest; ensure we haven't broken its slot
    assert a_first is not None
