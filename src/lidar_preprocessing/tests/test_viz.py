"""Tests for composite sweep lookup used by the visualization CLI."""

from __future__ import annotations

import os

import pytest

from wato_common.artifact_store import lidar_proc_index_path
from wato_common.io.parquet_io import write_table
from wato_common.schemas import PROCESSED_SWEEPS_SCHEMA
from wato_lidar_preprocessing.viz import resolve_sweep_lidar_id


@pytest.fixture()
def tmp_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", str(tmp_path))
    return tmp_path


def _write_index(bag_id: str, chunk_id: str, identities: list[tuple[str, int]]):
    rows = [
        {
            "bag_id": bag_id,
            "chunk_id": chunk_id,
            "lidar_id": lidar_id,
            "sweep_id": sweep_id,
            "reference_timestamp_ns": sweep_id,
            "n_points_total": 0,
            "n_points_static": 0,
            "n_points_dynamic": 0,
            "n_points_ground_candidate": 0,
            "n_points_ground": 0,
            "world_path": "unused",
            "has_intensity": False,
            "deskewed": True,
            "valid": True,
        }
        for lidar_id, sweep_id in identities
    ]
    path = lidar_proc_index_path(bag_id, chunk_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, path)


def test_infers_lidar_only_when_sweep_has_one_match(tmp_env):
    _write_index("bag", "chunk", [("LEFT", 4), ("RIGHT", 5)])

    assert resolve_sweep_lidar_id("bag", "chunk", 4, None) == "LEFT"


def test_ambiguous_sweep_requires_lidar_id(tmp_env):
    _write_index("bag", "chunk", [("LEFT", 4), ("RIGHT", 4)])

    with pytest.raises(ValueError, match="--lidar-id"):
        resolve_sweep_lidar_id("bag", "chunk", 4, None)
    assert resolve_sweep_lidar_id("bag", "chunk", 4, "RIGHT") == "RIGHT"


def test_rejects_lidar_that_does_not_own_requested_sweep(tmp_env):
    _write_index("bag", "chunk", [("LEFT", 4)])

    with pytest.raises(ValueError, match="not found"):
        resolve_sweep_lidar_id("bag", "chunk", 4, "RIGHT")
