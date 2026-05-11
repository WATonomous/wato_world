"""Tests for pipeline.py — orchestration, idempotency, isolation."""

from __future__ import annotations

import json
import os
from unittest import mock

import numpy as np
import pytest

from wato_common.artifact_store import (
    calibration_path,
    chunks_index_path,
    ensure_local_dir,
    ground_path,
    lidar_proc_dir,
    lidar_sweep_path,
    lidar_sweeps_path,
    local_path,
    poses_path,
)
from wato_common.io.parquet_io import write_table
from wato_common.schemas import (
    CHUNK_SCHEMA,
    LIDAR_SWEEPS_SCHEMA,
    POSES_SCHEMA,
)
from wato_lidar_preprocessing import pipeline
from wato_lidar_preprocessing.config import ComponentConfig


@pytest.fixture()
def tmp_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", str(tmp_path))
    return tmp_path


def _bootstrap_chunk(bag_id: str, chunk_id: str, sweep_id: int = 0):
    """Write the minimum on-disk state to make one chunk processable."""
    # calibration.json
    calib = {
        "calibration_version": "t", "ego_frame": "base_link",
        "cameras": {}, "lidars": {"LIDAR_TOP": {
            "frame_id": "v", "ego_T_lidar": np.eye(4).tolist(),
        }}, "static_transforms": {}, "checks": {"sanity": "ok", "notes": ""},
    }
    p = local_path(calibration_path(bag_id))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as fh:
        json.dump(calib, fh)

    pose_row = {
        "bag_id": bag_id, "chunk_id": chunk_id, "timestamp_ns": 0,
        "x": 0.0, "y": 0.0, "z": 0.0,
        "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
        "world_T_ego_flat": np.eye(4).flatten().tolist(),
        "source": "odom", "valid": True,
    }
    write_table([pose_row], POSES_SCHEMA, poses_path(bag_id, chunk_id))

    sw_path = local_path(lidar_sweep_path(bag_id, chunk_id, "LIDAR_TOP", sweep_id))
    os.makedirs(os.path.dirname(sw_path), exist_ok=True)
    np.savez_compressed(
        sw_path,
        x=np.array([1.0, 2.0], dtype=np.float32),
        y=np.zeros(2, dtype=np.float32),
        z=np.zeros(2, dtype=np.float32),
    )

    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    write_table([{
        "bag_id": bag_id, "chunk_id": chunk_id,
        "lidar_id": "LIDAR_TOP", "sweep_id": sweep_id,
        "lidar_path": lidar_sweep_path(bag_id, chunk_id, "LIDAR_TOP", sweep_id),
        "header_timestamp_ns": 0, "record_timestamp_ns": 0,
        "num_points": 2, "has_ring": False, "has_intensity": False,
        "has_point_time": False, "min_range_m": 1.0, "max_range_m": 2.0,
        "valid": True, "drop_reason": None,
    }], LIDAR_SWEEPS_SCHEMA, lidar_sweeps_path(bag_id, chunk_id))


def _write_chunks_index(bag_id: str, chunk_ids: list[str]):
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


def test_validates_chunks_index(tmp_env):
    """No chunks index for the bag → FileNotFoundError with a helpful message."""
    cfg = ComponentConfig()
    with pytest.raises(FileNotFoundError, match="run `watod run ingest"):
        pipeline.run(cfg, bag_id="never_ingested")


def test_skips_completed_chunks(tmp_env):
    """If ground.npz exists, the chunk is skipped and deskew not invoked."""
    bag_id = "bag_skip"
    _write_chunks_index(bag_id, ["chunk0"])
    # Pretend chunk0 was already processed.
    out = local_path(ground_path(bag_id, "chunk0"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, ground_xyz=np.empty((0, 3)))

    cfg = ComponentConfig()
    with mock.patch.object(pipeline.deskew, "process_chunk") as mock_deskew:
        pipeline.run(cfg, bag_id=bag_id)
    mock_deskew.assert_not_called()


def test_force_flag_overrides_skip(tmp_env):
    """force=True must re-run already-processed chunks."""
    bag_id = "bag_force"
    _write_chunks_index(bag_id, ["chunk0"])
    out = local_path(ground_path(bag_id, "chunk0"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, ground_xyz=np.empty((0, 3)))

    _bootstrap_chunk(bag_id, "chunk0")

    cfg = ComponentConfig()
    pipeline.run(cfg, bag_id=bag_id, force=True)
    # ground.npz now reflects the actual run, not the placeholder.
    data = np.load(out)
    assert "height_grid" in data


def test_isolates_chunk_failures(tmp_env):
    """One chunk failing must not stop subsequent chunks."""
    bag_id = "bag_iso"
    _write_chunks_index(bag_id, ["chunk_bad", "chunk_good"])
    # Bootstrap only chunk_good — chunk_bad has no calibration/poses, so deskew raises.
    _bootstrap_chunk(bag_id, "chunk_good")

    cfg = ComponentConfig()
    pipeline.run(cfg, bag_id=bag_id)
    # chunk_good produced ground.npz; chunk_bad did not.
    assert os.path.exists(local_path(ground_path(bag_id, "chunk_good")))
    assert not os.path.exists(local_path(ground_path(bag_id, "chunk_bad")))


def test_all_failures_raise(tmp_env):
    """If every chunk fails the pipeline must raise RuntimeError."""
    bag_id = "bag_all_fail"
    _write_chunks_index(bag_id, ["chunk_bad"])
    # No calibration / poses written → deskew will fail.

    cfg = ComponentConfig()
    with pytest.raises(RuntimeError, match="all 1 chunks failed"):
        pipeline.run(cfg, bag_id=bag_id)


def test_parallel_workers(tmp_env):
    """workers > 1 dispatches via ProcessPoolExecutor; all chunks complete."""
    bag_id = "bag_par"
    chunk_ids = [f"chunk{i}" for i in range(3)]
    _write_chunks_index(bag_id, chunk_ids)
    for cid in chunk_ids:
        _bootstrap_chunk(bag_id, cid)

    cfg = ComponentConfig()
    pipeline.run(cfg, bag_id=bag_id, workers=2)
    for cid in chunk_ids:
        assert os.path.exists(local_path(ground_path(bag_id, cid))), (
            f"chunk {cid} did not produce ground.npz"
        )


def test_chunk_id_filter(tmp_env):
    """chunk_id=X processes only that chunk."""
    bag_id = "bag_filter"
    _write_chunks_index(bag_id, ["chunk0", "chunk1"])
    _bootstrap_chunk(bag_id, "chunk0")
    _bootstrap_chunk(bag_id, "chunk1")

    cfg = ComponentConfig()
    pipeline.run(cfg, bag_id=bag_id, chunk_id="chunk0")
    assert os.path.exists(local_path(ground_path(bag_id, "chunk0")))
    assert not os.path.exists(local_path(ground_path(bag_id, "chunk1")))


def test_chunk_summary_written(tmp_env):
    """Successful chunk run produces lidar_proc_summary.parquet with expected fields."""
    from wato_common.artifact_store import lidar_proc_summary_path
    from wato_common.io.parquet_io import read_rows

    bag_id = "bag_summary"
    _write_chunks_index(bag_id, ["chunk0"])
    _bootstrap_chunk(bag_id, "chunk0")

    cfg = ComponentConfig()
    pipeline.run(cfg, bag_id=bag_id)

    summary_uri = lidar_proc_summary_path(bag_id, "chunk0")
    assert os.path.exists(local_path(summary_uri))
    rows = read_rows(summary_uri)
    assert len(rows) == 1
    row = rows[0]
    assert row["bag_id"] == bag_id
    assert row["chunk_id"] == "chunk0"
    assert row["n_sweeps_total"] == 1
    assert row["n_sweeps_valid"] == 1
    assert row["n_sweeps_invalid"] == 0
    # Bootstrap sweep has 2 points — both classified static (single sweep
    # below threshold; everything is dynamic by majority vote, but with
    # static_sweep_min=5 and one sweep neither static nor dynamic threshold
    # is reached and we just record what's there).
    assert row["n_points_total"] == 2
    # cache_auto_disabled is False for tiny bootstrap chunks.
    assert row["cache_auto_disabled"] is False
    # No Patchwork++ in tests → ground status flags it.
    assert row["ground_status"] in ("ok", "skipped_no_ground_mask", "empty")


def test_cache_auto_disable_logged_in_summary(tmp_env, monkeypatch):
    """Setting WATO_LIDAR_CACHE_BYTES low forces cache_auto_disabled=True."""
    from wato_common.artifact_store import lidar_proc_summary_path
    from wato_common.io.parquet_io import read_rows

    bag_id = "bag_cache_disabled"
    _write_chunks_index(bag_id, ["chunk0"])
    _bootstrap_chunk(bag_id, "chunk0")

    # Force the auto-disable path: 1-byte budget < estimated cache size.
    monkeypatch.setenv("WATO_LIDAR_CACHE_BYTES", "1")

    cfg = ComponentConfig()
    pipeline.run(cfg, bag_id=bag_id)

    rows = read_rows(lidar_proc_summary_path(bag_id, "chunk0"))
    assert rows[0]["cache_auto_disabled"] is True
    assert rows[0]["estimated_cache_bytes"] > 0
