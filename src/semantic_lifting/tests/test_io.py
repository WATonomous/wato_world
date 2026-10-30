"""Tests for the camera frames semantic_lifting pairs with each sweep: ingest's
pairing (frame_index), and the ego pose looked up at each frame's own time."""

from __future__ import annotations

import numpy as np
import pytest

from wato_common.artifact_store import frame_index_path, poses_path
from wato_common.artifacts import ArtifactFormatError
from wato_common.geometry import PoseSample
from wato_common.io.parquet_io import write_table
from wato_common.pose_lookup import interval_drop_reasons
from wato_common.schemas import FRAME_INDEX_SCHEMA, POSES_SCHEMA, FrameIndexRow
from wato_semantic_lifting.io import load_sweep_frames

MS = 1_000_000
BAG, CHUNK = "b", "0000"


@pytest.fixture(autouse=True)
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")


def _write_poses(times_ms: list[int]) -> None:
    """Ego moving +x at 10 m/s; stretches marked with the ingest defaults."""
    samples = [
        PoseSample(
            t * MS, np.array([t / 100.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0])
        )
        for t in times_ms
    ]
    reasons = interval_drop_reasons(
        samples, max_bracket_ns=250 * MS, max_speed_mps=30.0
    )
    rows = [
        {
            "bag_id": BAG,
            "chunk_id": CHUNK,
            "timestamp_ns": s.timestamp_ns,
            "x": float(s.translation[0]),
            "y": 0.0,
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
            "world_T_ego_flat": np.eye(4).flatten().tolist(),
            "source": "/odom",
            "valid": True,
            "interval_drop_reason": r,
        }
        for s, r in zip(samples, reasons)
    ]
    write_table(rows, POSES_SCHEMA, poses_path(BAG, CHUNK))


def _fi_row(
    sweep_id: int,
    sweep_ms: int,
    cam_seq: int | None,
    cam_ms: int | None,
    valid_cam=True,
    cam_id="cam_front",
) -> dict:
    return FrameIndexRow(
        frame_id=f"{sweep_id}_{cam_seq}",
        bag_id=BAG,
        chunk_id=CHUNK,
        sweep_id=sweep_id,
        lidar_id="lidar_cc",
        lidar_path="",
        reference_timestamp_ns=sweep_ms * MS,
        cam_id=cam_id,
        camera_seq=cam_seq,
        camera_timestamp_ns=None if cam_ms is None else cam_ms * MS,
        valid_camera=valid_cam,
        # The sweep's pose: deliberately wrong for the camera, to prove it's unused.
        world_T_ego_flat=np.eye(4).flatten().tolist(),
        valid_pose=True,
    ).model_dump()


def _frames(rows: list[dict]):
    write_table(rows, FRAME_INDEX_SCHEMA, frame_index_path(BAG, CHUNK))
    return load_sweep_frames(BAG, CHUNK)


def test_frame_pose_is_looked_up_at_camera_time():
    _write_poses([0, 100, 200, 300])
    (ref,) = _frames([_fi_row(0, 100, 7, 140)])[0]
    assert ref.timestamp_ns == 140 * MS
    np.testing.assert_allclose(ref.world_T_ego[:3, 3], [1.4, 0.0, 0.0])


def test_each_sweep_gets_the_frames_ingest_paired_with_it():
    _write_poses([0, 100, 200, 300])
    rows = [
        _fi_row(0, 100, 7, 140),
        _fi_row(0, 100, 3, 95, cam_id="cam_back"),
        _fi_row(1, 150, 7, 140),
        _fi_row(2, 200, 8, 225),
    ]
    frames = _frames(rows)
    assert {s: [(f.cam_id, f.camera_seq) for f in fs] for s, fs in frames.items()} == {
        0: [("cam_front", 7), ("cam_back", 3)],
        1: [("cam_front", 7)],
        2: [("cam_front", 8)],
    }
    assert frames[0][0] is frames[1][0]  # one frame, shared by both sweeps


def test_frames_in_untrusted_pose_stretches_and_dropped_cameras_are_skipped():
    _write_poses([0, 100, 1_100])  # 1 s dropout after 100 ms
    rows = [
        _fi_row(0, 50, 1, 60),
        _fi_row(1, 90, 2, 500),
        _fi_row(2, 50, None, None, valid_cam=False),
    ]
    frames = _frames(rows)
    assert [f.camera_seq for f in frames[0]] == [1]
    assert 1 not in frames and 2 not in frames


def test_a_valid_camera_row_without_a_frame_is_rejected():
    _write_poses([0, 100, 200])
    with pytest.raises(ArtifactFormatError, match="without camera_seq"):
        _frames([_fi_row(0, 50, None, None)])
