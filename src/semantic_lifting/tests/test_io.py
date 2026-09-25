"""Tests for io.load_sweeps / io.load_frame_refs: real timestamps, and the
ego pose looked up at each camera frame's own time."""

from __future__ import annotations

import numpy as np
import pytest

from wato_common.artifact_store import (
    frame_index_path,
    lidar_proc_index_path,
    poses_path,
)
from wato_common.geometry import PoseSample
from wato_common.io.parquet_io import write_table
from wato_common.pose_lookup import interval_drop_reasons
from wato_common.schemas import (
    FRAME_INDEX_SCHEMA,
    POSES_SCHEMA,
    PROCESSED_SWEEPS_SCHEMA,
    FrameIndexRow,
    ProcessedSweepMeta,
)
from wato_semantic_lifting.io import load_frame_refs, load_sweeps
from wato_semantic_lifting.temporal_match import match_sweep_to_frames

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
    sweep_id: int, sweep_ms: int, cam_seq: int, cam_ms: int, valid_cam=True
) -> dict:
    return FrameIndexRow(
        frame_id=f"{sweep_id}_{cam_seq}",
        bag_id=BAG,
        chunk_id=CHUNK,
        sweep_id=sweep_id,
        lidar_id="lidar_cc",
        lidar_path="",
        reference_timestamp_ns=sweep_ms * MS,
        cam_id="cam_front",
        camera_seq=cam_seq,
        camera_timestamp_ns=cam_ms * MS,
        valid_camera=valid_cam,
        # The sweep's pose: deliberately wrong for the camera, to prove it's unused.
        world_T_ego_flat=np.eye(4).flatten().tolist(),
        valid_pose=True,
    ).model_dump()


def test_frame_pose_is_looked_up_at_camera_time():
    _write_poses([0, 100, 200, 300])
    write_table(
        [_fi_row(0, 100, 7, 140)], FRAME_INDEX_SCHEMA, frame_index_path(BAG, CHUNK)
    )
    (ref,) = load_frame_refs(BAG, CHUNK)
    assert ref.timestamp_ns == 140 * MS
    np.testing.assert_allclose(ref.world_T_ego[:3, 3], [1.4, 0.0, 0.0])


def test_each_camera_frame_is_returned_once():
    _write_poses([0, 100, 200, 300])
    rows = [_fi_row(0, 100, 7, 140), _fi_row(1, 150, 7, 140), _fi_row(2, 200, 8, 225)]
    write_table(rows, FRAME_INDEX_SCHEMA, frame_index_path(BAG, CHUNK))
    assert [r.camera_seq for r in load_frame_refs(BAG, CHUNK)] == [7, 8]


def test_frames_in_untrusted_pose_stretches_and_invalid_cameras_are_skipped():
    _write_poses([0, 100, 1_100])  # 1 s dropout after 100 ms
    rows = [
        _fi_row(0, 50, 1, 60),
        _fi_row(1, 90, 2, 500),
        _fi_row(2, 50, 3, 70, valid_cam=False),
    ]
    write_table(rows, FRAME_INDEX_SCHEMA, frame_index_path(BAG, CHUNK))
    assert [r.camera_seq for r in load_frame_refs(BAG, CHUNK)] == [1]


def test_sweeps_carry_their_timestamp_and_match_the_nearest_frame():
    _write_poses([0, 100, 200, 300])
    meta = [
        ProcessedSweepMeta(
            bag_id=BAG,
            chunk_id=CHUNK,
            sweep_id=i,
            lidar_id="lidar_cc",
            reference_timestamp_ns=t * MS,
            n_points_total=1,
            n_points_static=1,
            n_points_dynamic=0,
            world_path="w",
            dynamic_mask_path="d",
            has_intensity=False,
            deskewed=True,
        ).model_dump()
        for i, t in enumerate((50, 150))
    ]
    write_table(meta, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(BAG, CHUNK))
    rows = [_fi_row(0, 50, 1, 45), _fi_row(1, 150, 2, 160)]
    write_table(rows, FRAME_INDEX_SCHEMA, frame_index_path(BAG, CHUNK))

    sweeps = load_sweeps(BAG, CHUNK)
    assert [s.timestamp_ns for s in sweeps] == [50 * MS, 150 * MS]
    refs = load_frame_refs(BAG, CHUNK)
    # Each sweep matches the frame taken during it, not the chunk's first frame.
    assert [
        match_sweep_to_frames(s.timestamp_ns, refs)["cam_front"].camera_seq
        for s in sweeps
    ] == [1, 2]
