"""Tests for the (sweep, camera) matching logic without rosbag/file I/O."""

from __future__ import annotations

import numpy as np

from wato_common.geometry import PoseSample
from wato_ingest.artifacts.frame_index import _build_rows


def _id_quat():
    return np.array([0.0, 0.0, 0.0, 1.0])


def _sweep(seq: int, ts_ns: int) -> dict:
    return {
        "bag_id": "b",
        "chunk_id": "0000",
        "lidar_id": "LIDAR_TOP",
        "sweep_id": seq,
        "lidar_path": f"/data/artifacts/raw/b/chunks/0000/lidar/LIDAR_TOP/{seq:06d}.npz",
        "header_timestamp_ns": ts_ns,
        "record_timestamp_ns": ts_ns,
        "num_points": 100,
        "has_ring": True,
        "has_intensity": True,
        "has_point_time": True,
        "min_range_m": 1.0,
        "max_range_m": 50.0,
        "valid": True,
        "drop_reason": None,
    }


def _cam(cam_id: str, seq: int, ts_ns: int) -> dict:
    return {
        "bag_id": "b",
        "chunk_id": "0000",
        "cam_id": cam_id,
        "camera_seq": seq,
        "image_path": f"/data/artifacts/raw/b/chunks/0000/cam_{cam_id}/{seq:06d}.jpg",
        "header_timestamp_ns": ts_ns,
        "record_timestamp_ns": ts_ns,
        "width": 1920,
        "height": 1080,
        "encoding": "jpeg",
        "is_compressed": True,
        "valid": True,
        "drop_reason": None,
    }


def _identity_pose_at(ts_ns: int) -> PoseSample:
    return PoseSample(ts_ns, np.zeros(3), _id_quat())


def test_one_camera_two_sweeps_within_threshold():
    sweeps = [_sweep(0, 100_000_000), _sweep(1, 200_000_000)]
    cams = [
        _cam("CAM_FRONT", 0, 98_000_000),  # 2 ms before sweep 0 → valid
        _cam("CAM_FRONT", 1, 205_000_000),  # 5 ms after  sweep 1 → valid
    ]
    poses = [_identity_pose_at(0), _identity_pose_at(300_000_000)]
    rows = _build_rows(
        bag_id="b",
        chunk_id="0000",
        sweeps=sweeps,
        camera_frames=cams,
        pose_samples=poses,
        calib_uri="file:///x",
        max_cam_offset_ms=50.0,
        max_pose_gap_ns=10_000_000_000,
    )
    assert len(rows) == 2
    assert all(r.valid_camera for r in rows)
    assert rows[0].camera_seq == 0
    assert rows[1].camera_seq == 1
    assert abs(rows[0].camera_offset_ms - (-2.0)) < 1e-6


def test_camera_drops_when_offset_exceeds_threshold():
    sweeps = [_sweep(0, 100_000_000)]
    cams = [_cam("CAM_FRONT", 0, 200_000_000)]  # 100 ms off
    rows = _build_rows(
        bag_id="b",
        chunk_id="0000",
        sweeps=sweeps,
        camera_frames=cams,
        pose_samples=[_identity_pose_at(0)],
        calib_uri="x",
        max_cam_offset_ms=50.0,
        max_pose_gap_ns=10_000_000_000,
    )
    assert len(rows) == 1
    assert rows[0].valid_camera is False
    assert "exceeds_threshold" in (rows[0].camera_drop_reason or "")


def test_two_cameras_each_get_their_own_row():
    sweeps = [_sweep(0, 100_000_000)]
    cams = [_cam("CAM_FRONT", 0, 99_000_000), _cam("CAM_LEFT", 0, 101_000_000)]
    rows = _build_rows(
        bag_id="b",
        chunk_id="0000",
        sweeps=sweeps,
        camera_frames=cams,
        pose_samples=[_identity_pose_at(0)],
        calib_uri="x",
        max_cam_offset_ms=50.0,
        max_pose_gap_ns=10_000_000_000,
    )
    assert len(rows) == 2
    assert {r.cam_id for r in rows} == {"CAM_FRONT", "CAM_LEFT"}


def test_pose_marked_invalid_when_no_samples_close_enough():
    sweeps = [_sweep(0, 100_000_000)]
    cams = [_cam("CAM_FRONT", 0, 100_000_000)]
    # Only sample is 1 second away; max_pose_gap_ns at 50 ms → invalid_pose.
    rows = _build_rows(
        bag_id="b",
        chunk_id="0000",
        sweeps=sweeps,
        camera_frames=cams,
        pose_samples=[_identity_pose_at(1_100_000_000)],
        calib_uri="x",
        max_cam_offset_ms=50.0,
        max_pose_gap_ns=50_000_000,
    )
    assert rows[0].valid_pose is False
    assert rows[0].world_T_ego_flat is None
