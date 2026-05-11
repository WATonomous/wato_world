"""Tests for deskew.py (Step A — motion compensation + world projection)."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from wato_common.artifact_store import (
    calibration_path,
    ensure_local_dir,
    lidar_proc_dir,
    lidar_sweeps_path,
    lidar_world_path,
    local_path,
    poses_path,
)
from wato_common.io.parquet_io import write_table
from wato_common.schemas import LIDAR_SWEEPS_SCHEMA, POSES_SCHEMA
from wato_lidar_preprocessing.config import ComponentConfig
from wato_lidar_preprocessing.deskew import process_chunk


@pytest.fixture()
def tmp_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", str(tmp_path))
    return tmp_path


def _write_calibration(bag_id: str, tx: float = 0.0):
    """Write a calibration.json with a known ego_T_lidar (pure translation)."""
    calib = {
        "calibration_version": "test",
        "ego_frame": "base_link",
        "cameras": {},
        "lidars": {
            "LIDAR_TOP": {
                "frame_id": "velodyne",
                "ego_T_lidar": np.eye(4).tolist(),  # identity — lidar == ego
            }
        },
        "static_transforms": {},
        "checks": {"sanity": "ok", "notes": ""},
    }
    path = local_path(calibration_path(bag_id))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(calib, fh)


def _write_poses(bag_id: str, chunk_id: str, poses: list[dict]):
    write_table(poses, POSES_SCHEMA, poses_path(bag_id, chunk_id))


def _write_sweep_index(bag_id: str, chunk_id: str, rows: list[dict]):
    write_table(rows, LIDAR_SWEEPS_SCHEMA, lidar_sweeps_path(bag_id, chunk_id))


def _write_raw_sweep(bag_id: str, chunk_id: str, sweep_id: int, **arrays):
    from wato_common.artifact_store import lidar_sweep_path
    path = local_path(lidar_sweep_path(bag_id, chunk_id, "LIDAR_TOP", sweep_id))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)


def _flat(T):
    return T.flatten().tolist()


def test_static_transform_no_motion(tmp_env):
    """All poses identical → world-frame points should equal sensor-frame points."""
    bag_id, chunk_id = "bag0", "chunk0"
    _write_calibration(bag_id)

    T_identity = np.eye(4)
    pose_row = {
        "bag_id": bag_id,
        "chunk_id": chunk_id,
        "timestamp_ns": 0,
        "x": 0.0, "y": 0.0, "z": 0.0,
        "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
        "world_T_ego_flat": _flat(T_identity),
        "source": "odom",
        "valid": True,
    }
    _write_poses(bag_id, chunk_id, [pose_row])

    # Sweep: 3 points in sensor frame at z=1.
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.zeros(3, dtype=np.float32)
    z = np.ones(3, dtype=np.float32)
    _write_raw_sweep(bag_id, chunk_id, 0, x=x, y=y, z=z)

    from wato_common.artifact_store import lidar_sweep_path
    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    _write_sweep_index(bag_id, chunk_id, [{
        "bag_id": bag_id, "chunk_id": chunk_id,
        "lidar_id": "LIDAR_TOP", "sweep_id": 0,
        "lidar_path": lidar_sweep_path(bag_id, chunk_id, "LIDAR_TOP", 0),
        "header_timestamp_ns": 0, "record_timestamp_ns": 0,
        "num_points": 3, "has_ring": False, "has_intensity": False,
        "has_point_time": False, "min_range_m": 1.0, "max_range_m": 3.2,
        "valid": True, "drop_reason": None,
    }])

    cfg = ComponentConfig()
    results = process_chunk(cfg, bag_id, chunk_id)
    assert len(results) == 1

    world_data = np.load(local_path(lidar_world_path(bag_id, chunk_id, 0)))
    np.testing.assert_allclose(world_data["x"], [1.0, 2.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(world_data["y"], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(world_data["z"], [1.0, 1.0, 1.0], atol=1e-6)


def test_translation_applied(tmp_env):
    """Ego translates by (10, 0, 0) → world-frame x should shift by 10."""
    bag_id, chunk_id = "bag1", "chunk0"
    _write_calibration(bag_id)

    T_shifted = np.eye(4)
    T_shifted[0, 3] = 10.0
    pose_row = {
        "bag_id": bag_id, "chunk_id": chunk_id,
        "timestamp_ns": 0,
        "x": 10.0, "y": 0.0, "z": 0.0,
        "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
        "world_T_ego_flat": _flat(T_shifted),
        "source": "odom", "valid": True,
    }
    _write_poses(bag_id, chunk_id, [pose_row])

    x = np.array([1.0, 2.0], dtype=np.float32)
    y = np.zeros(2, dtype=np.float32)
    z = np.zeros(2, dtype=np.float32)
    _write_raw_sweep(bag_id, chunk_id, 0, x=x, y=y, z=z)

    from wato_common.artifact_store import lidar_sweep_path
    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    _write_sweep_index(bag_id, chunk_id, [{
        "bag_id": bag_id, "chunk_id": chunk_id,
        "lidar_id": "LIDAR_TOP", "sweep_id": 0,
        "lidar_path": lidar_sweep_path(bag_id, chunk_id, "LIDAR_TOP", 0),
        "header_timestamp_ns": 0, "record_timestamp_ns": 0,
        "num_points": 2, "has_ring": False, "has_intensity": False,
        "has_point_time": False, "min_range_m": 1.0, "max_range_m": 2.0,
        "valid": True, "drop_reason": None,
    }])

    cfg = ComponentConfig()
    process_chunk(cfg, bag_id, chunk_id)
    world_data = np.load(local_path(lidar_world_path(bag_id, chunk_id, 0)))
    np.testing.assert_allclose(world_data["x"], [11.0, 12.0], atol=1e-6)
    np.testing.assert_allclose(world_data["y"], [0.0, 0.0], atol=1e-6)


def test_drops_nonfinite_points(tmp_env):
    """NaN/Inf in raw xyz must be filtered before deskew; output count reduced."""
    bag_id, chunk_id = "bag_nan", "chunk0"
    _write_calibration(bag_id)
    pose_row = {
        "bag_id": bag_id, "chunk_id": chunk_id, "timestamp_ns": 0,
        "x": 0.0, "y": 0.0, "z": 0.0,
        "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
        "world_T_ego_flat": _flat(np.eye(4)), "source": "odom", "valid": True,
    }
    _write_poses(bag_id, chunk_id, [pose_row])

    # 5 points, 2 of which are non-finite.
    x = np.array([1.0, np.nan, 2.0, np.inf, 3.0], dtype=np.float32)
    y = np.zeros(5, dtype=np.float32)
    z = np.zeros(5, dtype=np.float32)
    _write_raw_sweep(bag_id, chunk_id, 0, x=x, y=y, z=z)

    from wato_common.artifact_store import lidar_sweep_path
    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    _write_sweep_index(bag_id, chunk_id, [{
        "bag_id": bag_id, "chunk_id": chunk_id,
        "lidar_id": "LIDAR_TOP", "sweep_id": 0,
        "lidar_path": lidar_sweep_path(bag_id, chunk_id, "LIDAR_TOP", 0),
        "header_timestamp_ns": 0, "record_timestamp_ns": 0,
        "num_points": 5, "has_ring": False, "has_intensity": False,
        "has_point_time": False, "min_range_m": 1.0, "max_range_m": 3.0,
        "valid": True, "drop_reason": None,
    }])

    cfg = ComponentConfig()
    results = process_chunk(cfg, bag_id, chunk_id)
    assert results[0].n_points == 3  # 2 dropped
    world_data = np.load(local_path(lidar_world_path(bag_id, chunk_id, 0)))
    assert world_data["x"].shape[0] == 3
    np.testing.assert_allclose(world_data["x"], [1.0, 2.0, 3.0], atol=1e-6)


def test_empty_poses_writes_empty_index(tmp_env):
    """Chunk with no valid poses must still write a (zero-row) lidar_proc_index."""
    bag_id, chunk_id = "bag_no_pose", "chunk0"
    _write_calibration(bag_id)
    # Write a row but mark it invalid → no usable poses.
    pose_row = {
        "bag_id": bag_id, "chunk_id": chunk_id, "timestamp_ns": 0,
        "x": 0.0, "y": 0.0, "z": 0.0,
        "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
        "world_T_ego_flat": _flat(np.eye(4)), "source": "odom", "valid": False,
    }
    _write_poses(bag_id, chunk_id, [pose_row])

    cfg = ComponentConfig()
    results = process_chunk(cfg, bag_id, chunk_id)
    assert results == []

    from wato_common.artifact_store import lidar_proc_index_path
    from wato_common.io.parquet_io import read_rows
    rows = read_rows(lidar_proc_index_path(bag_id, chunk_id))
    assert rows == []


def test_deskewed_flag_false_when_no_point_time(tmp_env):
    bag_id, chunk_id = "bag2", "chunk0"
    _write_calibration(bag_id)
    T = np.eye(4)
    pose_row = {
        "bag_id": bag_id, "chunk_id": chunk_id, "timestamp_ns": 0,
        "x": 0.0, "y": 0.0, "z": 0.0,
        "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
        "world_T_ego_flat": _flat(T), "source": "odom", "valid": True,
    }
    _write_poses(bag_id, chunk_id, [pose_row])
    _write_raw_sweep(bag_id, chunk_id, 0,
                     x=np.ones(2, dtype=np.float32),
                     y=np.zeros(2, dtype=np.float32),
                     z=np.zeros(2, dtype=np.float32))
    from wato_common.artifact_store import lidar_sweep_path
    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    _write_sweep_index(bag_id, chunk_id, [{
        "bag_id": bag_id, "chunk_id": chunk_id,
        "lidar_id": "LIDAR_TOP", "sweep_id": 0,
        "lidar_path": lidar_sweep_path(bag_id, chunk_id, "LIDAR_TOP", 0),
        "header_timestamp_ns": 0, "record_timestamp_ns": 0,
        "num_points": 2, "has_ring": False, "has_intensity": False,
        "has_point_time": False, "min_range_m": 1.0, "max_range_m": 1.0,
        "valid": True, "drop_reason": None,
    }])
    cfg = ComponentConfig()
    results = process_chunk(cfg, bag_id, chunk_id)
    assert results[0].deskewed is False


def _rot_x(theta_rad: float) -> np.ndarray:
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(theta_rad: float) -> np.ndarray:
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_z(theta_rad: float) -> np.ndarray:
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


# Each case mounts the lidar at some position+orientation relative to the ego
# frame and asks: where does sensor-frame (1, 0, 0) land in world frame, given
# the ego is at the world origin with identity rotation?  Answer is computed
# manually to keep the test independent of the deskew implementation.
_EXTRINSIC_CASES: list[dict] = [
    {
        "id": "pure_translation_forward_2m",
        "R": np.eye(3),
        "t": np.array([2.0, 0.0, 0.0]),
        "expected_world": np.array([3.0, 0.0, 0.0]),  # (1+2, 0, 0)
    },
    {
        "id": "pure_translation_up_1.8m",
        "R": np.eye(3),
        "t": np.array([0.0, 0.0, 1.8]),
        "expected_world": np.array([1.0, 0.0, 1.8]),
    },
    {
        "id": "rotation_yaw_90",
        "R": _rot_z(np.pi / 2),
        "t": np.zeros(3),
        # Sensor +X rotates to ego +Y.
        "expected_world": np.array([0.0, 1.0, 0.0]),
    },
    {
        "id": "rotation_pitch_minus_90",
        "R": _rot_y(-np.pi / 2),
        "t": np.zeros(3),
        # R_y(-π/2) @ (1,0,0) = (0, 0, 1) — sensor forward becomes ego +Z.
        "expected_world": np.array([0.0, 0.0, 1.0]),
    },
    {
        "id": "yaw_90_plus_translation",
        "R": _rot_z(np.pi / 2),
        "t": np.array([2.0, 0.0, 1.8]),
        # Sensor (1,0,0) → ego (0,1,0); + (2,0,1.8) → (2,1,1.8).
        "expected_world": np.array([2.0, 1.0, 1.8]),
    },
    {
        "id": "realistic_roof_mount",
        # 5° roll about X, mounted 1.5 m forward and 1.8 m up on the roof.
        "R": _rot_x(np.deg2rad(5.0)),
        "t": np.array([1.5, 0.0, 1.8]),
        "expected_world": np.array([1.0, 0.0, 0.0]) + np.array([1.5, 0.0, 1.8]),
    },
]


@pytest.mark.parametrize("case", _EXTRINSIC_CASES, ids=lambda c: c["id"])
def test_non_identity_ego_T_lidar(tmp_env, case):
    """Sensor-frame (1, 0, 0) should land at the manually-computed world point.

    Sweeps the realistic shapes of `ego_T_lidar` (pure translation, pure
    rotation, combined, and a typical roof-mount) so a sign or convention
    error in calibration handling is caught regardless of which dimension
    drifts.
    """
    bag_id = f"bag_ex_{case['id']}"
    chunk_id = "chunk0"

    ego_T_lidar = np.eye(4)
    ego_T_lidar[:3, :3] = case["R"]
    ego_T_lidar[:3, 3] = case["t"]

    calib = {
        "calibration_version": "test", "ego_frame": "base_link", "cameras": {},
        "lidars": {"LIDAR_TOP": {"frame_id": "v", "ego_T_lidar": ego_T_lidar.tolist()}},
        "static_transforms": {}, "checks": {"sanity": "ok", "notes": ""},
    }
    calib_p = local_path(calibration_path(bag_id))
    os.makedirs(os.path.dirname(calib_p), exist_ok=True)
    with open(calib_p, "w") as fh:
        json.dump(calib, fh)

    # Ego at world origin, identity rotation.
    pose_row = {
        "bag_id": bag_id, "chunk_id": chunk_id, "timestamp_ns": 0,
        "x": 0.0, "y": 0.0, "z": 0.0,
        "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
        "world_T_ego_flat": _flat(np.eye(4)),
        "source": "odom", "valid": True,
    }
    _write_poses(bag_id, chunk_id, [pose_row])

    _write_raw_sweep(
        bag_id, chunk_id, 0,
        x=np.array([1.0], dtype=np.float32),
        y=np.zeros(1, dtype=np.float32),
        z=np.zeros(1, dtype=np.float32),
    )

    from wato_common.artifact_store import lidar_sweep_path
    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    _write_sweep_index(bag_id, chunk_id, [{
        "bag_id": bag_id, "chunk_id": chunk_id,
        "lidar_id": "LIDAR_TOP", "sweep_id": 0,
        "lidar_path": lidar_sweep_path(bag_id, chunk_id, "LIDAR_TOP", 0),
        "header_timestamp_ns": 0, "record_timestamp_ns": 0,
        "num_points": 1, "has_ring": False, "has_intensity": False,
        "has_point_time": False, "min_range_m": 1.0, "max_range_m": 1.0,
        "valid": True, "drop_reason": None,
    }])

    cfg = ComponentConfig()
    process_chunk(cfg, bag_id, chunk_id)

    world = np.load(local_path(lidar_world_path(bag_id, chunk_id, 0)))
    actual = np.array([world["x"][0], world["y"][0], world["z"][0]])
    np.testing.assert_allclose(actual, case["expected_world"], atol=1e-6)


def test_motion_compensation_with_per_point_timestamps(tmp_env):
    """Moving pose + per-point timestamps: the actual motion-comp path.

    Two poses, ego translates +10 m on X over 200 ms.  Two points share the
    same sensor-frame xyz but different `t_offset_us` (start vs. end of the
    sweep).  After deskew their world-frame X must differ by ~10 m, proving
    that batch_interpolate_poses handed each point a different pose.
    """
    bag_id, chunk_id = "bag_motion", "chunk0"
    _write_calibration(bag_id)

    T0 = np.eye(4)
    T1 = np.eye(4)
    T1[0, 3] = 10.0
    poses = [
        {
            "bag_id": bag_id, "chunk_id": chunk_id, "timestamp_ns": 0,
            "x": 0.0, "y": 0.0, "z": 0.0,
            "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
            "world_T_ego_flat": _flat(T0), "source": "odom", "valid": True,
        },
        {
            "bag_id": bag_id, "chunk_id": chunk_id, "timestamp_ns": 200_000_000,
            "x": 10.0, "y": 0.0, "z": 0.0,
            "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
            "world_T_ego_flat": _flat(T1), "source": "odom", "valid": True,
        },
    ]
    _write_poses(bag_id, chunk_id, poses)

    # Two points, both at sensor-frame (1, 0, 0).  point 0 is at the sweep
    # start (t=0 s), point 1 is at the sweep end (t=0.2 s).
    x = np.array([1.0, 1.0], dtype=np.float32)
    y = np.zeros(2, dtype=np.float32)
    z = np.zeros(2, dtype=np.float32)
    t_offset_us = np.array([0.0, 0.2], dtype=np.float32)  # config default unit = "seconds"
    _write_raw_sweep(bag_id, chunk_id, 0, x=x, y=y, z=z, t_offset_us=t_offset_us)

    from wato_common.artifact_store import lidar_sweep_path
    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    _write_sweep_index(bag_id, chunk_id, [{
        "bag_id": bag_id, "chunk_id": chunk_id,
        "lidar_id": "LIDAR_TOP", "sweep_id": 0,
        "lidar_path": lidar_sweep_path(bag_id, chunk_id, "LIDAR_TOP", 0),
        "header_timestamp_ns": 0, "record_timestamp_ns": 0,
        "num_points": 2, "has_ring": False, "has_intensity": False,
        "has_point_time": True, "min_range_m": 1.0, "max_range_m": 1.0,
        "valid": True, "drop_reason": None,
    }])

    cfg = ComponentConfig()
    results = process_chunk(cfg, bag_id, chunk_id)
    assert results[0].deskewed is True

    world = np.load(local_path(lidar_world_path(bag_id, chunk_id, 0)))
    # Point 0 (t=0): world ego at origin → sensor (1,0,0) → world (1,0,0).
    # Point 1 (t=0.2 s): world ego at (10,0,0) → sensor (1,0,0) → world (11,0,0).
    np.testing.assert_allclose(world["x"], [1.0, 11.0], atol=1e-6)
    np.testing.assert_allclose(world["y"], [0.0, 0.0], atol=1e-6)


def test_point_time_unit_mismatch_records_failure(tmp_env):
    """t_offset_us in microseconds but config says 'seconds' → sanity check fires.

    Per-sweep failure is caught and surfaces as a valid=False row, not a crash.
    """
    bag_id, chunk_id = "bag_unitmismatch", "chunk0"
    _write_calibration(bag_id)
    pose_row = {
        "bag_id": bag_id, "chunk_id": chunk_id, "timestamp_ns": 0,
        "x": 0.0, "y": 0.0, "z": 0.0,
        "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
        "world_T_ego_flat": _flat(np.eye(4)), "source": "odom", "valid": True,
    }
    _write_poses(bag_id, chunk_id, [pose_row])

    # 100_000 stored in a "seconds" field is 100_000 s = 1e14 ns of offset
    # — far above the 1 s sanity limit.
    x = np.ones(2, dtype=np.float32)
    y = np.zeros(2, dtype=np.float32)
    z = np.zeros(2, dtype=np.float32)
    t_offset_us = np.array([0.0, 100_000.0], dtype=np.float32)
    _write_raw_sweep(bag_id, chunk_id, 0, x=x, y=y, z=z, t_offset_us=t_offset_us)

    from wato_common.artifact_store import lidar_proc_index_path, lidar_sweep_path
    from wato_common.io.parquet_io import read_rows
    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    _write_sweep_index(bag_id, chunk_id, [{
        "bag_id": bag_id, "chunk_id": chunk_id,
        "lidar_id": "LIDAR_TOP", "sweep_id": 0,
        "lidar_path": lidar_sweep_path(bag_id, chunk_id, "LIDAR_TOP", 0),
        "header_timestamp_ns": 0, "record_timestamp_ns": 0,
        "num_points": 2, "has_ring": False, "has_intensity": False,
        "has_point_time": True, "min_range_m": 1.0, "max_range_m": 1.0,
        "valid": True, "drop_reason": None,
    }])

    cfg = ComponentConfig()  # default point_time_unit = "seconds"
    results = process_chunk(cfg, bag_id, chunk_id)
    # No DeskewResult appended (failed sweep), but a valid=False row exists.
    assert results == []
    rows = read_rows(lidar_proc_index_path(bag_id, chunk_id))
    assert len(rows) == 1
    assert rows[0]["valid"] is False
    assert "point_time_unit" in (rows[0]["drop_reason"] or "")
