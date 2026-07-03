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
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import LIDAR_SWEEPS_SCHEMA, POSES_SCHEMA
from wato_lidar_preprocessing.config import ComponentConfig, FrameSyncParams
from wato_lidar_preprocessing.deskew import _assign_frame_ids, process_chunk


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
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
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
    _write_sweep_index(
        bag_id,
        chunk_id,
        [
            {
                "bag_id": bag_id,
                "chunk_id": chunk_id,
                "lidar_id": "LIDAR_TOP",
                "sweep_id": 0,
                "lidar_path": lidar_sweep_path(bag_id, chunk_id, "LIDAR_TOP", 0),
                "header_timestamp_ns": 0,
                "record_timestamp_ns": 0,
                "num_points": 3,
                "has_ring": False,
                "has_intensity": False,
                "has_point_time": False,
                "min_range_m": 1.0,
                "max_range_m": 3.2,
                "valid": True,
                "drop_reason": None,
            }
        ],
    )

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
        "bag_id": bag_id,
        "chunk_id": chunk_id,
        "timestamp_ns": 0,
        "x": 10.0,
        "y": 0.0,
        "z": 0.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
        "world_T_ego_flat": _flat(T_shifted),
        "source": "odom",
        "valid": True,
    }
    _write_poses(bag_id, chunk_id, [pose_row])

    x = np.array([1.0, 2.0], dtype=np.float32)
    y = np.zeros(2, dtype=np.float32)
    z = np.zeros(2, dtype=np.float32)
    _write_raw_sweep(bag_id, chunk_id, 0, x=x, y=y, z=z)

    from wato_common.artifact_store import lidar_sweep_path

    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    _write_sweep_index(
        bag_id,
        chunk_id,
        [
            {
                "bag_id": bag_id,
                "chunk_id": chunk_id,
                "lidar_id": "LIDAR_TOP",
                "sweep_id": 0,
                "lidar_path": lidar_sweep_path(bag_id, chunk_id, "LIDAR_TOP", 0),
                "header_timestamp_ns": 0,
                "record_timestamp_ns": 0,
                "num_points": 2,
                "has_ring": False,
                "has_intensity": False,
                "has_point_time": False,
                "min_range_m": 1.0,
                "max_range_m": 2.0,
                "valid": True,
                "drop_reason": None,
            }
        ],
    )

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
        "bag_id": bag_id,
        "chunk_id": chunk_id,
        "timestamp_ns": 0,
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
        "world_T_ego_flat": _flat(np.eye(4)),
        "source": "odom",
        "valid": True,
    }
    _write_poses(bag_id, chunk_id, [pose_row])

    # 5 points, 2 of which are non-finite.
    x = np.array([1.0, np.nan, 2.0, np.inf, 3.0], dtype=np.float32)
    y = np.zeros(5, dtype=np.float32)
    z = np.zeros(5, dtype=np.float32)
    _write_raw_sweep(bag_id, chunk_id, 0, x=x, y=y, z=z)

    from wato_common.artifact_store import lidar_sweep_path

    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    _write_sweep_index(
        bag_id,
        chunk_id,
        [
            {
                "bag_id": bag_id,
                "chunk_id": chunk_id,
                "lidar_id": "LIDAR_TOP",
                "sweep_id": 0,
                "lidar_path": lidar_sweep_path(bag_id, chunk_id, "LIDAR_TOP", 0),
                "header_timestamp_ns": 0,
                "record_timestamp_ns": 0,
                "num_points": 5,
                "has_ring": False,
                "has_intensity": False,
                "has_point_time": False,
                "min_range_m": 1.0,
                "max_range_m": 3.0,
                "valid": True,
                "drop_reason": None,
            }
        ],
    )

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
        "bag_id": bag_id,
        "chunk_id": chunk_id,
        "timestamp_ns": 0,
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
        "world_T_ego_flat": _flat(np.eye(4)),
        "source": "odom",
        "valid": False,
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
        "bag_id": bag_id,
        "chunk_id": chunk_id,
        "timestamp_ns": 0,
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
        "world_T_ego_flat": _flat(T),
        "source": "odom",
        "valid": True,
    }
    _write_poses(bag_id, chunk_id, [pose_row])
    _write_raw_sweep(
        bag_id,
        chunk_id,
        0,
        x=np.ones(2, dtype=np.float32),
        y=np.zeros(2, dtype=np.float32),
        z=np.zeros(2, dtype=np.float32),
    )
    from wato_common.artifact_store import lidar_sweep_path

    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    _write_sweep_index(
        bag_id,
        chunk_id,
        [
            {
                "bag_id": bag_id,
                "chunk_id": chunk_id,
                "lidar_id": "LIDAR_TOP",
                "sweep_id": 0,
                "lidar_path": lidar_sweep_path(bag_id, chunk_id, "LIDAR_TOP", 0),
                "header_timestamp_ns": 0,
                "record_timestamp_ns": 0,
                "num_points": 2,
                "has_ring": False,
                "has_intensity": False,
                "has_point_time": False,
                "min_range_m": 1.0,
                "max_range_m": 1.0,
                "valid": True,
                "drop_reason": None,
            }
        ],
    )
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
        "calibration_version": "test",
        "ego_frame": "base_link",
        "cameras": {},
        "lidars": {"LIDAR_TOP": {"frame_id": "v", "ego_T_lidar": ego_T_lidar.tolist()}},
        "static_transforms": {},
        "checks": {"sanity": "ok", "notes": ""},
    }
    calib_p = local_path(calibration_path(bag_id))
    os.makedirs(os.path.dirname(calib_p), exist_ok=True)
    with open(calib_p, "w") as fh:
        json.dump(calib, fh)

    # Ego at world origin, identity rotation.
    pose_row = {
        "bag_id": bag_id,
        "chunk_id": chunk_id,
        "timestamp_ns": 0,
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
        "world_T_ego_flat": _flat(np.eye(4)),
        "source": "odom",
        "valid": True,
    }
    _write_poses(bag_id, chunk_id, [pose_row])

    _write_raw_sweep(
        bag_id,
        chunk_id,
        0,
        x=np.array([1.0], dtype=np.float32),
        y=np.zeros(1, dtype=np.float32),
        z=np.zeros(1, dtype=np.float32),
    )

    from wato_common.artifact_store import lidar_sweep_path

    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    _write_sweep_index(
        bag_id,
        chunk_id,
        [
            {
                "bag_id": bag_id,
                "chunk_id": chunk_id,
                "lidar_id": "LIDAR_TOP",
                "sweep_id": 0,
                "lidar_path": lidar_sweep_path(bag_id, chunk_id, "LIDAR_TOP", 0),
                "header_timestamp_ns": 0,
                "record_timestamp_ns": 0,
                "num_points": 1,
                "has_ring": False,
                "has_intensity": False,
                "has_point_time": False,
                "min_range_m": 1.0,
                "max_range_m": 1.0,
                "valid": True,
                "drop_reason": None,
            }
        ],
    )

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
            "bag_id": bag_id,
            "chunk_id": chunk_id,
            "timestamp_ns": 0,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
            "world_T_ego_flat": _flat(T0),
            "source": "odom",
            "valid": True,
        },
        {
            "bag_id": bag_id,
            "chunk_id": chunk_id,
            "timestamp_ns": 200_000_000,
            "x": 10.0,
            "y": 0.0,
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
            "world_T_ego_flat": _flat(T1),
            "source": "odom",
            "valid": True,
        },
    ]
    _write_poses(bag_id, chunk_id, poses)

    # Two points, both at sensor-frame (1, 0, 0).  point 0 is at the sweep
    # start (t=0 s), point 1 is at the sweep end (t=0.2 s).
    x = np.array([1.0, 1.0], dtype=np.float32)
    y = np.zeros(2, dtype=np.float32)
    z = np.zeros(2, dtype=np.float32)
    t_offset_us = np.array(
        [0.0, 0.2], dtype=np.float32
    )  # config default unit = "seconds"
    _write_raw_sweep(bag_id, chunk_id, 0, x=x, y=y, z=z, t_offset_us=t_offset_us)

    from wato_common.artifact_store import lidar_sweep_path

    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    _write_sweep_index(
        bag_id,
        chunk_id,
        [
            {
                "bag_id": bag_id,
                "chunk_id": chunk_id,
                "lidar_id": "LIDAR_TOP",
                "sweep_id": 0,
                "lidar_path": lidar_sweep_path(bag_id, chunk_id, "LIDAR_TOP", 0),
                "header_timestamp_ns": 0,
                "record_timestamp_ns": 0,
                "num_points": 2,
                "has_ring": False,
                "has_intensity": False,
                "has_point_time": True,
                "min_range_m": 1.0,
                "max_range_m": 1.0,
                "valid": True,
                "drop_reason": None,
            }
        ],
    )

    cfg = ComponentConfig()
    results = process_chunk(cfg, bag_id, chunk_id)
    assert results[0].deskewed is True

    world = np.load(local_path(lidar_world_path(bag_id, chunk_id, 0)))
    # Point 0 (t=0): world ego at origin → sensor (1,0,0) → world (1,0,0).
    # Point 1 (t=0.2 s): world ego at (10,0,0) → sensor (1,0,0) → world (11,0,0).
    np.testing.assert_allclose(world["x"], [1.0, 11.0], atol=1e-6)
    np.testing.assert_allclose(world["y"], [0.0, 0.0], atol=1e-6)


def test_deskew_raises_when_no_per_point_time_and_synthesis_disabled(tmp_env):
    """deskew refuses to silently fall back to header-pose-per-sweep.

    When a raw NPZ lacks t_offset_us AND synthesize_per_point_times=False
    AND allow_uncompensated_motion=False (defaults for strict mode), the
    sweep records a deskew_failed row instead of producing motion-uncomp
    output that smears statics across voxels.

    This is the safety net for the "buildings as dynamic" bug: a silent
    fallback was exactly what hid it.  We refuse the case explicitly.
    """
    bag_id, chunk_id = "bag_strict", "chunk0"
    _write_calibration(bag_id)
    _write_poses(
        bag_id,
        chunk_id,
        [
            {
                "bag_id": bag_id,
                "chunk_id": chunk_id,
                "timestamp_ns": 0,
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
                "world_T_ego_flat": _flat(np.eye(4)),
                "source": "odom",
                "valid": True,
            }
        ],
    )

    x = np.ones(2, dtype=np.float32)
    y = np.zeros(2, dtype=np.float32)
    z = np.zeros(2, dtype=np.float32)
    _write_raw_sweep(bag_id, chunk_id, 0, x=x, y=y, z=z)

    from wato_common.artifact_store import lidar_proc_index_path, lidar_sweep_path
    from wato_common.io.parquet_io import read_rows

    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    _write_sweep_index(
        bag_id,
        chunk_id,
        [
            {
                "bag_id": bag_id,
                "chunk_id": chunk_id,
                "lidar_id": "LIDAR_TOP",
                "sweep_id": 0,
                "lidar_path": lidar_sweep_path(bag_id, chunk_id, "LIDAR_TOP", 0),
                "header_timestamp_ns": 0,
                "record_timestamp_ns": 0,
                "num_points": 2,
                "has_ring": False,
                "has_intensity": False,
                "has_point_time": False,
                "min_range_m": 1.0,
                "max_range_m": 1.0,
                "valid": True,
                "drop_reason": None,
            }
        ],
    )

    cfg = ComponentConfig(
        synthesize_per_point_times=False,
        allow_uncompensated_motion=False,  # default — strict
    )
    results = process_chunk(cfg, bag_id, chunk_id)
    assert results == []
    rows = read_rows(lidar_proc_index_path(bag_id, chunk_id))
    assert len(rows) == 1
    assert rows[0]["valid"] is False
    assert "uncompensated" in (rows[0]["drop_reason"] or "")


def test_deskew_allows_uncompensated_motion_when_opted_in(tmp_env):
    """allow_uncompensated_motion=True bypasses the strict check.

    For users with stationary ego or who knowingly accept smear, the
    opt-out lets deskew use the header-pose-per-sweep fallback.
    """
    bag_id, chunk_id = "bag_optout", "chunk0"
    _write_calibration(bag_id)
    _write_poses(
        bag_id,
        chunk_id,
        [
            {
                "bag_id": bag_id,
                "chunk_id": chunk_id,
                "timestamp_ns": 0,
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
                "world_T_ego_flat": _flat(np.eye(4)),
                "source": "odom",
                "valid": True,
            }
        ],
    )

    x = np.ones(2, dtype=np.float32)
    y = np.zeros(2, dtype=np.float32)
    z = np.zeros(2, dtype=np.float32)
    _write_raw_sweep(bag_id, chunk_id, 0, x=x, y=y, z=z)

    from wato_common.artifact_store import lidar_sweep_path

    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    _write_sweep_index(
        bag_id,
        chunk_id,
        [
            {
                "bag_id": bag_id,
                "chunk_id": chunk_id,
                "lidar_id": "LIDAR_TOP",
                "sweep_id": 0,
                "lidar_path": lidar_sweep_path(bag_id, chunk_id, "LIDAR_TOP", 0),
                "header_timestamp_ns": 0,
                "record_timestamp_ns": 0,
                "num_points": 2,
                "has_ring": False,
                "has_intensity": False,
                "has_point_time": False,
                "min_range_m": 1.0,
                "max_range_m": 1.0,
                "valid": True,
                "drop_reason": None,
            }
        ],
    )

    cfg = ComponentConfig(
        synthesize_per_point_times=False,
        allow_uncompensated_motion=True,
    )
    results = process_chunk(cfg, bag_id, chunk_id)
    assert results, "with allow_uncompensated_motion=True the sweep should deskew"
    assert results[0].deskewed is False  # has_point_time was False


def test_synthesize_t_offset_from_azimuth_ccw_full_rotation():
    """Azimuth-based per-point time synthesis for a CCW rotating LiDAR.

    A CCW sensor's points at azimuth phi were fired at t = phi/(2π)*T_sweep.
    Four points at cardinal directions over a 1-second sweep should map to
    t = 0, 0.25, 0.5, 0.75 seconds.
    """
    from wato_lidar_preprocessing.deskew._core import (
        _synthesize_t_offset_ns_from_azimuth,
    )

    xyz = np.array(
        [
            [1.0, 0.0, 0.0],  # phi = 0      → t = 0
            [0.0, 1.0, 0.0],  # phi = π/2    → t = 0.25 s
            [-1.0, 0.0, 0.0],  # phi = π      → t = 0.5  s
            [0.0, -1.0, 0.0],  # phi = -π/2   → t = 0.75 s (wraps to +3π/2)
        ]
    )
    t_ns = _synthesize_t_offset_ns_from_azimuth(
        xyz, sweep_duration_ns=1_000_000_000.0, rotation_dir="ccw"
    )
    expected_ns = np.array([0.0, 0.25e9, 0.5e9, 0.75e9])
    np.testing.assert_allclose(t_ns, expected_ns, atol=1.0)


def test_synthesize_t_offset_anchored_at_phi_start():
    """phi_start = phi[0]: the sweep starts where the first point fired.

    NuScenes LIDAR_TOP starts scanning at azimuth ≈ -π (back of vehicle),
    NOT azimuth = 0.  The synthesis must anchor t=0 to phi[0], not to
    azimuth = 0, otherwise per-point times are offset by up to half a
    sweep duration and motion compensation goes in the wrong direction.
    """
    from wato_lidar_preprocessing.deskew._core import (
        _synthesize_t_offset_ns_from_azimuth,
    )

    # First point at phi = -π (back), then sweeping CCW.
    xyz = np.array(
        [
            [-1.0, 0.0, 0.0],  # phi = π (or -π)  → t = 0      (sweep start)
            [
                0.0,
                1.0,
                0.0,
            ],  # phi = π/2        → t = 0.75 s (CCW from -π is +3π/2 away)
            [1.0, 0.0, 0.0],  # phi = 0          → t = 0.5  s
            [0.0, -1.0, 0.0],  # phi = -π/2       → t = 0.25 s
        ]
    )
    t_ns = _synthesize_t_offset_ns_from_azimuth(
        xyz, sweep_duration_ns=1_000_000_000.0, rotation_dir="ccw"
    )
    # First point is t=0 by definition (it's the anchor).
    assert t_ns[0] == 0.0
    # CCW from phi_start = π (treating -π as +π for the wrap):
    # phi=π/2  → CCW delta = (π/2 - π) mod 2π = 3π/2 → t = 0.75 s
    # phi=0    → CCW delta = (0 - π) mod 2π   = π    → t = 0.5  s
    # phi=-π/2 → CCW delta = (-π/2 - π) mod 2π = π/2 → t = 0.25 s
    expected_ns = np.array([0.0, 0.75e9, 0.5e9, 0.25e9])
    np.testing.assert_allclose(t_ns, expected_ns, atol=1.0)


def test_synthesize_t_offset_cw_quarter_rotation_monotonic():
    """Explicit CW direction: azimuth decreasing from phi[0] yields monotonic t.

    NuScenes / Velodyne LIDAR_TOP rotates CW in the lidar frame (azimuth
    decreases with time). Direction is taken from the sensor_model profile
    (no per-sweep azimuth-sign probe), so we pass it explicitly here.
    """
    from wato_lidar_preprocessing.deskew._core import (
        _synthesize_t_offset_ns_from_azimuth,
    )

    # 250 points sweeping CW from phi=0 down to phi=-π/2 (quarter rotation).
    n = 250
    phi = np.linspace(0.0, -np.pi / 2, n)
    xyz = np.stack([np.cos(phi), np.sin(phi), np.zeros(n)], axis=1)
    t_ns = _synthesize_t_offset_ns_from_azimuth(
        xyz, sweep_duration_ns=1_000_000_000.0, rotation_dir="cw"
    )
    # Quarter rotation in 250 points → t goes from 0 to ~0.25 s, monotonically.
    assert t_ns[0] == 0.0
    assert 0.24e9 < t_ns[-1] < 0.26e9, (
        f"expected ~0.25s at end of quarter-rotation, got {t_ns[-1]/1e9:.3f}s"
    )
    assert np.all(np.diff(t_ns) >= 0), "t_ns must be monotonic for CW firing order"


def test_synthesize_t_offset_from_azimuth_cw_reverses():
    """CW rotation: t increases in the opposite azimuth direction."""
    from wato_lidar_preprocessing.deskew._core import (
        _synthesize_t_offset_ns_from_azimuth,
    )

    xyz = np.array(
        [
            [1.0, 0.0, 0.0],  # phi = 0      → t = 0
            [0.0, -1.0, 0.0],  # phi = -π/2   → t = 0.25 s (CW)
            [-1.0, 0.0, 0.0],  # phi = π      → t = 0.5  s
            [0.0, 1.0, 0.0],  # phi = π/2    → t = 0.75 s (CW)
        ]
    )
    t_ns = _synthesize_t_offset_ns_from_azimuth(
        xyz, sweep_duration_ns=1_000_000_000.0, rotation_dir="cw"
    )
    expected_ns = np.array([0.0, 0.25e9, 0.5e9, 0.75e9])
    np.testing.assert_allclose(t_ns, expected_ns, atol=1.0)


def test_synthesis_eliminates_intra_sweep_smear_for_static_wall(tmp_env):
    """End-to-end: missing per-point times + moving ego = smeared statics; fix removes the smear.

    Bug: when raw NPZ lacks t_offset_us, deskew uses one pose for the whole
    sweep.  For a rotating LiDAR with moving ego, this places points fired
    at different intra-sweep times at the wrong world positions — the
    "static wall smear" that leaks buildings into dynamic_map.npz.

    Setup: ego moves +10 m/s on X over a 1-second sweep (extreme to make the
    effect crisp).  Two LiDAR-frame points at azimuth 0 (front) and azimuth
    π (back) hit a static wall.  Because they're observed half a sweep
    apart, ego has moved 5 m between them in reality.

    Without the fix (synthesize_per_point_times=False): both points get the
    header pose at t=0, so the deskewed world X differs by exactly the
    lidar-frame offsets — the smear is visible.

    With the fix: the azimuth-π point gets the t=0.5 s pose, which puts
    the ego 5 m ahead, exactly compensating the lidar-frame offset.  Both
    deskewed world positions converge.
    """
    bag_id, chunk_id = "bag_synth", "chunk0"
    _write_calibration(bag_id)

    # Ego at (0,0,0) at t=0, at (10,0,0) at t=1 s.
    T0 = np.eye(4)
    T1 = np.eye(4)
    T1[0, 3] = 10.0
    poses = [
        {
            "bag_id": bag_id,
            "chunk_id": chunk_id,
            "timestamp_ns": 0,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
            "world_T_ego_flat": _flat(T0),
            "source": "odom",
            "valid": True,
        },
        {
            "bag_id": bag_id,
            "chunk_id": chunk_id,
            "timestamp_ns": 1_000_000_000,
            "x": 10.0,
            "y": 0.0,
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
            "world_T_ego_flat": _flat(T1),
            "source": "odom",
            "valid": True,
        },
    ]
    _write_poses(bag_id, chunk_id, poses)

    # Two LiDAR-frame points: one at azimuth 0 (front, t=0), one at azimuth
    # π (back, t=0.5 s).  Both hit the same true world position (5, 0, 0).
    # - At t=0:  ego at (0,0,0); wall at world (5,0,0) → sensor (5,0,0).
    # - At t=0.5 s: ego at (5,0,0); same wall → sensor (0,0,0) which would
    #   technically be at the lidar, so use a slightly offset wall position
    #   for the back-azimuth point that ALSO lives at world (5,0,0)... but
    #   azimuth π means the point is BEHIND the lidar at lidar-frame
    #   (-r, 0, 0).  Ego at (5,0,0) + sensor (-r, 0, 0) = world (5-r, 0, 0).
    #   For that to equal world (5, 0, 0), r=0 — pathological.
    # Instead, place wall_back at sensor (-5, 0, 0) at t=0.5 s:
    #   world = ego(5,0,0) + sensor(-5,0,0) = (0,0,0).
    # So front-wall world = (5,0,0); back-wall world = (0,0,0).  Both
    # static; with the fix they should land at these distinct positions.
    # Without the fix (single t=0 pose), back-wall world = ego(0,0,0) +
    # sensor(-5,0,0) = (-5,0,0) — wrong by 5 m on x.
    x = np.array([5.0, -5.0], dtype=np.float32)
    y = np.zeros(2, dtype=np.float32)
    z = np.zeros(2, dtype=np.float32)
    _write_raw_sweep(bag_id, chunk_id, 0, x=x, y=y, z=z)

    from wato_common.artifact_store import lidar_sweep_path

    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    _write_sweep_index(
        bag_id,
        chunk_id,
        [
            {
                "bag_id": bag_id,
                "chunk_id": chunk_id,
                "lidar_id": "LIDAR_TOP",
                "sweep_id": 0,
                "lidar_path": lidar_sweep_path(bag_id, chunk_id, "LIDAR_TOP", 0),
                "header_timestamp_ns": 0,
                "record_timestamp_ns": 0,
                "num_points": 2,
                "has_ring": False,
                "has_intensity": False,
                # has_point_time=False is the case this test exercises.
                "has_point_time": False,
                "min_range_m": 5.0,
                "max_range_m": 5.0,
                "valid": True,
                "drop_reason": None,
            }
        ],
    )

    # WITH the fix: synthesize_per_point_times=True, 1-second sweep. Rotation
    # direction comes from the sensor_model profile; for this symmetric
    # phi∈{0,π} pair cw and ccw both place the back point at t=0.5 s.
    cfg = ComponentConfig(
        synthesize_per_point_times=True,
        lidar_sweep_duration_ms=1_000.0,
    )
    results = process_chunk(cfg, bag_id, chunk_id)
    assert results, "deskew should have processed one sweep"

    world = np.load(local_path(lidar_world_path(bag_id, chunk_id, 0)))
    # Point 0 at azimuth 0, t=0: ego at (0,0,0) + sensor (5,0,0) = world (5,0,0).
    # Point 1 at azimuth π, t=0.5 s: ego at (5,0,0) + sensor (-5,0,0) = world (0,0,0).
    np.testing.assert_allclose(world["x"], [5.0, 0.0], atol=1e-3)

    # SANITY: without the fix, point 1 lands at world (-5,0,0) because both
    # points share the t=0 pose at world origin.  Re-run with the flag off.
    bag2, chunk2 = "bag_synth_off", "chunk0"
    _write_calibration(bag2)
    _write_poses(bag2, chunk2, poses)
    _write_raw_sweep(bag2, chunk2, 0, x=x, y=y, z=z)
    ensure_local_dir(lidar_proc_dir(bag2, chunk2))
    _write_sweep_index(
        bag2,
        chunk2,
        [
            {
                "bag_id": bag2,
                "chunk_id": chunk2,
                "lidar_id": "LIDAR_TOP",
                "sweep_id": 0,
                "lidar_path": lidar_sweep_path(bag2, chunk2, "LIDAR_TOP", 0),
                "header_timestamp_ns": 0,
                "record_timestamp_ns": 0,
                "num_points": 2,
                "has_ring": False,
                "has_intensity": False,
                "has_point_time": False,
                "min_range_m": 5.0,
                "max_range_m": 5.0,
                "valid": True,
                "drop_reason": None,
            }
        ],
    )
    cfg_off = ComponentConfig(
        synthesize_per_point_times=False,
        # Opt out of the strict-error gate — this test explicitly wants to
        # observe the smear, which is what allow_uncompensated_motion permits.
        allow_uncompensated_motion=True,
    )
    process_chunk(cfg_off, bag2, chunk2)
    world_off = np.load(local_path(lidar_world_path(bag2, chunk2, 0)))
    (
        np.testing.assert_allclose(world_off["x"], [5.0, -5.0], atol=1e-3),
        (
            "without synthesis, the back-azimuth point should land at world "
            "x=-5 (sensor (-5,0,0) + ego (0,0,0)), demonstrating the smear bug"
        ),
    )


def test_point_time_unit_mismatch_records_failure(tmp_env):
    """t_offset_us in microseconds but config says 'seconds' → sanity check fires.

    Per-sweep failure is caught and surfaces as a valid=False row, not a crash.
    """
    bag_id, chunk_id = "bag_unitmismatch", "chunk0"
    _write_calibration(bag_id)
    pose_row = {
        "bag_id": bag_id,
        "chunk_id": chunk_id,
        "timestamp_ns": 0,
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
        "world_T_ego_flat": _flat(np.eye(4)),
        "source": "odom",
        "valid": True,
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
    _write_sweep_index(
        bag_id,
        chunk_id,
        [
            {
                "bag_id": bag_id,
                "chunk_id": chunk_id,
                "lidar_id": "LIDAR_TOP",
                "sweep_id": 0,
                "lidar_path": lidar_sweep_path(bag_id, chunk_id, "LIDAR_TOP", 0),
                "header_timestamp_ns": 0,
                "record_timestamp_ns": 0,
                "num_points": 2,
                "has_ring": False,
                "has_intensity": False,
                "has_point_time": True,
                "min_range_m": 1.0,
                "max_range_m": 1.0,
                "valid": True,
                "drop_reason": None,
            }
        ],
    )

    cfg = ComponentConfig()  # default point_time_unit = "seconds"
    results = process_chunk(cfg, bag_id, chunk_id)
    # No DeskewResult appended (failed sweep), but a valid=False row exists.
    assert results == []
    rows = read_rows(lidar_proc_index_path(bag_id, chunk_id))
    assert len(rows) == 1
    assert rows[0]["valid"] is False
    assert "point_time_unit" in (rows[0]["drop_reason"] or "")


# ---------------------------------------------------------------------------
# Frame-id assignment (Issue E) — multi-lidar canonical-frame grouping.
# Tested as a pure function so we don't need to fabricate real sweeps.
# ---------------------------------------------------------------------------


def _row(lidar_id: str, ts_ns: int, *, valid: bool = True) -> dict:
    return {
        "lidar_id": lidar_id,
        "reference_timestamp_ns": ts_ns,
        "valid": valid,
    }


def test_assign_frame_ids_single_lidar_sequential():
    """canonical_lidar=None → per-lidar sequential frame_ids by timestamp."""
    rows = [
        _row("LIDAR_TOP", 0),
        _row("LIDAR_TOP", 100_000_000),
        _row("LIDAR_TOP", 200_000_000),
    ]
    _assign_frame_ids(rows, FrameSyncParams(canonical_lidar=None))
    assert [r["frame_id"] for r in rows] == [0, 1, 2]


def test_assign_frame_ids_invalid_rows_get_none():
    rows = [
        _row("LIDAR_TOP", 0, valid=False),
        _row("LIDAR_TOP", 100_000_000),
        _row("LIDAR_TOP", 200_000_000),
    ]
    _assign_frame_ids(rows, FrameSyncParams(canonical_lidar=None))
    assert rows[0]["frame_id"] is None
    # The two valid rows form a sequence on their own.
    assert rows[1]["frame_id"] == 0
    assert rows[2]["frame_id"] == 1


def test_assign_frame_ids_canonical_with_tolerance():
    """Non-canonical sweeps within tolerance share canonical's frame_id."""
    # 3-LiDAR rig.  Frame 0 fires near t=0; frame 1 fires near t=100ms.
    # Center is canonical.  NE/NW arrive ~10ms after center (within 25ms).
    rows = [
        _row("lidar_ne", 10_000_000),  # +10 ms from canonical@0
        _row("lidar_cc", 0),
        _row("lidar_nw", -8_000_000),  # -8 ms from canonical@0
        _row("lidar_cc", 100_000_000),
        _row("lidar_ne", 105_000_000),  # +5 ms from canonical@100ms
        _row("lidar_nw", 130_000_000),  # +30 ms from canonical@100ms → outside ±25ms
    ]
    _assign_frame_ids(
        rows,
        FrameSyncParams(canonical_lidar="lidar_cc", tolerance_ms=25.0),
    )
    by_lidar_ts = {
        (r["lidar_id"], r["reference_timestamp_ns"]): r["frame_id"] for r in rows
    }
    # Canonical sweeps get sequential frame_ids.
    assert by_lidar_ts[("lidar_cc", 0)] == 0
    assert by_lidar_ts[("lidar_cc", 100_000_000)] == 1
    # Within-tolerance non-canonical inherit.
    assert by_lidar_ts[("lidar_ne", 10_000_000)] == 0
    assert by_lidar_ts[("lidar_nw", -8_000_000)] == 0
    assert by_lidar_ts[("lidar_ne", 105_000_000)] == 1
    # Outside tolerance → None (orphan).
    assert by_lidar_ts[("lidar_nw", 130_000_000)] is None


def test_assign_frame_ids_canonical_missing_falls_back(caplog):
    """canonical_lidar configured but not in chunk → log warning + per-lidar fallback."""
    rows = [
        _row("lidar_ne", 0),
        _row("lidar_ne", 100_000_000),
        _row("lidar_nw", 0),
    ]
    with caplog.at_level("WARNING", logger="wato_lidar_preprocessing.deskew"):
        _assign_frame_ids(
            rows,
            FrameSyncParams(canonical_lidar="lidar_cc", tolerance_ms=25.0),
        )
    assert "not present in chunk" in caplog.text
    # Each lidar gets its own sequential frame_ids.
    by_lidar_ts = {
        (r["lidar_id"], r["reference_timestamp_ns"]): r["frame_id"] for r in rows
    }
    assert by_lidar_ts[("lidar_ne", 0)] == 0
    assert by_lidar_ts[("lidar_ne", 100_000_000)] == 1
    assert by_lidar_ts[("lidar_nw", 0)] == 0


def test_process_chunk_populates_frame_id(tmp_env):
    """End-to-end: deskew populates frame_id in lidar_proc_index.parquet."""
    bag_id, chunk_id = "bag_frameid", "chunk0"
    _write_calibration(bag_id)
    pose_row = {
        "bag_id": bag_id,
        "chunk_id": chunk_id,
        "timestamp_ns": 0,
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
        "world_T_ego_flat": _flat(np.eye(4)),
        "source": "odom",
        "valid": True,
    }
    _write_poses(bag_id, chunk_id, [pose_row])

    from wato_common.artifact_store import lidar_proc_index_path, lidar_sweep_path

    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))

    sweep_rows = []
    for sid in range(3):
        _write_raw_sweep(
            bag_id,
            chunk_id,
            sid,
            x=np.ones(1, dtype=np.float32),
            y=np.zeros(1, dtype=np.float32),
            z=np.zeros(1, dtype=np.float32),
        )
        sweep_rows.append(
            {
                "bag_id": bag_id,
                "chunk_id": chunk_id,
                "lidar_id": "LIDAR_TOP",
                "sweep_id": sid,
                "lidar_path": lidar_sweep_path(bag_id, chunk_id, "LIDAR_TOP", sid),
                "header_timestamp_ns": sid * 100_000_000,
                "record_timestamp_ns": 0,
                "num_points": 1,
                "has_ring": False,
                "has_intensity": False,
                "has_point_time": False,
                "min_range_m": 1.0,
                "max_range_m": 1.0,
                "valid": True,
                "drop_reason": None,
            }
        )
    _write_sweep_index(bag_id, chunk_id, sweep_rows)

    cfg = ComponentConfig()  # canonical_lidar=None → per-lidar sequential
    process_chunk(cfg, bag_id, chunk_id)

    rows = read_rows(lidar_proc_index_path(bag_id, chunk_id))
    assert [r["frame_id"] for r in rows] == [0, 1, 2]
