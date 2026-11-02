"""wato_common.artifacts.frame_index: what ingest writes, consumers read back."""

from __future__ import annotations

import numpy as np
import pytest

from wato_common.artifact_store import poses_path
from wato_common.artifacts import ArtifactFormatError, frame_index
from wato_common.artifacts.frame_index import FrameIndex
from wato_common.geometry import PoseSample
from wato_common.io.parquet_io import write_table
from wato_common.pose_lookup import interval_drop_reasons
from wato_common.schemas import POSES_SCHEMA, FrameIndexRow

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


def _row(
    sweep_id: int,
    cam_seq: int | None,
    cam_ms: int | None,
    *,
    lidar_id="lidar_cc",
    cam_id="cam_front",
    valid_cam=True,
) -> FrameIndexRow:
    return FrameIndexRow(
        frame_id=f"{sweep_id}_{cam_id}",
        bag_id=BAG,
        chunk_id=CHUNK,
        sweep_id=sweep_id,
        lidar_id=lidar_id,
        lidar_path="",
        reference_timestamp_ns=sweep_id * 50 * MS,
        cam_id=cam_id,
        image_path=None if cam_seq is None else f"/img/{cam_id}/{cam_seq:06d}.jpg",
        camera_seq=cam_seq,
        camera_timestamp_ns=None if cam_ms is None else cam_ms * MS,
        valid_camera=valid_cam,
        valid_pose=True,
    )


def _load(rows: list[FrameIndexRow]) -> FrameIndex:
    frame_index.write(BAG, CHUNK, rows)
    return FrameIndex.load(BAG, CHUNK)


def test_an_image_paired_with_sweeps_of_several_lidars_is_one_frame():
    # The 3-LiDAR WATO shape: one image nearest to a sweep of each scanner.
    _write_poses([0, 100, 200, 300])
    index = _load(
        [
            _row(4, 7, 140, lidar_id="lidar_ne"),
            _row(3, 7, 140, lidar_id="lidar_cc"),
            _row(5, 7, 140, lidar_id="lidar_nw"),
            _row(6, 8, 225, lidar_id="lidar_cc"),
        ]
    )
    first, second = index.frames
    assert (first.camera_seq, first.sweep_ids) == (7, (3, 4, 5))
    assert first.image_path == "/img/cam_front/000007.jpg"
    assert (second.camera_seq, second.sweep_ids) == (8, (6,))
    assert index.for_sweep(3)[0] is index.for_sweep(5)[0] is first


def test_each_frame_is_posed_at_its_own_time_not_the_sweeps():
    _write_poses([0, 100, 200, 300])
    (frame,) = _load([_row(2, 7, 140)]).frames  # the sweep is at 100 ms
    assert frame.timestamp_ns == 140 * MS
    np.testing.assert_allclose(frame.world_T_ego[:3, 3], [1.4, 0.0, 0.0])


def test_by_camera_and_for_sweep():
    _write_poses([0, 100, 200, 300])
    index = _load(
        [
            _row(0, 9, 95, cam_id="cam_back"),
            _row(0, 7, 105),
            _row(1, 8, 160),
            _row(1, 9, 95, cam_id="cam_back"),
        ]
    )
    assert {
        cam: [f.camera_seq for f in fs] for cam, fs in index.by_camera().items()
    } == {"cam_back": [9], "cam_front": [7, 8]}
    assert [(f.cam_id, f.camera_seq) for f in index.for_sweep(1)] == [
        ("cam_back", 9),
        ("cam_front", 8),
    ]
    assert index.for_sweep(99) == []


def test_untrusted_pose_at_camera_time_and_dropped_cameras():
    _write_poses([0, 100, 1_100])  # 1 s dropout after 100 ms
    index = _load(
        [
            _row(0, 1, 60),
            _row(1, 2, 500),
            _row(2, None, None, valid_cam=False),
        ]
    )
    trusted, in_gap = index.frames
    assert trusted.world_T_ego is not None
    assert in_gap.camera_seq == 2 and in_gap.world_T_ego is None
    assert index.for_sweep(2) == []


def test_a_valid_camera_row_without_a_frame_is_rejected():
    _write_poses([0, 100, 200])
    with pytest.raises(ArtifactFormatError, match="without camera_seq"):
        _load([_row(0, None, None)])
