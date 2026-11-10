"""detection_3d end to end on a hand-built chunk, with a fake model in place
of BEVFusion: what goes in (points, ages, images, calibration) and what is
written (frames, world boxes)."""

from __future__ import annotations

import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from wato_common.artifact_store import (
    bag_root,
    camera_image_path,
    chunks_index_path,
    ensure_local_dir,
    lidar_proc_index_path,
    local_path,
    poses_path,
)
from wato_common.artifacts import calibration, detections_3d, frame_index
from wato_common.artifacts.calibration import (
    Calibration,
    CameraCalibration,
    LidarCalibration,
)
from wato_common.artifacts.lidar_sweeps import WorldSweep, write_world
from wato_common.io.parquet_io import write_table
from wato_common.schemas import (
    POSES_SCHEMA,
    PROCESSED_SWEEPS_SCHEMA,
    FrameIndexRow,
    ProcessedSweepMeta,
)
from wato_detection_3d import model, pipeline
from wato_detection_3d.config import MODEL_FILES, ComponentConfig

MS = 1_000_000
BAG, CHUNK = "b", "0000"
CAMS = ["CAM_A", "CAM_B", "CAM_C", "CAM_D", "CAM_E", "CAM_F"]
LIDAR_Z = 1.8


@pytest.fixture(autouse=True)
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")


def _cfg(**overrides) -> ComponentConfig:
    data = dict(
        model={"sha256": {f: "0" * 64 for f in MODEL_FILES}},
        cameras=CAMS,
        input_frame={"kind": "sensor"},
    )
    data.update(overrides)
    return ComponentConfig.model_validate(data)


def _ego_T_lidar() -> np.ndarray:
    m = np.eye(4)
    m[2, 3] = LIDAR_Z
    return m


def _write_chunk(sweep_ms=(0, 100, 200, 300), cams_missing_after_ms=None):
    """A stationary ego at the world origin, one LiDAR at 10 Hz, six
    1600x900 cameras at 10 Hz offset by 20 ms."""
    ensure_local_dir(local_path(bag_root(BAG)) + "/chunks")
    pq.write_table(
        pa.table({"chunk_id": [CHUNK], "bag_id": [BAG]}),
        local_path(chunks_index_path(BAG)),
    )
    K = np.array([[1266.0, 0.0, 800.0], [0.0, 1266.0, 490.0], [0.0, 0.0, 1.0]])
    cam_T = np.array(
        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 1.5], [0, 0, 0, 1]], float
    )
    calibration.write(
        BAG,
        Calibration(
            cameras={
                c: CameraCalibration(
                    c, K, cam_T, width=1600, height=900, rectified=True
                )
                for c in CAMS
            },
            lidars={"top": LidarCalibration("top", _ego_T_lidar())},
        ),
    )
    write_table(
        [
            {
                "bag_id": BAG,
                "chunk_id": CHUNK,
                "timestamp_ns": t * MS,
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
                "world_T_ego_flat": np.eye(4).flatten().tolist(),
                "source": "/odom",
                "valid": True,
                "interval_drop_reason": None,
            }
            for t in range(-100, 600, 50)
        ],
        POSES_SCHEMA,
        poses_path(BAG, CHUNK),
    )
    metas, rows = [], []
    ensure_local_dir(
        local_path(lidar_proc_index_path(BAG, CHUNK)).rsplit("/", 1)[0] + "/lidar_proc"
    )
    for sid, t in enumerate(sweep_ms):
        # A point 10 m ahead of the sensor, one 0.5 m from it (dropped from
        # earlier sweeps), one out of the voxel range.
        xyz = np.array(
            [[10.0, 0.0, LIDAR_Z], [0.5, 0.0, LIDAR_Z], [80.0, 0.0, LIDAR_Z]]
        )
        sweep = WorldSweep(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            intensity=np.full(3, 20.0, np.float32),
        )
        metas.append(
            ProcessedSweepMeta(
                bag_id=BAG,
                chunk_id=CHUNK,
                sweep_id=sid,
                lidar_id="top",
                reference_timestamp_ns=t * MS,
                n_points_total=3,
                n_points_static=3,
                n_points_dynamic=0,
                world_path=write_world(BAG, CHUNK, sid, sweep),
                dynamic_mask_path="",
                has_intensity=True,
                deskewed=True,
                frame_id=sid,
            ).model_dump()
        )
        for c in CAMS:
            cam_t = t + 20
            if (
                cams_missing_after_ms is not None
                and c == "CAM_F"
                and cam_t > cams_missing_after_ms
            ):
                continue
            path = camera_image_path(BAG, CHUNK, c, sid, "png")
            ensure_local_dir(local_path(path).rsplit("/", 1)[0])
            Image.new("RGB", (1600, 900), (40, 40, 40)).save(local_path(path))
            rows.append(
                FrameIndexRow(
                    frame_id=f"{sid}_{c}",
                    bag_id=BAG,
                    chunk_id=CHUNK,
                    sweep_id=sid,
                    lidar_id="top",
                    lidar_path="",
                    reference_timestamp_ns=t * MS,
                    cam_id=c,
                    image_path=path,
                    camera_seq=sid,
                    camera_timestamp_ns=cam_t * MS,
                    valid_camera=True,
                    valid_pose=True,
                )
            )
    write_table(metas, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(BAG, CHUNK))
    frame_index.write(BAG, CHUNK, rows)


class _Fake:
    """Returns one car 10 m ahead of the sensor, facing forward."""

    def __init__(self):
        self.calls = []

    def infer(self, points, cameras):
        self.calls.append((points.copy(), cameras))
        box = np.zeros(11, np.float32)
        # Model frame = the LiDAR (kind sensor): x forward. Facing +x is
        # heading 0 = -rot - pi/2, so rot = -pi/2.
        box[[model.X, model.Y, model.Z]] = (10.0, 0.0, -1.0)
        box[[model.W, model.L, model.H]] = (1.9, 4.5, 1.6)
        box[model.ROT] = -math.pi / 2
        box[[model.VX, model.VY]] = (2.0, 0.0)
        box[model.CLASS_ID] = 0
        box[model.SCORE] = 0.7
        return box[None]


def _run(cfg, fake):
    pipeline.run(
        cfg,
        bag_id=BAG,
        make_detector=lambda: (fake, {"model_dir": "m", "model_files_sha256": {}}),
    )


def test_every_frame_runs_with_earlier_sweeps_and_world_boxes():
    _write_chunk()
    fake = _Fake()
    _run(_cfg(), fake)
    frames, dets = detections_3d.read(BAG, CHUNK)
    assert [f.status for f in frames] == ["ok"] * 4
    assert frames[3].input_sweep_ids == (0, 1, 2, 3)
    assert frames[0].input_sweep_ids == (0,)
    # The last frame: the current sweep's two in-range points at age 0, then
    # the far point only of each earlier sweep (its close point is removed).
    points = fake.calls[3][0]
    assert points.shape == (5, 5)
    np.testing.assert_allclose(
        sorted(points[:, 4]), [0.0, 0.0, 0.1, 0.2, 0.3], atol=1e-6
    )
    np.testing.assert_allclose(points[:, 3], 20.0)
    assert frames[3].camera_timestamps_ns == tuple([320 * MS] * 6)
    np.testing.assert_allclose(frames[3].world_T_model, _ego_T_lidar())
    d = dets[3]
    assert (
        d.detection_id == "det-000003-0"
        and d.cls == "car"
        and d.score == pytest.approx(0.7)
    )
    assert (d.cx, d.cy, d.cz) == pytest.approx((10.0, 0.0, LIDAR_Z - 1.0))
    assert d.heading == pytest.approx(0.0, abs=1e-6)
    assert (d.vx, d.vy) == pytest.approx((2.0, 0.0))
    assert d.model_rot == pytest.approx(-math.pi / 2)


def test_images_arrive_as_the_models_1600x900_with_their_calibration():
    _write_chunk()
    fake = _Fake()
    _run(_cfg(), fake)
    cams = fake.calls[0][1]
    assert cams.images.shape == (6, 900, 1600, 3) and cams.images.dtype == np.uint8
    np.testing.assert_allclose(
        cams.intrinsics[0, :3, :3], [[1266, 0, 800], [0, 1266, 490], [0, 0, 1]]
    )
    np.testing.assert_allclose(
        cams.lidar2image[0],
        cams.intrinsics[0] @ np.linalg.inv(cams.camera2lidar[0]),
        atol=1e-4,
    )


def test_a_frame_without_one_cameras_image_is_skipped_and_said_why():
    _write_chunk(cams_missing_after_ms=150)
    _run(_cfg(camera_sync={"max_offset_ms": 60.0}), _Fake())
    frames, dets = detections_3d.read(BAG, CHUNK)
    assert [f.status for f in frames] == [
        "ok",
        "ok",
        "missing_camera",
        "missing_camera",
    ]
    assert frames[2].detail == "CAM_F"
    assert {d.sweep_id for d in dets} == {0, 1}


def test_completed_chunks_are_skipped_unless_forced():
    _write_chunk()
    fake = _Fake()
    _run(_cfg(), fake)
    _run(_cfg(), fake)
    assert len(fake.calls) == 4
    pipeline.run(
        _cfg(),
        bag_id=BAG,
        force=True,
        make_detector=lambda: (fake, {"model_dir": "m", "model_files_sha256": {}}),
    )
    assert len(fake.calls) == 8
