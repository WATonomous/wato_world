"""End to end over one staged chunk: every upstream artifact semantic_lifting
reads, written in the format its producer writes, and points inside a mask
come out labeled."""

from __future__ import annotations

import os

import numpy as np
import pytest

from wato_common.artifact_store import (
    chunks_index_path,
    frame_index_path,
    lidar_proc_index_path,
    lidar_proc_dir,
    local_path,
    poses_path,
)
from wato_common.artifacts import (
    calibration,
    depth,
    lidar_sweeps,
    lifted_labels,
    masklets,
)
from wato_common.artifacts.calibration import (
    Calibration,
    CameraCalibration,
    LidarCalibration,
)
from wato_common.io.parquet_io import write_table
from wato_common.schemas import (
    CHUNK_SCHEMA,
    FRAME_INDEX_SCHEMA,
    POSES_SCHEMA,
    PROCESSED_SWEEPS_SCHEMA,
    FrameIndexRow,
    ProcessedSweepMeta,
)
from wato_semantic_lifting.config import ComponentConfig
from wato_semantic_lifting.pipeline import run

MS = 1_000_000
BAG, CHUNK, CAM, SEQ, SWEEP = "b", "0000", "CAM_FRONT", 7, 3
W, H = 64, 48
K = [[50.0, 0.0, 32.0], [0.0, 50.0, 24.0], [0.0, 0.0, 1.0]]
# Camera looking along ego +x: optical z = ego x, x = -ego y, y = -ego z.
EGO_T_CAM = [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
# World == ego (identity pose). Points 0-2 land inside the mask, 3 beside it,
# 4 behind the camera.
POINTS = np.array(
    [[10.0, 0.0, 0.0], [10.0, 0.5, 0.0], [10.0, -0.5, 0.2], [10.0, 3.0, 0.0],
     [-5.0, 0.0, 0.0]]
)  # fmt: skip


@pytest.fixture(autouse=True)
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")


def _stage_chunk() -> None:
    write_table(
        [
            {
                "bag_id": BAG,
                "chunk_id": CHUNK,
                "t_start_ns": 0,
                "t_end_ns": 200 * MS,
                "t_overlap_start_ns": 0,
                "t_overlap_end_ns": 200 * MS,
            }
        ],
        CHUNK_SCHEMA,
        chunks_index_path(BAG),
    )
    pose = {"x": 0.0, "y": 0.0, "z": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
    write_table(
        [
            {
                "bag_id": BAG,
                "chunk_id": CHUNK,
                "timestamp_ns": t * MS,
                **pose,
                "world_T_ego_flat": np.eye(4).flatten().tolist(),
                "source": "/odom",
                "valid": True,
                "interval_drop_reason": None,
            }
            for t in (0, 200)
        ],
        POSES_SCHEMA,
        poses_path(BAG, CHUNK),
    )
    write_table(
        [
            FrameIndexRow(
                frame_id="f0",
                bag_id=BAG,
                chunk_id=CHUNK,
                sweep_id=SWEEP,
                lidar_id="lidar_top",
                lidar_path="",
                reference_timestamp_ns=100 * MS,
                cam_id=CAM,
                camera_seq=SEQ,
                camera_timestamp_ns=100 * MS,
                valid_camera=True,
                world_T_ego_flat=np.eye(4).flatten().tolist(),
                valid_pose=True,
            ).model_dump()
        ],
        FRAME_INDEX_SCHEMA,
        frame_index_path(BAG, CHUNK),
    )
    calibration.write(
        BAG,
        Calibration(
            cameras={
                CAM: CameraCalibration(
                    cam_id=CAM, K=np.array(K), ego_T_cam=np.array(EGO_T_CAM, float)
                )
            },
            lidars={"lidar_top": LidarCalibration("lidar_top", np.eye(4))},
        ),
    )

    os.makedirs(local_path(lidar_proc_dir(BAG, CHUNK)), exist_ok=True)
    world_uri = lidar_sweeps.write_world(
        BAG,
        CHUNK,
        SWEEP,
        lidar_sweeps.WorldSweep(
            x=POINTS[:, 0], y=POINTS[:, 1], z=POINTS[:, 2], origin=np.zeros(3)
        ),
    )
    mask_uri = lidar_sweeps.write_dynamic_mask(
        BAG, CHUNK, SWEEP, np.zeros(len(POINTS), dtype=bool)
    )
    write_table(
        [
            ProcessedSweepMeta(
                bag_id=BAG,
                chunk_id=CHUNK,
                sweep_id=SWEEP,
                lidar_id="lidar_top",
                reference_timestamp_ns=100 * MS,
                n_points_total=len(POINTS),
                n_points_static=len(POINTS),
                n_points_dynamic=0,
                world_path=world_uri,
                dynamic_mask_path=mask_uri,
                has_intensity=False,
                deskewed=True,
            ).model_dump()
        ],
        PROCESSED_SWEEPS_SCHEMA,
        lidar_proc_index_path(BAG, CHUNK),
    )

    depth.write_frame(
        BAG,
        CHUNK,
        CAM,
        SEQ,
        depth.DepthFrame(
            depth_m=np.full((H, W), 10.0, dtype=np.float16),
            affine_a=1.0,
            affine_b=0.0,
            n_anchors=10,
            n_inliers=10,
            rmse_inliers_m=0.1,
            fit_status=depth.FitStatus.OK,
        ),
    )

    mask = np.zeros((H, W), dtype=bool)
    mask[16:33, 20:45] = True
    masklets.write_mask(BAG, CHUNK, "m0", SEQ, mask)
    masklets.write(
        BAG,
        CHUNK,
        [
            masklets.MaskletRecord(
                masklet_id="m0",
                cam_id=CAM,
                cls="vehicle",
                score=0.8,
                frames_present=(SEQ,),
                det_score=0.8,
            )
        ],
    )


def test_points_inside_a_mask_are_labeled():
    _stage_chunk()
    run(ComponentConfig(), bag_id=BAG)

    (stats,) = lifted_labels.read_stats(BAG, CHUNK)
    assert stats.sweep_id == SWEEP
    assert stats.n_points_total == len(POINTS)
    assert stats.n_points_labeled == 3
    assert stats.n_cameras_used == 1

    labels = lifted_labels.read_sweep(BAG, CHUNK, SWEEP, n_points=len(POINTS))
    assert labels.point_indices.tolist() == [0, 1, 2]
    assert labels.classes.tolist() == ["vehicle"] * 3
    assert labels.instance_ids.tolist() == ["m0"] * 3
