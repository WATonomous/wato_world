"""End to end over one staged chunk: every upstream artifact semantic_lifting
reads, written in the format its producer writes."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from wato_common.artifact_store import (
    chunks_index_path,
    frame_index_path,
    lidar_proc_index_path,
    lidar_proc_dir,
    lidar_world_path,
    local_path,
    poses_path,
)
from wato_common.artifacts import (
    ArtifactFormatError,
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


def _stage_chunk(
    cams: tuple[tuple[str, float], ...] = ((CAM, 0.8),),
    depth_m: float = 10.0,
    points: np.ndarray = POINTS,
    ground_mask: bool = True,
) -> None:
    """One sweep of `points`; per (camera, det_score) one frame, all cameras
    looking along ego +x, a masklet m_<camera> over the pixels points 0-2 land
    on, and a depth map of depth_m everywhere (None: no depth artifact)."""
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
                frame_id=f"f{i}",
                bag_id=BAG,
                chunk_id=CHUNK,
                sweep_id=SWEEP,
                lidar_id="lidar_top",
                lidar_path="",
                reference_timestamp_ns=100 * MS,
                cam_id=cam,
                camera_seq=SEQ,
                camera_timestamp_ns=100 * MS,
                valid_camera=True,
                world_T_ego_flat=np.eye(4).flatten().tolist(),
                valid_pose=True,
            ).model_dump()
            for i, (cam, _) in enumerate(cams)
        ],
        FRAME_INDEX_SCHEMA,
        frame_index_path(BAG, CHUNK),
    )
    calibration.write(
        BAG,
        Calibration(
            cameras={
                cam: CameraCalibration(
                    cam_id=cam, K=np.array(K), ego_T_cam=np.array(EGO_T_CAM, float)
                )
                for cam, _ in cams
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
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            origin=np.zeros(3),
            ground_mask=np.zeros(len(points), bool) if ground_mask else None,
        ),
    )
    mask_uri = lidar_sweeps.write_dynamic_mask(
        BAG, CHUNK, SWEEP, np.zeros(len(points), dtype=bool)
    )
    write_table(
        [
            ProcessedSweepMeta(
                bag_id=BAG,
                chunk_id=CHUNK,
                sweep_id=SWEEP,
                lidar_id="lidar_top",
                reference_timestamp_ns=100 * MS,
                n_points_total=len(points),
                n_points_static=len(points),
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

    mask = np.zeros((H, W), dtype=bool)
    mask[16:33, 20:45] = True
    records = []
    for cam, det_score in cams:
        if depth_m is not None:
            depth.write_frame(
                BAG,
                CHUNK,
                cam,
                SEQ,
                depth.DepthFrame(
                    depth_m=np.full((H, W), depth_m, dtype=np.float16),
                    disp_scale=1.0,
                    disp_shift=0.0,
                    n_anchors=10,
                    n_inliers=10,
                    rmse_inliers_m=0.1,
                    fit_status=depth.FitStatus.OK,
                ),
            )
        masklets.write_mask(BAG, CHUNK, f"m_{cam}", SEQ, mask)
        records.append(
            masklets.MaskletRecord(
                masklet_id=f"m_{cam}",
                cam_id=cam,
                cls="vehicle",
                score=det_score,
                frames_present=(SEQ,),
                det_score=det_score,
            )
        )
    masklets.write(BAG, CHUNK, records)


def test_points_inside_a_mask_are_labeled():
    _stage_chunk()
    run(ComponentConfig(), bag_id=BAG)

    (stats,) = lifted_labels.read_stats(BAG, CHUNK)
    assert stats.sweep_id == SWEEP
    assert stats.n_points_total == len(POINTS)
    assert stats.n_points_labeled == stats.n_points_in_any_mask == 3
    assert stats.n_points_failed_visibility == 0
    assert stats.n_cameras_used == 1

    labels = lifted_labels.read_sweep(BAG, CHUNK, SWEEP, n_points=len(POINTS))
    assert labels.point_indices.tolist() == [0, 1, 2]
    assert labels.classes.tolist() == ["vehicle"] * 3
    assert labels.instance_ids.tolist() == [f"m_{CAM}"] * 3
    (inst,) = lifted_labels.read_instances(BAG, CHUNK)
    assert inst.masklet_ids == (f"m_{CAM}",) and inst.n_points_labeled == 3


def test_points_behind_a_nearer_surface_take_no_label():
    """The mask covers points 0-2, but the depth map has a surface at 5 m in
    front of them (10 m): the mask belongs to that surface, not to them."""
    _stage_chunk(depth_m=5.0)
    run(ComponentConfig(), bag_id=BAG)

    (stats,) = lifted_labels.read_stats(BAG, CHUNK)
    assert stats.n_points_in_any_mask == 3
    assert stats.n_points_labeled == 0
    assert stats.n_points_failed_visibility == 3
    assert lifted_labels.read_sweep(BAG, CHUNK, SWEEP).point_indices.size == 0


def test_points_behind_a_nearer_lidar_return_take_no_label():
    """A return at 5 m on point 0's line of sight hides points 0-2 (10 m) from
    the camera, though the depth map (10 m everywhere) doesn't show it."""
    near = np.vstack([POINTS, [[5.0, 0.0, 0.0]]])
    _stage_chunk(points=near)
    run(ComponentConfig(), bag_id=BAG)
    labels = lifted_labels.read_sweep(BAG, CHUNK, SWEEP, n_points=len(near))
    assert labels.point_indices.tolist() == [5]


def test_a_far_depth_map_surface_does_not_occlude():
    """The depth map puts a surface at 20 m in front of points 0-2 (40 m).
    Past the map's trusted range that is its own error, not an object, and no
    LiDAR return is in front of them."""
    far = POINTS.copy()
    far[:4] *= 4.0  # same pixels, four times as far
    _stage_chunk(points=far, depth_m=20.0)
    run(ComponentConfig(), bag_id=BAG)
    assert lifted_labels.read_sweep(BAG, CHUNK, SWEEP).point_indices.tolist() == [
        0,
        1,
        2,
    ]


def test_a_sweep_without_a_ground_mask_stops_the_run():
    _stage_chunk(ground_mask=False)
    with pytest.raises(ArtifactFormatError, match="ground_mask"):
        run(ComponentConfig(), bag_id=BAG)


def test_two_cameras_on_one_object_agree():
    _stage_chunk(cams=(("CAM_A", 0.6), ("CAM_B", 0.9)))
    run(ComponentConfig(), bag_id=BAG)

    labels = lifted_labels.read_sweep(BAG, CHUNK, SWEEP)
    # Too little evidence to merge (3 shared points < min_shared_points): the
    # higher-scoring camera's masklet names the points.
    assert labels.instance_ids.tolist() == ["m_CAM_B"] * 3
    assert labels.confidences.tolist() == [1.0] * 3
    assert labels.n_supporting_cameras.tolist() == [2] * 3
    assert labels.n_disagreeing_cameras.tolist() == [0] * 3
    (stats,) = lifted_labels.read_stats(BAG, CHUNK)
    assert stats.n_cameras_used == 2 and stats.n_points_disagreement == 0


def test_one_objects_masklets_merge_into_one_instance():
    _stage_chunk(cams=(("CAM_B", 0.9), ("CAM_A", 0.6)))
    cfg = ComponentConfig()
    cfg.instances.min_shared_points = 3
    run(cfg, bag_id=BAG)

    labels = lifted_labels.read_sweep(BAG, CHUNK, SWEEP)
    assert labels.instance_ids.tolist() == ["m_CAM_A"] * 3  # the smaller id
    (inst,) = lifted_labels.read_instances(BAG, CHUNK)
    assert inst.instance_id == "m_CAM_A" and inst.cls == "vehicle"
    assert inst.masklet_ids == ("m_CAM_A", "m_CAM_B")
    assert inst.cam_ids == ("CAM_A", "CAM_B")
    assert (inst.n_points_labeled, inst.n_sweeps) == (3, 1)


def test_a_frame_without_depth_is_skipped():
    _stage_chunk(depth_m=None)
    run(ComponentConfig(), bag_id=BAG)
    (stats,) = lifted_labels.read_stats(BAG, CHUNK)
    assert stats.n_cameras_used == 0 and stats.n_points_labeled == 0


def test_a_missing_world_sweep_stops_the_run():
    _stage_chunk()
    os.remove(local_path(lidar_world_path(BAG, CHUNK, SWEEP)))
    with pytest.raises(FileNotFoundError):
        run(ComponentConfig(), bag_id=BAG)
    assert not lifted_labels.exists(BAG, CHUNK)


def test_each_chunk_gets_a_manifest_of_its_inputs_and_outputs():
    _stage_chunk()
    run(ComponentConfig(), bag_id=BAG)
    path = os.path.join(
        os.path.dirname(local_path(frame_index_path(BAG, CHUNK))),
        "manifest_semantic_lifting.json",
    )
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    assert manifest["component"] == "semantic_lifting"
    assert manifest["inputs"]["frame_index"]["sha256"]
    assert manifest["inputs"]["manifest_perception_2d"]["present"] is False
    assert set(manifest["outputs"]) == {
        "lifted_labels",
        "lifted_instances",
        "lifted_stats",
    }


def test_complete_chunks_are_skipped_unless_forced(monkeypatch):
    _stage_chunk()
    run(ComponentConfig(), bag_id=BAG)
    import wato_semantic_lifting.pipeline as pipeline

    calls = []
    monkeypatch.setattr(pipeline, "_process_chunk", lambda *a: calls.append(a))
    run(ComponentConfig(), bag_id=BAG)
    assert calls == []
    run(ComponentConfig(), bag_id=BAG, force=True)
    assert len(calls) == 1
