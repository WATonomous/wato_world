"""The depth pass projects LiDAR into an image with the frame's own pose (looked
up at the image's time by wato_common.artifacts.frame_index), anchoring on
every sweep the image is paired with."""

from __future__ import annotations

from collections import deque

import numpy as np

from wato_common.artifacts.calibration import CameraCalibration
from wato_common.artifacts.frame_index import CameraFrame
from wato_common.geometry import invert_se3
from wato_perception_2d import pipeline
from wato_perception_2d.config import ComponentConfig


def _frame(world_T_ego: np.ndarray | None, sweep_ids=(0,)) -> CameraFrame:
    return CameraFrame(
        cam_id="cam_front",
        camera_seq=0,
        timestamp_ns=0,
        image_path="/img/0.jpg",
        sweep_ids=tuple(sweep_ids),
        world_T_ego=world_T_ego,
    )


def _run(monkeypatch, frame: CameraFrame, per_sweep=None):
    """(points, cam_T_world) passed to each build_anchor_pairs call."""
    seen: list[tuple[np.ndarray, np.ndarray]] = []
    monkeypatch.setattr(
        pipeline,
        "load_static_lidar_points",
        lambda b, c, sid: (
            np.array([[5.0, 0.0, 0.0]]) if per_sweep is None else per_sweep[sid]
        ),
    )

    def fake_anchor_pairs(points, disparity, K, cam_T_world, size, **kw):
        seen.append((points, cam_T_world))
        return np.zeros(0), np.zeros(0)

    monkeypatch.setattr(pipeline, "build_anchor_pairs", fake_anchor_pairs)
    monkeypatch.setattr(
        pipeline, "ransac_disparity_fit", lambda *a, **k: {"fit_status": 2}
    )
    ego_T_cam = np.eye(4)
    ego_T_cam[2, 3] = 1.5
    pipeline._align_and_write_depth(
        ComponentConfig(),
        "bag",
        "0000",
        "cam_front",
        frame,
        np.ones((4, 5), dtype=np.float32),
        CameraCalibration(cam_id="cam_front", K=np.eye(3), ego_T_cam=ego_T_cam),
        deque(),
    )
    return seen, ego_T_cam


def test_projection_uses_the_frames_pose(monkeypatch):
    world_T_ego = np.eye(4)
    world_T_ego[0, 3] = 1.4
    seen, ego_T_cam = _run(monkeypatch, _frame(world_T_ego))
    ((_, cam_T_world),) = seen
    np.testing.assert_allclose(
        cam_T_world, invert_se3(ego_T_cam) @ invert_se3(world_T_ego), atol=1e-9
    )


def test_frame_without_a_trusted_pose_gets_no_anchors(monkeypatch):
    seen, _ = _run(monkeypatch, _frame(None))
    assert seen == []


def test_depth_anchors_on_every_paired_sweep(monkeypatch):
    per_sweep = {
        470: np.array([[1.0, 0.0, 0.0]]),
        471: None,  # a sweep with no static points contributes nothing
        472: np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
    }
    seen, _ = _run(monkeypatch, _frame(np.eye(4), sweep_ids=(470, 471, 472)), per_sweep)
    ((points, _),) = seen
    assert points[:, 0].tolist() == [1.0, 2.0, 3.0]
