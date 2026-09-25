"""The depth pass projects LiDAR into an image with the ego pose at the image's
own timestamp, not the LiDAR sweep's pose."""

from __future__ import annotations

from collections import deque

import numpy as np

from wato_common.geometry import PoseSample, invert_se3
from wato_common.pose_lookup import PoseLookup, interval_drop_reasons
from wato_perception_2d import pipeline
from wato_perception_2d.config import ComponentConfig
from wato_perception_2d.io import CalibrationInfo, CameraFrameInfo

MS = 1_000_000


def _poses() -> PoseLookup:
    # Ego moving +x at 10 m/s, samples every 100 ms, then a 1 s dropout.
    samples = [
        PoseSample(
            t * MS, np.array([t / 100.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0])
        )
        for t in (0, 100, 200, 1_200)
    ]
    return PoseLookup(
        samples,
        interval_drop_reasons(samples, max_bracket_ns=250 * MS, max_speed_mps=30.0),
    )


def _frame(camera_ms: int | None) -> CameraFrameInfo:
    return CameraFrameInfo(
        frame_id="f0",
        bag_id="bag",
        chunk_id="0000",
        sweep_id=0,
        cam_id="cam_front",
        image_path="/img/0.jpg",
        camera_seq=0,
        camera_timestamp_ns=None if camera_ms is None else camera_ms * MS,
        valid_camera=True,
    )


def _run(monkeypatch, frame: CameraFrameInfo) -> tuple[list[np.ndarray], np.ndarray]:
    seen: list[np.ndarray] = []
    monkeypatch.setattr(
        pipeline, "load_static_lidar_points", lambda *a: np.array([[5.0, 0.0, 0.0]])
    )

    def fake_anchor_pairs(points, rel_depth, K, cam_T_world, size, **kw):
        seen.append(cam_T_world)
        return np.zeros(0), np.zeros(0)

    monkeypatch.setattr(pipeline, "build_anchor_pairs", fake_anchor_pairs)
    monkeypatch.setattr(
        pipeline, "ransac_affine_fit", lambda *a, **k: {"fit_status": 2}
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
        CalibrationInfo(K=np.eye(3), ego_T_cam=ego_T_cam),
        _poses(),
        deque(),
    )
    return seen, ego_T_cam


def test_projection_uses_the_pose_at_camera_time(monkeypatch):
    seen, ego_T_cam = _run(monkeypatch, _frame(camera_ms=140))
    world_T_ego = np.eye(4)
    world_T_ego[0, 3] = 1.4  # 10 m/s × 140 ms
    np.testing.assert_allclose(
        seen[0], invert_se3(ego_T_cam) @ invert_se3(world_T_ego), atol=1e-9
    )


def test_camera_time_in_an_untrusted_stretch_gets_no_anchors(monkeypatch):
    seen, _ = _run(monkeypatch, _frame(camera_ms=600))  # inside the 1 s dropout
    assert seen == []


def test_frame_without_camera_time_gets_no_anchors(monkeypatch):
    seen, _ = _run(monkeypatch, _frame(camera_ms=None))
    assert seen == []
