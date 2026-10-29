"""Both passes see each camera image once, not once per frame_index row."""

from __future__ import annotations

from collections import deque

import numpy as np

from wato_common.artifacts.calibration import CameraCalibration
from wato_common.geometry import PoseSample
from wato_common.pose_lookup import PoseLookup
from wato_perception_2d import pipeline
from wato_perception_2d.config import ComponentConfig
from wato_perception_2d.io import CameraFrameInfo


def _row(seq: int, sweep_id: int) -> CameraFrameInfo:
    return CameraFrameInfo(
        frame_id=f"f{sweep_id}",
        bag_id="bag",
        chunk_id="0000",
        sweep_id=sweep_id,
        cam_id="PANO_NE",
        image_path=f"/img/{seq:06d}.png",
        camera_seq=seq,
        camera_timestamp_ns=seq * 83_000_000,
        valid_camera=True,
    )


def test_rows_sharing_an_image_collapse_to_one_with_all_sweeps():
    # Measured shape on the 3-LiDAR rig: image 105 nearest to sweeps 470-474,
    # 106 to 475-480, 107 to 481-484.
    rows = (
        [_row(105, s) for s in range(470, 475)]
        + [_row(106, s) for s in range(475, 481)]
        + [_row(107, s) for s in range(481, 485)]
    )
    out = pipeline._one_frame_per_image(rows)
    assert [f.camera_seq for f in out] == [105, 106, 107]
    assert [f.sweep_id for f in out] == [470, 475, 481]  # first row of each
    assert out[0].sweep_ids == (470, 471, 472, 473, 474)
    assert out[1].sweep_ids == tuple(range(475, 481))


def test_a_skipped_image_stays_skipped():
    rows = [_row(1, 10), _row(1, 11), _row(3, 12)]
    out = pipeline._one_frame_per_image(rows)
    assert [(f.camera_seq, f.sweep_ids) for f in out] == [(1, (10, 11)), (3, (12,))]


def test_empty():
    assert pipeline._one_frame_per_image([]) == []


def test_depth_anchors_on_every_matched_sweep(monkeypatch):
    per_sweep = {
        470: np.array([[1.0, 0.0, 0.0]]),
        471: None,  # a sweep with no static points contributes nothing
        472: np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
    }
    monkeypatch.setattr(
        pipeline, "load_static_lidar_points", lambda b, c, sid: per_sweep[sid]
    )
    seen = []

    def fake_anchor_pairs(points, *a, **k):
        seen.append(points)
        return np.zeros(0), np.zeros(0)

    monkeypatch.setattr(pipeline, "build_anchor_pairs", fake_anchor_pairs)
    monkeypatch.setattr(
        pipeline, "ransac_disparity_fit", lambda *a, **k: {"fit_status": 2}
    )
    (frame,) = pipeline._one_frame_per_image([_row(0, s) for s in (470, 471, 472)])
    poses = PoseLookup(
        [PoseSample(t, np.zeros(3), np.array([0, 0, 0, 1.0])) for t in (0, 1)]
    )
    pipeline._align_and_write_depth(
        ComponentConfig(),
        "bag",
        "0000",
        "PANO_NE",
        frame,
        np.ones((4, 5), dtype=np.float32),
        CameraCalibration(cam_id="PANO_NE", K=np.eye(3), ego_T_cam=np.eye(4)),
        poses,
        deque(),
    )
    (points,) = seen
    assert points[:, 0].tolist() == [1.0, 2.0, 3.0]
