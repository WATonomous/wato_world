"""Integration tests for the perception_2d pipeline orchestrator.

All tests redirect ARTIFACT_ROOT_URI to a tmp_path so no real artifacts
are required. Model loading is patched out — only the pipeline logic is
exercised.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image as PILImage

from wato_perception_2d.detector import Detection
from wato_perception_2d.segmenter import SAM2Segmenter, SegmentedDetection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_fake_image(path: str, h: int = 64, w: int = 64) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    PILImage.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(path)


def _fake_calibration(cam_ids: list[str]) -> dict:
    K = [[500.0, 0.0, 32.0], [0.0, 500.0, 32.0], [0.0, 0.0, 1.0]]
    ego_T_cam = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    return {"cameras": {cam: {"K": K, "ego_T_cam": ego_T_cam} for cam in cam_ids}}


def _fake_frame_index_row(
    bag_id: str,
    chunk_id: str,
    image_path: str,
    cam_id: str = "cam_front",
    sweep_id: int = 0,
) -> dict:
    return {
        "frame_id": "f0",
        "bag_id": bag_id,
        "chunk_id": chunk_id,
        "sweep_id": sweep_id,
        "cam_id": cam_id,
        "image_path": f"file://{image_path}",
        "camera_seq": 0,
        "world_T_ego_flat": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        "valid_camera": True,
        "valid_pose": True,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def artifact_root(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    root.mkdir()
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{root}")
    return root


@pytest.fixture()
def bag_id():
    return "test_bag"


@pytest.fixture()
def chunk_id():
    return "chunk_000"


# ---------------------------------------------------------------------------
# Tests for _chunk_complete
# ---------------------------------------------------------------------------


def test_chunk_complete_false_before_write(artifact_root, bag_id, chunk_id):
    from wato_perception_2d.pipeline import _chunk_complete

    assert not _chunk_complete(bag_id, chunk_id)


def test_chunk_complete_true_after_write(artifact_root, bag_id, chunk_id):
    from wato_common.artifact_store import detections_2d_path, local_path, tracklets_2d_path
    from wato_common.io.parquet_io import write_table
    from wato_common.schemas import MASKLET_SCHEMA
    from wato_perception_2d.pipeline import _chunk_complete

    write_table([], MASKLET_SCHEMA, detections_2d_path(bag_id, chunk_id))
    write_table([], MASKLET_SCHEMA, tracklets_2d_path(bag_id, chunk_id))
    assert _chunk_complete(bag_id, chunk_id)


# ---------------------------------------------------------------------------
# Tests for _process_chunk
# ---------------------------------------------------------------------------


def test_process_chunk_empty_frame_index_writes_empty_parquets(
    artifact_root, bag_id, chunk_id
):
    """Empty frame index → writes empty parquets without raising."""
    from wato_common.artifact_store import detections_2d_path, local_path
    from wato_perception_2d.config import ComponentConfig
    from wato_perception_2d.pipeline import _process_chunk

    cfg = ComponentConfig()
    detector = MagicMock()
    segmenter = MagicMock()

    with patch("wato_perception_2d.pipeline.load_frame_index", return_value=[]):
        _process_chunk(cfg, bag_id, chunk_id, detector, segmenter)

    assert os.path.exists(local_path(detections_2d_path(bag_id, chunk_id)))


def test_process_chunk_single_frame_writes_one_row(
    artifact_root, bag_id, chunk_id, tmp_path
):
    """Single frame with one detection → detections_2d.parquet has one row."""
    import pyarrow.parquet as pq

    from wato_common.artifact_store import (
        calibration_path,
        chunks_index_path,
        detections_2d_path,
        ensure_local_dir,
        local_path,
    )
    from wato_common.io.parquet_io import write_table
    from wato_common.schemas import CHUNK_SCHEMA, ChunkRow
    from wato_perception_2d.config import ComponentConfig
    from wato_perception_2d.io import CameraFrameInfo
    from wato_perception_2d.pipeline import _process_chunk

    # Write calibration.json
    calib_path = local_path(calibration_path(bag_id))
    os.makedirs(os.path.dirname(calib_path), exist_ok=True)
    with open(calib_path, "w") as fh:
        json.dump(_fake_calibration(["cam_front"]), fh)

    # Write a fake image
    img_dir = str(artifact_root / "raw" / bag_id / "images")
    img_path = os.path.join(img_dir, "frame_000.png")
    _write_fake_image(img_path)

    # Build a fake CameraFrameInfo
    frame = CameraFrameInfo(
        frame_id="f0",
        bag_id=bag_id,
        chunk_id=chunk_id,
        sweep_id=0,
        cam_id="cam_front",
        image_path=img_path,
        camera_seq=0,
        world_T_ego_flat=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        valid_camera=True,
        valid_pose=True,
    )

    # Fake detector returns one box
    fake_det = Detection(
        bbox_xyxy=np.array([5, 5, 30, 30], dtype=np.float32),
        class_name="car",
        score=0.9,
    )

    # Fake segmenter returns bbox-fill mask
    H, W = 64, 64
    mask = np.zeros((H, W), dtype=bool)
    mask[5:30, 5:30] = True
    fake_seg = SegmentedDetection(detection=fake_det, mask=mask)

    cfg = ComponentConfig()
    detector_mock = MagicMock()
    segmenter_mock = MagicMock()
    segmenter_mock.segment.return_value = [fake_seg]

    with (
        patch("wato_perception_2d.pipeline.load_frame_index", return_value=[frame]),
        patch("wato_perception_2d.pipeline.load_calibration", return_value={
            "cam_front": __import__(
                "wato_perception_2d.io", fromlist=["CalibrationInfo"]
            ).CalibrationInfo(
                K=np.array([[500, 0, 32], [0, 500, 32], [0, 0, 1]], dtype=np.float64),
                ego_T_cam=np.eye(4, dtype=np.float64),
            )
        }),
        patch("wato_perception_2d.pipeline.load_dynamic_lidar_points", return_value=None),
    ):
        detector_mock.detect.return_value = [fake_det]
        _process_chunk(cfg, bag_id, chunk_id, detector_mock, segmenter_mock)

    table = pq.read_table(local_path(detections_2d_path(bag_id, chunk_id)))
    assert len(table) == 1
    assert table["cls"][0].as_py() == "car"
