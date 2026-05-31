"""Smoke tests for the perception_2d pipeline.

Contract (fail-loud): when a required ML model is unavailable, the pipeline
RAISES rather than emitting degraded placeholder output.  Orchestration logic
(skip/force, parquet writing) is exercised via the empty-frame path, which
writes valid empty parquets without touching any model.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image as PILImage

from wato_perception_2d.config import ComponentConfig
from wato_perception_2d.pipeline import run


# ---------------------------------------------------------------------------
# Fixtures: minimal bag + chunk artifact tree in tmp_path
# ---------------------------------------------------------------------------


def _make_bag(tmp_path, monkeypatch, *, valid_camera: bool):
    """Create the minimum artifact tree for one bag / one chunk / one camera.

    valid_camera=False makes load_frame_index drop the frame, so the pipeline
    takes the model-free empty-output path.
    """
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")

    bag_id = "smoke_bag"
    chunk_id = "chunk_0000"

    bag_root = tmp_path / "raw" / bag_id
    chunks_dir = bag_root / "chunks" / chunk_id
    frames_dir = chunks_dir / "frames" / "cam_front"
    frames_dir.mkdir(parents=True, exist_ok=True)
    (chunks_dir / "lidar").mkdir(parents=True, exist_ok=True)

    calib = {
        "cameras": {
            "cam_front": {
                "K": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
                "ego_T_cam": np.eye(4).tolist(),
            }
        }
    }
    (bag_root / "calibration.json").write_text(json.dumps(calib))

    chunks_tbl = pa.table({"chunk_id": [chunk_id], "bag_id": [bag_id]})
    pq.write_table(chunks_tbl, str(bag_root / "chunks" / "index.parquet"))

    img_path = str(frames_dir / "000000.jpg")
    PILImage.fromarray(np.zeros((480, 640, 3), dtype=np.uint8)).save(img_path)

    pose_flat = np.eye(4, dtype=np.float64).ravel().tolist()
    frame_tbl = pa.table(
        {
            "frame_id": ["f0"],
            "bag_id": [bag_id],
            "chunk_id": [chunk_id],
            "sweep_id": [0],
            "cam_id": ["cam_front"],
            "image_path": [img_path],
            "camera_seq": [0],
            "world_T_ego_flat": pa.array([pose_flat], type=pa.list_(pa.float64())),
            "valid_camera": [valid_camera],
            "valid_pose": [True],
        }
    )
    pq.write_table(frame_tbl, str(chunks_dir / "frame_index.parquet"))

    lidar_tbl = pa.table(
        {
            "sweep_id": pa.array([], type=pa.int64()),
            "valid": pa.array([], type=pa.bool_()),
            "world_path": pa.array([], type=pa.string()),
            "dynamic_mask_path": pa.array([], type=pa.string()),
        }
    )
    pq.write_table(lidar_tbl, str(chunks_dir / "lidar_proc_index.parquet"))

    return bag_id, chunk_id, tmp_path


@pytest.fixture()
def live_bag(tmp_path, monkeypatch):
    """A bag with a real, valid frame — running it needs the ML models."""
    return _make_bag(tmp_path, monkeypatch, valid_camera=True)


@pytest.fixture()
def empty_bag(tmp_path, monkeypatch):
    """A bag whose only frame is invalid → model-free empty-output path."""
    return _make_bag(tmp_path, monkeypatch, valid_camera=False)


# ---------------------------------------------------------------------------
# Fail-loud contract
# ---------------------------------------------------------------------------


def test_pipeline_raises_when_models_unavailable(live_bag):
    """A valid frame + no SAM 3.1 installed → loud RuntimeError."""
    bag_id, _, _ = live_bag
    cfg = ComponentConfig()  # defaults: discovery.backend=fixed, depth enabled

    with pytest.raises(RuntimeError):
        run(cfg, bag_id=bag_id)


def test_fixed_mode_raises_without_sam3(live_bag):
    """Fixed-vocab mode skips Florence-2 but still needs SAM3 → loud failure."""
    bag_id, _, _ = live_bag
    cfg = ComponentConfig()
    cfg.discovery.backend = "fixed"
    cfg.discovery.fixed_classes = ["car"]
    cfg.depth.enabled = False  # isolate the failure to segmentation

    with pytest.raises(RuntimeError):
        run(cfg, bag_id=bag_id)


# ---------------------------------------------------------------------------
# Orchestration (model-free empty-output path)
# ---------------------------------------------------------------------------


def test_pipeline_writes_valid_parquets(empty_bag):
    """Empty-frame chunk writes non-corrupt (empty) parquets, no models needed."""
    from wato_common.artifact_store import (
        detections_2d_path,
        local_path,
        tracklets_2d_path,
    )

    bag_id, chunk_id, _ = empty_bag
    run(ComponentConfig(), bag_id=bag_id)

    det_path = local_path(detections_2d_path(bag_id, chunk_id))
    trk_path = local_path(tracklets_2d_path(bag_id, chunk_id))
    assert os.path.exists(det_path)
    assert os.path.exists(trk_path)
    assert pq.read_table(det_path).schema is not None
    assert pq.read_table(trk_path).schema is not None


def test_pipeline_force_reruns_chunk(empty_bag):
    """force=True rewrites the output even if it already exists."""
    from wato_common.artifact_store import detections_2d_path, local_path

    bag_id, chunk_id, _ = empty_bag
    run(ComponentConfig(), bag_id=bag_id)
    mtime1 = os.path.getmtime(local_path(detections_2d_path(bag_id, chunk_id)))

    import time

    time.sleep(0.05)
    run(ComponentConfig(), bag_id=bag_id, force=True)
    mtime2 = os.path.getmtime(local_path(detections_2d_path(bag_id, chunk_id)))

    assert mtime2 > mtime1


def test_pipeline_skips_complete_chunk(empty_bag):
    """Without force=True, an already-complete chunk is skipped."""
    from wato_common.artifact_store import detections_2d_path, local_path

    bag_id, chunk_id, _ = empty_bag
    run(ComponentConfig(), bag_id=bag_id)
    mtime1 = os.path.getmtime(local_path(detections_2d_path(bag_id, chunk_id)))

    import time

    time.sleep(0.05)
    run(ComponentConfig(), bag_id=bag_id)
    mtime2 = os.path.getmtime(local_path(detections_2d_path(bag_id, chunk_id)))

    assert mtime2 == mtime1
