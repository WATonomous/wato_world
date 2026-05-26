"""Smoke tests for the perception_2d v2 pipeline.

Verifies that the pipeline runs gracefully when all ML models are unavailable
(no GPU, no transformers/sam3/depth_anything_v2 installed).  Empty-but-valid
parquets should be written; no exceptions should escape run().
"""

from __future__ import annotations

import json
import os

import numpy as np
import pyarrow.parquet as pq
import pytest
from PIL import Image as PILImage

from wato_perception_2d.config import ComponentConfig
from wato_perception_2d.pipeline import run


# ---------------------------------------------------------------------------
# Fixture: minimal bag + chunk artifact tree in tmp_path
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_bag(tmp_path, monkeypatch):
    """Create the minimum artifact tree for one bag / one chunk / one camera."""
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")

    bag_id = "smoke_bag"
    chunk_id = "chunk_0000"

    # Directories
    bag_root = tmp_path / "raw" / bag_id
    chunks_dir = bag_root / "chunks" / chunk_id
    chunks_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = chunks_dir / "frames" / "cam_front"
    frames_dir.mkdir(parents=True, exist_ok=True)
    lidar_dir = chunks_dir / "lidar"
    lidar_dir.mkdir(parents=True, exist_ok=True)

    # --- calibration.json ---
    calib = {
        "cameras": {
            "cam_front": {
                "K": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
                "ego_T_cam": np.eye(4).tolist(),
            }
        }
    }
    (bag_root / "calibration.json").write_text(json.dumps(calib))

    # --- chunks/index.parquet ---
    import pyarrow as pa
    import pyarrow.parquet as pq_w

    chunks_tbl = pa.table({"chunk_id": [chunk_id], "bag_id": [bag_id]})
    pq_w.write_table(chunks_tbl, str(bag_root / "chunks" / "index.parquet"))

    # --- One synthetic image ---
    img_path = str(frames_dir / "000000.jpg")
    PILImage.fromarray(
        np.zeros((480, 640, 3), dtype=np.uint8)
    ).save(img_path)

    # --- frame_index.parquet ---
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
            "valid_camera": [True],
            "valid_pose": [True],
        }
    )
    pq_w.write_table(frame_tbl, str(chunks_dir / "frame_index.parquet"))

    # --- lidar_proc_index.parquet (empty — no LiDAR points) ---
    lidar_tbl = pa.table(
        {
            "sweep_id": pa.array([], type=pa.int64()),
            "valid": pa.array([], type=pa.bool_()),
            "world_path": pa.array([], type=pa.string()),
            "dynamic_mask_path": pa.array([], type=pa.string()),
        }
    )
    pq_w.write_table(lidar_tbl, str(chunks_dir / "lidar_proc_index.parquet"))

    return bag_id, chunk_id, tmp_path


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_pipeline_runs_without_ml_models(fake_bag):
    """Pipeline completes without error when all ML models are unavailable."""
    bag_id, chunk_id, tmp_path = fake_bag
    cfg = ComponentConfig()  # all defaults

    # run() should not raise even when Florence-2, SAM3, DA-V2 are missing.
    run(cfg, bag_id=bag_id)


def test_pipeline_writes_valid_parquets(fake_bag):
    """Pipeline writes non-corrupt parquets even with zero detections."""
    from wato_common.artifact_store import detections_2d_path, local_path, tracklets_2d_path

    bag_id, chunk_id, tmp_path = fake_bag
    cfg = ComponentConfig()

    run(cfg, bag_id=bag_id)

    det_path = local_path(detections_2d_path(bag_id, chunk_id))
    trk_path = local_path(tracklets_2d_path(bag_id, chunk_id))

    assert os.path.exists(det_path), "detections_2d.parquet not written"
    assert os.path.exists(trk_path), "tracklets_2d.parquet not written"

    # Must be readable as parquet.
    det_tbl = pq.read_table(det_path)
    trk_tbl = pq.read_table(trk_path)
    assert det_tbl.schema is not None
    assert trk_tbl.schema is not None


def test_pipeline_force_reruns_chunk(fake_bag):
    """force=True causes the chunk to be re-processed even if outputs exist."""
    from wato_common.artifact_store import detections_2d_path, local_path

    bag_id, chunk_id, tmp_path = fake_bag
    cfg = ComponentConfig()

    run(cfg, bag_id=bag_id)
    mtime1 = os.path.getmtime(local_path(detections_2d_path(bag_id, chunk_id)))

    import time
    time.sleep(0.05)

    run(cfg, bag_id=bag_id, force=True)
    mtime2 = os.path.getmtime(local_path(detections_2d_path(bag_id, chunk_id)))

    assert mtime2 > mtime1, "force=True should rewrite the output"


def test_pipeline_skips_complete_chunk(fake_bag):
    """Without force=True, already-complete chunks are skipped."""
    from wato_common.artifact_store import detections_2d_path, local_path

    bag_id, chunk_id, tmp_path = fake_bag
    cfg = ComponentConfig()

    run(cfg, bag_id=bag_id)
    mtime1 = os.path.getmtime(local_path(detections_2d_path(bag_id, chunk_id)))

    import time
    time.sleep(0.05)

    run(cfg, bag_id=bag_id)  # no force
    mtime2 = os.path.getmtime(local_path(detections_2d_path(bag_id, chunk_id)))

    assert mtime2 == mtime1, "chunk should be skipped when already complete"
