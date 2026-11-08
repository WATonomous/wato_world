"""Smoke tests for the perception_2d pipeline.

Contract (fail-loud): when a required ML model is unavailable, the pipeline
RAISES rather than emitting degraded placeholder output.  Orchestration logic
(skip/force, parquet writing) is exercised via the empty-frame path, which
writes valid empty parquets without touching any model.
"""

from __future__ import annotations

import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image as PILImage

from wato_common.artifact_store import poses_path
from wato_common.artifacts import calibration, frame_index
from wato_common.artifacts.calibration import Calibration, CameraCalibration
from wato_common.io.parquet_io import write_table
from wato_common.schemas import POSES_SCHEMA, FrameIndexRow
from wato_perception_2d.config import ComponentConfig
from wato_perception_2d.pipeline import run


# ---------------------------------------------------------------------------
# Fixtures: minimal bag + chunk artifact tree in tmp_path
# ---------------------------------------------------------------------------


def _make_bag(tmp_path, monkeypatch, *, valid_camera: bool):
    """Create the minimum artifact tree for one bag / one chunk / one camera.

    valid_camera=False makes FrameIndex drop the frame, so the pipeline takes
    the model-free empty-output path.
    """
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")

    bag_id = "smoke_bag"
    chunk_id = "chunk_0000"

    bag_root = tmp_path / "raw" / bag_id
    chunks_dir = bag_root / "chunks" / chunk_id
    frames_dir = chunks_dir / "frames" / "cam_front"
    frames_dir.mkdir(parents=True, exist_ok=True)
    (chunks_dir / "lidar").mkdir(parents=True, exist_ok=True)

    calibration.write(
        bag_id,
        Calibration(
            cameras={
                "cam_front": CameraCalibration(
                    cam_id="cam_front",
                    K=np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]]),
                    ego_T_cam=np.eye(4),
                )
            },
            lidars={},
        ),
    )

    chunks_tbl = pa.table({"chunk_id": [chunk_id], "bag_id": [bag_id]})
    pq.write_table(chunks_tbl, str(bag_root / "chunks" / "index.parquet"))

    img_path = str(frames_dir / "000000.jpg")
    PILImage.fromarray(np.zeros((480, 640, 3), dtype=np.uint8)).save(img_path)

    write_table(
        [
            {
                "bag_id": bag_id,
                "chunk_id": chunk_id,
                "timestamp_ns": 0,
                **dict.fromkeys(("x", "y", "z", "qx", "qy", "qz"), 0.0),
                "qw": 1.0,
                "world_T_ego_flat": np.eye(4).ravel().tolist(),
                "source": "/odom",
                "valid": True,
                "interval_drop_reason": None,
            }
        ],
        POSES_SCHEMA,
        poses_path(bag_id, chunk_id),
    )
    frame_index.write(
        bag_id,
        chunk_id,
        [
            FrameIndexRow(
                frame_id="f0",
                bag_id=bag_id,
                chunk_id=chunk_id,
                sweep_id=0,
                lidar_id="lidar_top",
                lidar_path="",
                reference_timestamp_ns=0,
                cam_id="cam_front",
                image_path=img_path,
                camera_seq=0,
                camera_timestamp_ns=0,
                valid_camera=valid_camera,
                valid_pose=True,
            )
        ],
    )

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


def test_fixed_mode_raises_without_models(live_bag, tmp_path):
    """Fixed-vocab mode skips Florence-2 but still needs SAM2 → loud failure."""
    bag_id, _, _ = live_bag
    cfg = ComponentConfig()
    cfg.discovery.backend = "fixed"
    cfg.discovery.fixed_classes = ["car"]
    cfg.depth.enabled = False  # isolate the failure to tracking
    # Missing even where data/models is mounted (the dev image).
    cfg.segmentation.checkpoint = str(tmp_path / "missing_sam2.pt")

    with pytest.raises(RuntimeError):
        run(cfg, bag_id=bag_id)


def test_missing_tracking_weights_fail_before_the_depth_pass(
    live_bag, tmp_path, monkeypatch
):
    """The tracking pass loads its weights after a chunk's depth pass, so a
    missing YOLO checkpoint must be caught before DA-V2 is even built."""
    from wato_perception_2d import pipeline

    class _DepthMustNotRun:
        def __init__(self, *a, **k):
            raise AssertionError("depth pass started before the weights check")

    monkeypatch.setattr(pipeline, "sam2_importable", lambda: True)
    monkeypatch.setattr(pipeline, "yolo_importable", lambda: True)
    monkeypatch.setattr(pipeline, "DepthAnythingV2", _DepthMustNotRun)
    sam2_ckpt = tmp_path / "sam2.pt"
    sam2_ckpt.write_bytes(b"")
    bag_id, _, _ = live_bag
    cfg = ComponentConfig()
    cfg.segmentation.checkpoint = str(sam2_ckpt)
    cfg.detection.yolo.checkpoint = str(tmp_path / "missing_yolo26x.pt")

    with pytest.raises(RuntimeError, match="missing_yolo26x.pt"):
        run(cfg, bag_id=bag_id)


# ---------------------------------------------------------------------------
# Orchestration (model-free empty-output path)
# ---------------------------------------------------------------------------


def test_cuda_cleanup_is_checked_before_depth(live_bag, monkeypatch):
    from wato_perception_2d import pipeline

    monkeypatch.setattr(pipeline, "sam2_importable", lambda: True)
    monkeypatch.setattr(pipeline, "yolo_importable", lambda: True)
    monkeypatch.setattr(pipeline, "_check_tracking_weights", lambda *args: None)

    def broken_extension(device):
        raise RuntimeError("extension kernel unavailable")

    def unexpected_depth(*args, **kwargs):
        raise AssertionError("depth started before CUDA cleanup check")

    monkeypatch.setattr(pipeline, "check_sam2_cuda_extension", broken_extension)
    monkeypatch.setattr(pipeline, "DepthAnythingV2", unexpected_depth)
    bag_id, _, _ = live_bag
    with pytest.raises(RuntimeError, match="extension kernel unavailable"):
        run(ComponentConfig(), bag_id=bag_id)


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


def test_manifest_records_effective_config_including_runtime_override(empty_bag):
    import json

    bag_id, chunk_id, root = empty_bag
    cfg = ComponentConfig()
    cfg.segmentation.refinement.enabled = True
    run(cfg, bag_id=bag_id)
    path = root / "raw" / bag_id / "chunks" / chunk_id / "manifest_perception_2d.json"
    manifest = json.loads(path.read_text())
    assert manifest["effective_config"]["segmentation"]["refinement"]["enabled"] is True
    from wato_perception_2d.pipeline import _default_device

    from wato_perception_2d.models._sam2_runtime import installed_sam2_commit

    assert manifest["mask_propagation"] == {
        "backend": "dam4sam",
        "sam2_commit": installed_sam2_commit(),
        "dam4sam_commit": "67d5dfe86d8ae7b557193231c6058a46f5c0c490",
        "adapter_version": 2,
        "mask_semantics": "independent",
        "cuda_hole_filling": _default_device() == "cuda",
    }


def test_resume_rejects_mixed_mask_policies(empty_bag):
    from wato_perception_2d.pipeline import _save_cam_partial, _load_cam_partial

    bag_id, chunk_id, _ = empty_bag
    cfg = ComponentConfig()
    _save_cam_partial(bag_id, chunk_id, "cam_front", [], cfg, "cuda")
    assert _load_cam_partial(bag_id, chunk_id, "cam_front", cfg, "cuda") == []
    cfg.segmentation.refinement.enabled = True
    with pytest.raises(RuntimeError, match="avoid mixing mask policies"):
        _load_cam_partial(bag_id, chunk_id, "cam_front", cfg, "cuda")
    cfg.segmentation.refinement.enabled = False
    cfg.segmentation.backend = "sam2"
    with pytest.raises(RuntimeError, match="avoid mixing mask policies"):
        _load_cam_partial(bag_id, chunk_id, "cam_front", cfg, "cuda")


def test_resume_rejects_masks_made_without_cuda_hole_filling(empty_bag):
    """SAM2 skips hole filling on CPU, so a CPU camera's masks differ from a
    CUDA run's and must not be resumed into one."""
    from wato_perception_2d.pipeline import _save_cam_partial, _load_cam_partial

    bag_id, chunk_id, _ = empty_bag
    cfg = ComponentConfig()
    _save_cam_partial(bag_id, chunk_id, "cam_front", [], cfg, "cpu")
    assert _load_cam_partial(bag_id, chunk_id, "cam_front", cfg, "cpu") == []
    with pytest.raises(RuntimeError, match="avoid mixing mask policies"):
        _load_cam_partial(bag_id, chunk_id, "cam_front", cfg, "cuda")


def test_resume_rejects_cache_from_before_required_cuda_cleanup(empty_bag):
    import pickle
    from pathlib import Path

    from wato_perception_2d.pipeline import _cam_partial_path, _load_cam_partial

    bag_id, chunk_id, _ = empty_bag
    cfg = ComponentConfig()
    path = Path(_cam_partial_path(bag_id, chunk_id, "cam_front"))
    path.parent.mkdir(parents=True)
    path.write_bytes(
        pickle.dumps({"config": cfg.model_dump(mode="json"), "masklets": []})
    )
    with pytest.raises(RuntimeError, match="avoid mixing mask policies"):
        _load_cam_partial(bag_id, chunk_id, "cam_front", cfg, "cuda")


@pytest.mark.parametrize("semantics", [None, "owned"])
def test_resume_rejects_old_owned_masks_even_with_matching_config(empty_bag, semantics):
    """A cached mask cannot acquire independent semantics by being re-serialized."""
    import pickle
    from pathlib import Path

    from wato_perception_2d.pipeline import (
        _cam_partial_path,
        _load_cam_partial,
        _save_cam_partial,
    )

    bag_id, chunk_id, _ = empty_bag
    cfg = ComponentConfig()
    _save_cam_partial(bag_id, chunk_id, "cam_front", [], cfg, "cpu")
    path = Path(_cam_partial_path(bag_id, chunk_id, "cam_front"))
    cached = pickle.loads(path.read_bytes())
    if semantics is None:
        del cached["mask_propagation"]["mask_semantics"]
    else:
        cached["mask_propagation"]["mask_semantics"] = semantics
    path.write_bytes(pickle.dumps(cached))
    with pytest.raises(RuntimeError, match="rerun with --force"):
        _load_cam_partial(bag_id, chunk_id, "cam_front", cfg, "cpu")


def test_complete_chunk_is_skipped_unless_forced(empty_bag):
    import time

    from wato_common.artifact_store import detections_2d_path, local_path

    bag_id, chunk_id, _ = empty_bag
    path = local_path(detections_2d_path(bag_id, chunk_id))
    run(ComponentConfig(), bag_id=bag_id)
    mtime = os.path.getmtime(path)

    time.sleep(0.05)
    run(ComponentConfig(), bag_id=bag_id)
    assert os.path.getmtime(path) == mtime

    run(ComponentConfig(), bag_id=bag_id, force=True)
    assert os.path.getmtime(path) > mtime


def test_force_clears_previous_masks_and_depth(empty_bag):
    """A forced rerun doesn't leave the previous run's masks / depth behind.

    Mask dirs are named by masklet_id, new each run, so stale ones would sit
    beside the new run's; a non-forced rerun (resume) must keep them.
    """
    from wato_common.artifact_store import depth_2d_dir, local_path, masks_2d_dir
    from wato_perception_2d.pipeline import _cam_partial_path

    bag_id, chunk_id, _ = empty_bag
    stale_mask = os.path.join(
        local_path(masks_2d_dir(bag_id, chunk_id)), "old", "000001.png"
    )
    stale_depth = os.path.join(
        local_path(depth_2d_dir(bag_id, chunk_id)), "CAM", "000001.npz"
    )
    stale_partial = _cam_partial_path(bag_id, chunk_id, "cam_front")
    for p in (stale_mask, stale_depth, stale_partial):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").close()

    run(ComponentConfig(), bag_id=bag_id)
    assert os.path.exists(stale_mask) and os.path.exists(stale_depth)

    run(ComponentConfig(), bag_id=bag_id, force=True)
    assert not os.path.exists(stale_mask)
    assert not os.path.exists(stale_depth)
    assert not os.path.exists(stale_partial)
