"""Tests for the batched depth pass (depth.batch_size).

These exercise the pipeline's PASS-1 batching loop without torch / DA-V2: the
depth model is replaced by a fake that records how images are grouped and
echoes a per-image marker, so we can assert (a) frames are grouped into batches
of depth.batch_size, (b) every frame still gets aligned exactly once, and (c)
each frame is aligned with ITS OWN relative-depth map, in camera_seq order.
"""

from __future__ import annotations

import numpy as np

from wato_perception_2d import pipeline
from wato_perception_2d.config import ComponentConfig
from wato_perception_2d.io import CalibrationInfo, CameraFrameInfo


def _frame(seq: int) -> CameraFrameInfo:
    return CameraFrameInfo(
        frame_id=f"f{seq}",
        bag_id="bag",
        chunk_id="chunk_0000",
        sweep_id=seq,
        cam_id="cam_front",
        image_path=f"/img/{seq}.jpg",
        camera_seq=seq,
        world_T_ego_flat=None,
        valid_camera=True,
        valid_pose=True,
    )


class _FakeDepth:
    """Stand-in for DepthAnythingV2 that echoes each image's [0,0,0] marker."""

    instances: list["_FakeDepth"] = []

    def __init__(self, **_kw):
        self.batch_sizes: list[int] = []
        _FakeDepth.instances.append(self)

    def infer_batch(self, images, *, batch_size=1, with_confidence=False):
        self.batch_sizes.append(len(images))
        # Echo the marker baked into _load_image so the caller can verify the
        # depth map is matched to the right frame.
        depths = [np.full((4, 5), float(im[0, 0, 0]), dtype=np.float32) for im in images]
        return depths, [None] * len(images)

    def unload(self):
        pass


def _run(monkeypatch, *, n_frames: int, batch_size: int, skip_seq: set[int] | None = None):
    skip_seq = skip_seq or set()
    _FakeDepth.instances.clear()
    aligned: list[tuple[int, float]] = []

    monkeypatch.setattr(pipeline, "DepthAnythingV2", _FakeDepth)
    # Bake camera_seq into the image so infer_batch can echo it back.
    monkeypatch.setattr(
        pipeline,
        "_load_image",
        lambda path: (
            None
            if (seq := int(path.split("/")[-1].split(".")[0])) in skip_seq
            else np.full((4, 5, 3), seq, dtype=np.uint8)
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_align_and_write_depth",
        lambda cfg, bag, chunk, cam, frame, rel_depth, calib, fb: aligned.append(
            (frame.camera_seq, float(rel_depth[0, 0]))
        ),
    )

    cfg = ComponentConfig()
    cfg.depth.batch_size = batch_size
    calib = {"cam_front": CalibrationInfo(K=np.eye(3), ego_T_cam=np.eye(4))}
    frames_by_cam = {"cam_front": [_frame(i) for i in range(n_frames)]}

    pipeline._run_depth_pass(cfg, "bag", "chunk_0000", frames_by_cam, calib)
    return aligned, _FakeDepth.instances[0]


def test_frames_grouped_into_batches(monkeypatch):
    aligned, model = _run(monkeypatch, n_frames=5, batch_size=2)
    # 5 frames at batch_size 2 → batches of 2, 2, 1.
    assert model.batch_sizes == [2, 2, 1]
    # Every frame aligned once, in order, each with its own depth map.
    assert aligned == [(i, float(i)) for i in range(5)]


def test_batch_size_one_is_per_frame(monkeypatch):
    aligned, model = _run(monkeypatch, n_frames=3, batch_size=1)
    assert model.batch_sizes == [1, 1, 1]
    assert aligned == [(0, 0.0), (1, 1.0), (2, 2.0)]


def test_unreadable_frame_dropped_from_batch(monkeypatch):
    # Frame 1 fails to load → excluded from inference + alignment, others intact.
    aligned, model = _run(monkeypatch, n_frames=4, batch_size=2, skip_seq={1})
    # First batch had only frame 0 (1 image); second batch frames 2,3 (2 images).
    assert model.batch_sizes == [1, 2]
    assert aligned == [(0, 0.0), (2, 2.0), (3, 3.0)]


def test_default_batch_size_is_one():
    assert ComponentConfig().depth.batch_size == 1
