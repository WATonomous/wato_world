"""Tests for sam2_tracker — detect on keyframes, box-prompt SAM2, propagate, and
group per-object masks into Masklets — using a fake SAM2 video predictor and a
fake detector so no `sam2` / `transformers` install or GPU is required.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from wato_perception_2d.io import CameraFrameInfo
from wato_perception_2d.models.detector import Detection
from wato_perception_2d.models.sam2_tracker import track_camera


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakePredictor:
    """Minimal SAM2VideoPredictor stand-in.

    Stores the box of each added object and, on propagate, yields a +1/-1 logit
    map (positive inside the box) for every object whose add-frame ≤ the frame.
    init_state reads the window's frame count from the temp JPEG dir the tracker
    writes, so a per-window OOM can be simulated by frame count.
    """

    def __init__(self, H: int, W: int, oom_above: int | None = None) -> None:
        self.H, self.W = H, W
        self.oom_above = oom_above
        self._objs: dict[int, tuple[int, tuple]] = {}
        self._cur_n = 0

    def init_state(self, video_path, offload_video_to_cpu=False):
        self._cur_n = len([f for f in os.listdir(video_path) if f.endswith(".jpg")])
        self._objs = {}
        return {"n": self._cur_n}

    def reset_state(self, state):
        self._objs = {}

    def add_new_points_or_box(self, inference_state, frame_idx, obj_id, box):
        self._objs[int(obj_id)] = (int(frame_idx), tuple(float(v) for v in box))

    def _box_logit(self, box) -> np.ndarray:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        m = np.full((self.H, self.W), -1.0, dtype=np.float32)
        m[y1:y2, x1:x2] = 1.0
        return m[None]  # (1, H, W)

    def propagate_in_video(self, state, start_frame_idx=0, max_frame_num_to_track=None):
        if self.oom_above is not None and self._cur_n > self.oom_above:
            raise _OOM("CUDA out of memory")
        n = state["n"]
        stop = (
            n - 1
            if max_frame_num_to_track is None
            else min(n - 1, start_frame_idx + max_frame_num_to_track)
        )
        for f in range(start_frame_idx, stop + 1):
            present = [
                (oid, box)
                for oid, (af, box) in sorted(self._objs.items())
                if af <= f
            ]
            obj_ids = [oid for oid, _ in present]
            if present:
                logits = np.stack([self._box_logit(b) for _, b in present], axis=0)
            else:
                logits = np.zeros((0, 1, self.H, self.W), dtype=np.float32)
            yield f, obj_ids, logits


class _PerCallDetector:
    """Returns a scripted detection list per detect() call (in keyframe order)."""

    def __init__(self, per_call: list[list[Detection]]) -> None:
        self._per_call = per_call
        self.calls = 0

    def detect(self, image_rgb, concepts):
        out = self._per_call[self.calls] if self.calls < len(self._per_call) else []
        self.calls += 1
        return out


class _ConstDetector:
    """Returns the same detections on every call (used for windowed/OOM tests)."""

    def __init__(self, dets: list[Detection]) -> None:
        self._dets = dets

    def detect(self, image_rgb, concepts):
        return list(self._dets)


def _frames(n: int, cam: str = "cam_front") -> list[CameraFrameInfo]:
    return [
        CameraFrameInfo(
            frame_id=f"f{i}",
            bag_id="bag0",
            chunk_id="chunk0",
            sweep_id=i,
            cam_id=cam,
            image_path=f"/tmp/{i}.png",
            camera_seq=i,
            world_T_ego_flat=None,
            valid_camera=True,
            valid_pose=False,
        )
        for i in range(n)
    ]


def _track(predictor, detector, frames, images, **kw):
    base = dict(
        bag_id="bag0",
        chunk_id="chunk0",
        cam_id="cam_front",
        dino_model="dinov2_vitl14",
        dino_every_k=0,  # skip DINOv2 (no torch in test env)
        device="cpu",
    )
    base.update(kw)
    return track_camera(predictor, detector, frames, images, [("car", "car")], **base)


# ---------------------------------------------------------------------------
# Re-detection + IoU dedup
# ---------------------------------------------------------------------------


def test_redetection_adds_new_object_and_dedups_existing(tmp_path):
    H, W, n = 64, 64, 4
    frames = _frames(n)
    images = [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(n)]
    car = Detection(box_xyxy=(10, 10, 30, 30), cls="car", score=0.9)
    ped = Detection(box_xyxy=(40, 40, 60, 60), cls="pedestrian", score=0.8)
    # keyframes at 0 and 2: frame 0 sees the car; frame 2 re-sees the same car
    # (must dedup) plus a new pedestrian (must start a track).
    detector = _PerCallDetector([[car], [car, ped]])
    predictor = _FakePredictor(H, W)

    masklets = _track(
        predictor,
        detector,
        frames,
        images,
        masks_2d_base_dir=str(tmp_path),
        redetect_every_k=2,
        iou_match_threshold=0.5,
    )

    assert detector.calls == 2  # re-detection actually ran at the 2nd keyframe
    by_cls = {m.cls: m for m in masklets}
    assert set(by_cls) == {"car", "pedestrian"}
    # The car, prompted once at frame 0, propagates across both segments.
    assert by_cls["car"].frames_present == [0, 1, 2, 3]
    # The pedestrian only enters at the 2nd keyframe.
    assert by_cls["pedestrian"].frames_present == [2, 3]
    assert by_cls["car"].score == pytest.approx(0.9)
    for m in masklets:
        assert m.tracker_backend == "sam2"
        assert len(m.mask_paths) == len(m.frames_present)


def test_mask_saved_at_frame_resolution(tmp_path):
    from PIL import Image as PILImage

    H, W, n = 48, 96, 2  # wide frame
    frames = _frames(n)
    images = [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(n)]
    car = Detection(box_xyxy=(10, 10, 40, 30), cls="car", score=0.9)
    masklets = _track(
        _FakePredictor(H, W),
        _PerCallDetector([[car]]),
        frames,
        images,
        masks_2d_base_dir=str(tmp_path),
        redetect_every_k=1000,  # detect once
    )
    assert len(masklets) == 1
    arr = np.array(PILImage.open(masklets[0].mask_paths[0]))
    assert arr.shape == (H, W)


def test_empty_inputs_return_empty(tmp_path):
    assert _track(_FakePredictor(8, 8), _ConstDetector([]), [], [], masks_2d_base_dir=str(tmp_path)) == []


# ---------------------------------------------------------------------------
# OOM fallback (sub-clip windowing contract)
# ---------------------------------------------------------------------------


class _OOM(RuntimeError):
    """Named to satisfy _is_cuda_oom (matches OutOfMemoryError by class name)."""


_OOM.__name__ = "OutOfMemoryError"


def _two_cars() -> list[Detection]:
    # Two non-overlapping boxes → two distinct objects per window.
    return [
        Detection(box_xyxy=(1, 1, 6, 6), cls="car", score=0.9),
        Detection(box_xyxy=(9, 9, 15, 15), cls="car", score=0.9),
    ]


def test_oom_falls_back_to_sub_clips(tmp_path):
    H, W, n = 16, 16, 10
    frames = _frames(n)
    images = [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(n)]
    # A session larger than 4 frames OOMs; 4-frame windows fit.
    predictor = _FakePredictor(H, W, oom_above=4)

    masklets = _track(
        predictor,
        _ConstDetector(_two_cars()),
        frames,
        images,
        masks_2d_base_dir=str(tmp_path),
        redetect_every_k=1000,  # one keyframe per window
        sub_clip_frames=4,  # 10 frames → windows [0:4], [4:8], [8:10]
    )

    # 2 objects per window × 3 windows = 6; object ids reset between windows.
    assert len(masklets) == 6
    assert {m.cls for m in masklets} == {"car"}
    covered = sorted(s for m in masklets for s in m.frames_present)
    assert covered == sorted([0, 1, 2, 3] * 2 + [4, 5, 6, 7] * 2 + [8, 9] * 2)


def test_oom_window_recursively_halves(tmp_path):
    H, W, n = 16, 16, 8
    frames = _frames(n)
    images = [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(n)]
    predictor = _FakePredictor(H, W, oom_above=2)  # only ≤ 2-frame sessions fit

    masklets = _track(
        predictor,
        _ConstDetector(_two_cars()),
        frames,
        images,
        masks_2d_base_dir=str(tmp_path),
        redetect_every_k=1000,
        sub_clip_frames=8,  # one window, then halved 8→4→2
        min_sub_clip_frames=2,
    )

    assert len(masklets) == 8  # 4 two-frame windows × 2 objects
    covered = sorted(s for m in masklets for s in m.frames_present)
    assert covered == sorted([0, 1] * 2 + [2, 3] * 2 + [4, 5] * 2 + [6, 7] * 2)


def test_oom_at_floor_skips_window_without_raising(tmp_path):
    H, W, n = 16, 16, 4
    frames = _frames(n)
    images = [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(n)]
    predictor = _FakePredictor(H, W, oom_above=0)  # every session OOMs

    masklets = _track(
        predictor,
        _ConstDetector(_two_cars()),
        frames,
        images,
        masks_2d_base_dir=str(tmp_path),
        redetect_every_k=1000,
        sub_clip_frames=4,
        min_sub_clip_frames=1,
    )

    assert masklets == []  # degrades to nothing, does not raise
