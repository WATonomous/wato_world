"""Tests for sam2_tracker — detect on keyframes, box-prompt SAM2, propagate, and
group per-object masks into Masklets — using a fake SAM2 video predictor and a
fake detector so no `sam2` / `transformers` install or GPU is required.
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest

from wato_perception_2d.io import CameraFrameInfo
from wato_perception_2d.models.detector import Detection
from wato_perception_2d.models.sam2_tracker import track_camera


@pytest.fixture(autouse=True)
def artifact_root(tmp_path, monkeypatch):
    """Masks are written under the artifact root (wato_common.artifacts.masklets)."""
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakePredictor:
    """Minimal SAM2VideoPredictor stand-in.

    Stores the box of each added object and, on propagate, yields a +1/-1 logit
    map (positive inside the box) for every object whose add-frame ≤ the frame —
    or an all-negative map once the frame passes the object's ``vanish_after``
    (keyed by box), i.e. SAM2 reporting it absent. ``trajectories`` (keyed by
    the prompt box) scripts where SAM2 puts an object's mask per frame — a box,
    or None for absent — to reproduce SAM2 losing an object and latching onto
    another. init_state reads the window's
    frame count from the temp JPEG dir the tracker writes, so a per-window OOM
    can be simulated by frame count. Like real SAM2, propagating with no objects
    raises.
    """

    def __init__(
        self,
        H: int,
        W: int,
        oom_above: int | None = None,
        vanish_after: dict[tuple, int] | None = None,
        trajectories: dict[tuple, object] | None = None,
        strength: dict[tuple, float] | None = None,
    ) -> None:
        self.H, self.W = H, W
        self.oom_above = oom_above
        self.vanish_after = vanish_after or {}
        self.trajectories = trajectories or {}
        self.strength = strength or {}  # logit inside the mask, per prompt box
        self._objs: dict[int, tuple[int, tuple]] = {}
        self._cur_n = 0
        self.removed: list[int] = []
        self.object_frames = 0  # (object, frame) inferences run, i.e. the cost

    def init_state(
        self, video_path, offload_video_to_cpu=False, async_loading_frames=False
    ):
        self._cur_n = len([f for f in os.listdir(video_path) if f.endswith(".jpg")])
        self._objs = {}
        return {"n": self._cur_n}

    def reset_state(self, state):
        self._objs = {}

    def add_new_points_or_box(self, inference_state, frame_idx, obj_id, box):
        self._objs[int(obj_id)] = (int(frame_idx), tuple(float(v) for v in box))

    def remove_object(self, inference_state, obj_id, strict=False, need_output=True):
        self._objs.pop(int(obj_id), None)
        self.removed.append(int(obj_id))

    def _box_logit(self, box, frame) -> np.ndarray:
        m = np.full((self.H, self.W), -1.0, dtype=np.float32)
        key = tuple(int(v) for v in box)
        value = self.strength.get(key, 1.0)
        if key in self.trajectories:
            box = self.trajectories[key](frame)
            if box is None:
                return m[None]
        if frame <= self.vanish_after.get(key, frame):
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            m[y1:y2, x1:x2] = value
        return m[None]  # (1, H, W)

    def propagate_in_video(self, state, start_frame_idx=0, max_frame_num_to_track=None):
        if self.oom_above is not None and self._cur_n > self.oom_above:
            raise _OOM("CUDA out of memory")
        if not self._objs:
            raise RuntimeError("No input points or masks are provided for any object")
        n = state["n"]
        stop = (
            n - 1
            if max_frame_num_to_track is None
            else min(n - 1, start_frame_idx + max_frame_num_to_track)
        )
        for f in range(start_frame_idx, stop + 1):
            present = [
                (oid, box) for oid, (af, box) in sorted(self._objs.items()) if af <= f
            ]
            self.object_frames += len(present)
            obj_ids = [oid for oid, _ in present]
            if present:
                logits = np.stack([self._box_logit(b, f) for _, b in present], axis=0)
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
            camera_timestamp_ns=None,
            valid_camera=True,
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
        redetect_every_k=1000,  # detect once
    )
    assert len(masklets) == 1
    arr = np.array(PILImage.open(masklets[0].mask_paths[0]))
    assert arr.shape == (H, W)


def test_empty_inputs_return_empty(tmp_path):
    assert (
        _track(
            _FakePredictor(8, 8),
            _ConstDetector([]),
            [],
            [],
        )
        == []
    )


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
        redetect_every_k=1000,
        sub_clip_frames=4,
        min_sub_clip_frames=1,
    )

    assert masklets == []  # degrades to nothing, does not raise


# ---------------------------------------------------------------------------
# Session housekeeping: empty keyframes, lost objects, frame dir
# ---------------------------------------------------------------------------


def test_empty_first_keyframe_does_not_abandon_the_window(tmp_path):
    """Nothing detected at frame 0 (an empty road): an object detected at a
    later keyframe is still tracked. SAM2 refuses to propagate zero objects, so
    the tracker must skip the segment rather than let that error end the
    window."""
    H, W, n = 32, 32, 6
    car = Detection(box_xyxy=(4, 4, 12, 12), cls="car", score=0.9)
    masklets = _track(
        _FakePredictor(H, W),
        _PerCallDetector([[], [car], []]),  # keyframes 0, 2, 4
        _frames(n),
        [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(n)],
        redetect_every_k=2,
    )
    assert len(masklets) == 1
    assert masklets[0].frames_present == [2, 3, 4, 5]


def test_lost_object_is_dropped_from_the_session(tmp_path):
    """An object SAM2 reports absent for drop_lost_after_frames is removed at
    the next keyframe and costs nothing afterwards; the visible one stays."""
    H, W, n = 32, 32, 12
    gone = Detection(box_xyxy=(2, 2, 8, 8), cls="car", score=0.9)
    stays = Detection(box_xyxy=(20, 20, 28, 28), cls="car", score=0.9)
    images = [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(n)]

    def run(drop_after):
        pred = _FakePredictor(H, W, vanish_after={(2, 2, 8, 8): 1})
        mk = _track(
            pred,
            _PerCallDetector([[gone, stays]]),
            _frames(n),
            images,
            redetect_every_k=4,  # keyframes 0, 4, 8
            drop_lost_after_frames=drop_after,
        )
        return pred, sorted(m.frames_present for m in mk)

    pred, frames_present = run(drop_after=3)
    assert pred.removed == [0]  # dropped at keyframe 4 (last seen at frame 1)
    # Its masks up to vanishing are kept; the visible object is untouched.
    assert frames_present == [[0, 1], list(range(n))]

    never, frames_never = run(drop_after=0)
    assert never.removed == []
    assert frames_never == frames_present  # same labels…
    assert pred.object_frames < never.object_frames  # …for less SAM2 work


def test_frame_dir_symlinks_source_jpegs_and_encodes_the_rest(tmp_path):
    from PIL import Image as PILImage

    from wato_perception_2d.models.sam2_tracker import _prepare_frame_dir

    img = np.full((8, 8, 3), 127, dtype=np.uint8)
    jpg = tmp_path / "src.jpg"
    PILImage.fromarray(img).save(jpg)
    frames = _frames(2)
    frames[0] = dataclasses.replace(frames[0], image_path=str(jpg))
    d = _prepare_frame_dir(frames, [img, img])  # frames[1] is a missing .png
    try:
        assert sorted(os.listdir(d)) == ["00000000.jpg", "00000001.jpg"]
        assert os.path.realpath(os.path.join(d, "00000000.jpg")) == str(jpg)
        assert not os.path.islink(os.path.join(d, "00000001.jpg"))
    finally:
        import shutil

        shutil.rmtree(d)


# ---------------------------------------------------------------------------
# Identity: SAM2 latching onto another object after losing its own
# ---------------------------------------------------------------------------


def test_lost_object_is_not_resumed_on_another_object(tmp_path):
    """SAM2 loses a pedestrian (it walks out of view) and, 5 frames later —
    before the next keyframe, as on real data — reports a mask on a car
    elsewhere. Every such resume seen on real WATO data
    was a different object, so the masklet must end when the object is lost —
    not pick up the car's pixels under the pedestrian's id."""
    H, W, n = 32, 32, 20
    walker = Detection(box_xyxy=(2, 2, 6, 10), cls="pedestrian", score=0.9)
    pred = _FakePredictor(
        H,
        W,
        trajectories={
            (2, 2, 6, 10): lambda f: (
                (2, 2, 6, 10) if f <= 3 else None if f < 9 else (20, 20, 30, 28)
            )
        },
    )
    masklets = _track(
        pred,
        _PerCallDetector([[walker]]),
        _frames(n),
        [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(n)],
        redetect_every_k=10,
        drop_lost_after_frames=5,
    )
    assert [m.frames_present for m in masklets] == [[0, 1, 2, 3]]
    assert pred.removed == [0]  # and SAM2 stops paying for it


def test_track_that_jumps_onto_another_tracked_object_ends(tmp_path):
    """An oncoming car passes a parked car and leaves the image; in the same
    frame SAM2 moves the oncoming car's mask onto the parked car (no empty
    frame in between). Two tracks now hold one object: the one whose mask
    jumped ends there, the parked car's track carries on."""
    H, W, n = 40, 40, 12
    parked = Detection(box_xyxy=(2, 20, 10, 28), cls="car", score=0.9)
    oncoming = Detection(box_xyxy=(30, 20, 38, 30), cls="car", score=0.8)
    pred = _FakePredictor(
        H,
        W,
        trajectories={
            (2, 20, 10, 28): lambda f: (2, 20, 10, 28),
            # moves left 3 px/frame, then at frame 7 snaps onto the parked car
            (30, 20, 38, 30): lambda f: (
                (30 - 3 * f, 20, 38 - 3 * f, 30) if f < 7 else (2, 20, 10, 28)
            ),
        },
    )
    masklets = _track(
        pred,
        _PerCallDetector([[parked, oncoming]]),
        _frames(n),
        [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(n)],
        redetect_every_k=100,
    )
    by_score = {m.score: m.frames_present for m in masklets}
    assert by_score[0.9] == list(range(n))  # parked car: whole clip
    assert by_score[0.8] == list(range(7))  # oncoming: ends where it jumped


def test_same_keyframe_cross_class_duplicates_make_one_track(tmp_path):
    """GroundingDINO's NMS is per class, so one shuttle comes back as both
    'truck' and 'bus'. Only the higher-scoring box may start a track."""
    H, W, n = 32, 32, 3
    truck = Detection(box_xyxy=(4, 4, 20, 16), cls="truck", score=0.60)
    bus = Detection(box_xyxy=(4, 4, 20, 17), cls="bus", score=0.43)
    masklets = _track(
        _FakePredictor(H, W),
        _PerCallDetector([[bus, truck]]),  # lower score first: order must not matter
        _frames(n),
        [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(n)],
        redetect_every_k=100,
    )
    assert [m.cls for m in masklets] == ["truck"]


def test_overlapping_masks_give_each_pixel_to_one_object(tmp_path):
    """SAM2 infers each object independently, so a bus's mask can wrap around
    a car half-hidden beside it. A pixel is recorded for one object only — the
    one SAM2 is most confident about there — or the car's LiDAR points would be
    lifted as both car and bus."""
    from PIL import Image as PILImage

    H, W, n = 32, 32, 2
    bus = Detection(box_xyxy=(2, 2, 20, 20), cls="bus", score=0.9)
    car = Detection(box_xyxy=(16, 10, 24, 18), cls="car", score=0.8)
    pred = _FakePredictor(
        H,
        W,
        trajectories={(2, 2, 20, 20): lambda f: (2, 2, 20, 20)},
        strength={(2, 2, 20, 20): 1.0, (16, 10, 24, 18): 3.0},
    )
    masklets = _track(
        pred,
        _PerCallDetector([[bus, car]]),
        _frames(n),
        [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(n)],
        redetect_every_k=100,
    )
    m = {mk.cls: np.array(PILImage.open(mk.mask_paths[0])) > 0 for mk in masklets}
    assert not (m["bus"] & m["car"]).any()
    assert m["car"].sum() == 8 * 8  # the car keeps all of its pixels
    assert m["bus"].sum() == 18 * 18 - 4 * 8  # the bus loses the contested ones
