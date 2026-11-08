"""Tests for sam2_tracker — detect on keyframes, box-prompt SAM2, propagate, and
group per-object masks into Masklets — using a fake SAM2 video predictor and a
fake detector so no `sam2` / `transformers` install or GPU is required.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image as PILImage

from wato_common.artifacts.frame_index import CameraFrame
from wato_perception_2d.models.detector import Detection
from wato_perception_2d.models.sam2_tracker import track_camera
from wato_perception_2d.config import MaskRefinementConfig
from wato_perception_2d.models.mask_refinement import RefinedMask


@pytest.fixture(autouse=True)
def artifact_root(tmp_path, monkeypatch):
    """Masks are written under the artifact root (wato_common.artifacts.masklets)."""
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _OOM(RuntimeError):
    """Named to satisfy _is_cuda_oom (matches OutOfMemoryError by class name)."""


_OOM.__name__ = "OutOfMemoryError"


class _FakePredictor:
    """Minimal SAM2VideoPredictor stand-in.

    Stores the box of each added object and, on propagate, yields a logit map
    (``strength``, default 1, inside the box; -1 elsewhere) for every object
    whose add-frame ≤ the frame. ``trajectories`` (keyed by the prompt box)
    scripts where SAM2 puts an object's mask per frame — a box, or None for
    absent. init_state reads the window's frame count from the temp JPEG dir
    the tracker writes, so a per-window OOM can be simulated by frame count.
    Like real SAM2, propagating with no objects raises.
    """

    def __init__(
        self,
        H: int,
        W: int,
        oom_above: int | None = None,
        trajectories: dict[tuple, object] | None = None,
        strength: dict[tuple, float] | None = None,
    ) -> None:
        self.H, self.W = H, W
        self.oom_above = oom_above
        self.trajectories = trajectories or {}
        self.strength = strength or {}
        self._objs: dict[int, tuple[int, tuple]] = {}
        self._cur_n = 0
        self.prompts: list[tuple[int, int, tuple]] = []  # (frame, obj_id, box)
        self.removed: list[int] = []

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
        self.prompts.append((int(frame_idx), int(obj_id), tuple(int(v) for v in box)))
        return self._prompt_output(frame_idx)

    def _prompt_output(self, frame_idx):
        """Like SAM2: (frame_idx, every object's id, their logits there)."""
        ids = sorted(self._objs)
        logits = np.stack(
            [self._box_logit(self._objs[o][1], frame_idx) for o in ids], axis=0
        )
        return frame_idx, ids, logits

    def remove_object(self, inference_state, obj_id, strict=False, need_output=True):
        self._objs.pop(int(obj_id), None)
        self.removed.append(int(obj_id))

    def _box_logit(self, box, frame) -> np.ndarray:
        m = np.full((self.H, self.W), -1.0, dtype=np.float32)
        key = tuple(int(v) for v in box)
        if key in self.trajectories:
            box = self.trajectories[key](frame)
            if box is None:
                return m[None]
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        m[y1:y2, x1:x2] = self.strength.get(key, 1.0)
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
    """Returns the same detections on every call."""

    def __init__(self, dets: list[Detection]) -> None:
        self._dets = dets

    def detect(self, image_rgb, concepts):
        return list(self._dets)


def _frames(n: int) -> list[CameraFrame]:
    return [
        CameraFrame(
            cam_id="cam_front",
            camera_seq=i,
            timestamp_ns=i,
            image_path=f"/tmp/{i}.png",
            sweep_ids=(i,),
            world_T_ego=None,
        )
        for i in range(n)
    ]


def _track(predictor, detector, n, H, W, stamped=False, **kw):
    base = dict(
        bag_id="bag0",
        chunk_id="chunk0",
        cam_id="cam_front",
        tracker_backend="sam2",
        dino_model="dinov2_vitl14",
        dino_every_k=0,  # skip DINOv2 (no torch in test env)
        device="cpu",
        # Detector support and re-prompting off unless a test turns them on,
        # so each test exercises its own rule.
        birth_score=0.0,
        support_keyframes=0,
        reprompt_below_iou=0.0,
        birth_gate="none",
        backfill_frames=0,
    )
    base.update(kw)
    # stamped: image i is filled with 8 * i, which _StampedPredictor reads
    # back to know the window frame behind each session frame.
    images = [
        np.full((H, W, 3), 8 * i if stamped else 0, dtype=np.uint8) for i in range(n)
    ]
    return track_camera(
        predictor, detector, _frames(n), images, [("car", "car")], **base
    )


def _mask(masklet, i: int = 0) -> np.ndarray:
    return np.array(PILImage.open(masklet.mask_paths[i])) > 0


def _two_cars() -> list[Detection]:
    return [
        Detection(box_xyxy=(1, 1, 6, 6), cls="car", score=0.9),
        Detection(box_xyxy=(9, 9, 15, 15), cls="car", score=0.9),
    ]


# ---------------------------------------------------------------------------
# Keyframes, windows, failures
# ---------------------------------------------------------------------------


def test_redetection_adds_new_object_and_dedups_existing():
    H, W, n = 64, 64, 4
    car = Detection(box_xyxy=(10, 10, 30, 30), cls="car", score=0.9)
    ped = Detection(box_xyxy=(40, 40, 60, 60), cls="pedestrian", score=0.8)
    # Keyframes 0 and 2: frame 2 re-sees the car (deduped) and a new pedestrian.
    detector = _PerCallDetector([[car], [car, ped]])

    masklets = _track(_FakePredictor(H, W), detector, n, H, W, redetect_every_k=2)

    assert detector.calls == 2
    by_cls = {m.cls: m for m in masklets}
    assert by_cls["car"].frames_present == [0, 1, 2, 3]
    assert by_cls["pedestrian"].frames_present == [2, 3]
    assert by_cls["car"].score == pytest.approx(0.9)
    for m in masklets:
        assert len(m.mask_paths) == len(m.frames_present)
        assert _mask(m).shape == (H, W)


def test_oom_window_recursively_halves():
    H, W, n = 16, 16, 8
    predictor = _FakePredictor(H, W, oom_above=2)  # only ≤ 2-frame sessions fit

    masklets = _track(
        predictor,
        _ConstDetector(_two_cars()),
        n,
        H,
        W,
        redetect_every_k=1000,
        sub_clip_frames=8,  # one window, then halved 8→4→2
        min_sub_clip_frames=2,
    )

    assert len(masklets) == 8  # 4 two-frame windows × 2 objects
    covered = sorted(s for m in masklets for s in m.frames_present)
    assert covered == sorted([0, 1] * 2 + [2, 3] * 2 + [4, 5] * 2 + [6, 7] * 2)


def test_oom_at_smallest_window_stops_instead_of_publishing_gaps():
    events: list[dict] = []
    with pytest.raises(RuntimeError, match="camera incomplete"):
        _track(
            _FakePredictor(16, 16, oom_above=1),
            _ConstDetector(_two_cars()),
            4,
            16,
            16,
            sub_clip_frames=4,
            min_sub_clip_frames=2,
            diagnostics=events,
        )
    assert events == []  # a failed camera reports nothing


@pytest.mark.parametrize("oom", [False, True])
def test_late_prompt_failure_reaches_caller_or_oom_retry(oom):
    class FailingPrompt(_FakePredictor):
        def add_new_points_or_box(self, inference_state, frame_idx, obj_id, box):
            if obj_id == 1:
                if oom and self._cur_n > 2:
                    raise _OOM("CUDA out of memory during late birth")
                if not oom:
                    raise RuntimeError("late object failed")
            super().add_new_points_or_box(inference_state, frame_idx, obj_id, box)

    pred = FailingPrompt(16, 16)
    if oom:
        masks = _track(
            pred,
            _ConstDetector(_two_cars()),
            4,
            16,
            16,
            redetect_every_k=100,
            sub_clip_frames=2,
            min_sub_clip_frames=2,
        )
        assert len(masks) == 4  # both objects in each successful window
    else:
        with pytest.raises(RuntimeError, match="late object failed"):
            _track(pred, _ConstDetector(_two_cars()), 4, 16, 16)


def test_diagnostics_contain_only_successful_windows():
    diagnostics: list[dict] = []
    _track(
        _FakePredictor(16, 16, oom_above=2),
        _ConstDetector(_two_cars()),
        8,
        16,
        16,
        redetect_every_k=100,
        sub_clip_frames=8,
        min_sub_clip_frames=2,
        diagnostics=diagnostics,
    )
    events = [e for e in diagnostics if e["kind"] == "mask"]
    assert len(events) == 16  # no rows from abandoned full / half-sized attempts
    assert {e["window_start"] for e in events} == {0, 2, 4, 6}
    assert all(e["masklet_id"] for e in events)
    assert sorted(e["camera_seq"] for e in events) == sorted(list(range(8)) * 2)


def test_without_diagnostics_no_stability_is_computed(monkeypatch):
    """Stability costs a GPU sync per frame; off by default, nothing pays it."""
    from wato_perception_2d.models import sam2_tracker

    def unexpected(*args, **kwargs):
        raise AssertionError("stability computed with diagnostics off")

    monkeypatch.setattr(sam2_tracker, "stability_scores", unexpected)
    masks = _track(_FakePredictor(16, 16), _ConstDetector(_two_cars()), 4, 16, 16)
    assert len(masks) == 2


def test_masklets_carry_the_backend_that_made_them():
    masks = _track(
        _FakePredictor(16, 16),
        _ConstDetector(_two_cars()),
        2,
        16,
        16,
        tracker_backend="dam4sam",
    )
    assert {m.tracker_backend for m in masks} == {"dam4sam"}


@pytest.mark.parametrize("failing", ["detector", "embeddings"])
def test_model_failure_propagates(failing, monkeypatch):
    """Swallowed, a missing checkpoint left a camera with no masklets or no
    embeddings, which looks like a finished chunk."""
    from wato_perception_2d.models import sam2_tracker

    def unavailable(*_a, **_k):
        raise RuntimeError("checkpoint not found")

    detector = _ConstDetector(_two_cars())
    if failing == "detector":
        detector.detect = unavailable
    else:
        monkeypatch.setattr(sam2_tracker, "extract_dino_features", unavailable)

    with pytest.raises(RuntimeError, match="not found"):
        _track(_FakePredictor(16, 16), detector, 4, 16, 16, dino_every_k=1)


def test_empty_first_keyframe_does_not_abandon_the_window():
    """SAM2 refuses to propagate zero objects, so an empty first keyframe must
    skip its segment, not end the window."""
    H, W, n = 32, 32, 6
    car = Detection(box_xyxy=(4, 4, 12, 12), cls="car", score=0.9)
    (m,) = _track(
        _FakePredictor(H, W),
        _PerCallDetector([[], [car], []]),  # keyframes 0, 2, 4
        n,
        H,
        W,
        redetect_every_k=2,
    )
    assert m.frames_present == [2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Identity: SAM2 latching onto another object
# ---------------------------------------------------------------------------


def test_lost_object_is_not_resumed_on_another_object():
    """SAM2 loses a pedestrian and, before the next keyframe, reports its mask
    on a car elsewhere: the masklet ends when the object is lost, and the
    track leaves the session at the next keyframe."""
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
        n,
        H,
        W,
        redetect_every_k=10,
        drop_lost_after_frames=5,
    )
    assert [m.frames_present for m in masklets] == [[0, 1, 2, 3]]
    assert pred.removed == [0]


def test_tracks_with_identical_masks_remain_independent_hypotheses():
    """A mask that snaps onto another track remains available for 3D rejection.

    Pixel overlap alone cannot establish which object its LiDAR points support.
    """
    H, W, n = 40, 40, 12
    parked = Detection(box_xyxy=(2, 20, 10, 28), cls="car", score=0.9)
    oncoming = Detection(box_xyxy=(30, 20, 38, 30), cls="car", score=0.8)
    pred = _FakePredictor(
        H,
        W,
        trajectories={
            (2, 20, 10, 28): lambda f: (2, 20, 10, 28),
            (30, 20, 38, 30): lambda f: (
                (30 - 3 * f, 20, 38 - 3 * f, 30) if f < 7 else (2, 20, 10, 28)
            ),
        },
    )
    masklets = _track(
        pred,
        _PerCallDetector([[parked, oncoming]]),
        n,
        H,
        W,
        redetect_every_k=100,
    )
    by_score = {m.score: m.frames_present for m in masklets}
    assert by_score[0.9] == list(range(n))
    assert by_score[0.8] == list(range(n))
    assert pred.removed == []


def test_same_keyframe_cross_class_duplicates_make_one_track():
    """The detector's NMS is per class, so one shuttle comes back as both
    'truck' and 'bus'. Only the higher-scoring box may start a track."""
    truck = Detection(box_xyxy=(4, 4, 20, 16), cls="truck", score=0.60)
    bus = Detection(box_xyxy=(4, 4, 20, 17), cls="bus", score=0.43)
    masklets = _track(
        _FakePredictor(32, 32),
        _PerCallDetector([[bus, truck]]),  # lower score first: order must not matter
        3,
        32,
        32,
        redetect_every_k=100,
    )
    assert [m.cls for m in masklets] == ["truck"]


# ---------------------------------------------------------------------------
# Independent competing masks
# ---------------------------------------------------------------------------


def test_overlapping_masks_preserve_every_objects_claimed_pixels():
    """Higher car logits cannot remove the bus's claims before 3D refinement."""
    H, W, n = 32, 32, 2
    bus = Detection(box_xyxy=(2, 2, 20, 20), cls="bus", score=0.9)
    car = Detection(box_xyxy=(16, 10, 24, 18), cls="car", score=0.8)
    pred = _FakePredictor(H, W, strength={(16, 10, 24, 18): 3.0})
    masklets = _track(
        pred, _PerCallDetector([[bus, car]]), n, H, W, redetect_every_k=100
    )
    for frame in range(n):
        masks = {mk.cls: _mask(mk, frame) for mk in masklets}
        assert (masks["bus"] & masks["car"]).sum() == 4 * 8
        assert masks["car"].sum() == 8 * 8
        assert masks["bus"].sum() == 18 * 18


def test_lower_logit_track_is_not_collapsed_or_ended():
    """The small candidate keeps its complete mask across several keyframes."""
    H, W, n = 64, 64, 8
    car = Detection(box_xyxy=(10, 10, 50, 50), cls="car", score=0.9)
    ped = Detection(box_xyxy=(20, 20, 30, 51), cls="pedestrian", score=0.5)
    pred = _FakePredictor(H, W, strength={(10, 10, 50, 50): 5.0})
    masklets = _track(
        pred,
        _ConstDetector([car, ped]),
        n,
        H,
        W,
        redetect_every_k=3,
    )
    assert {m.cls: m.frames_present for m in masklets} == {
        "car": list(range(n)),
        "pedestrian": list(range(n)),
    }
    assert pred.removed == []
    for masklet in masklets:
        assert all(
            _mask(masklet, i).sum() == (1600 if masklet.cls == "car" else 310)
            for i in range(n)
        )


def test_drop_islands_keeps_the_body_and_comparable_parts():
    from wato_perception_2d.models.sam2_tracker import _drop_islands

    m = np.zeros((40, 40), dtype=bool)
    m[5:25, 5:25] = True  # body, 400 px
    m[30, 30] = m[2, 35] = True  # specks
    m[5:15, 30:38] = True  # a part split off by a pole, 80 px (20% of the body)
    out = _drop_islands(m, 0.1)
    assert out.sum() == 400 + 80 and not out[30, 30] and not out[2, 35]

    tiny = np.zeros((40, 40), dtype=bool)
    tiny[10:13, 10:13] = True  # a distant object: its only component stays
    assert (_drop_islands(tiny, 0.1) == tiny).all()


# ---------------------------------------------------------------------------
# Ego vehicle
# ---------------------------------------------------------------------------


def _ego(H: int, W: int, top: int) -> np.ndarray:
    """The rack along the bottom of the image, from row ``top`` down."""
    ego = np.zeros((H, W), dtype=bool)
    ego[top:] = True
    return ego


def test_ego_vehicle_is_never_an_object():
    """The detector boxes the rack as a truck at every keyframe, and a car
    right behind has its mask run down over the rack: neither the rack nor its
    pixels are recorded."""
    H, W, n = 64, 64, 4
    rack = Detection(box_xyxy=(0, 50, 64, 64), cls="truck", score=0.35)
    car = Detection(box_xyxy=(20, 30, 40, 56), cls="car", score=0.9)
    (m,) = _track(
        _FakePredictor(H, W),
        _ConstDetector([rack, car]),
        n,
        H,
        W,
        redetect_every_k=2,
        ego_mask=_ego(H, W, 48),
    )
    assert m.cls == "car"
    assert _mask(m).sum() == 20 * 18 and not _mask(m)[48:].any()


def test_track_segmenting_the_ego_vehicle_ends_and_is_not_prompted_again():
    """A 'truck' box over the rack and the road beside it is mostly road, but
    SAM2 segments the rack: the track ends unrecorded, and the same box at
    later keyframes is not prompted again."""
    H, W = 64, 64
    box = (0, 30, 64, 64)  # 16 of its 34 rows are rack
    truck = Detection(box_xyxy=box, cls="truck", score=0.26)
    pred = _FakePredictor(H, W, trajectories={box: lambda f: (0, 50, 64, 64)})
    masklets = _track(
        pred,
        _ConstDetector([truck]),
        6,
        H,
        W,
        redetect_every_k=2,
        ego_mask=_ego(H, W, 48),
    )
    assert masklets == []
    assert pred.removed == [0]


# ---------------------------------------------------------------------------
# Contained image masks remain competing hypotheses
# ---------------------------------------------------------------------------


def test_mask_contained_inside_vehicle_survives_for_3d_refinement():
    """A contained image mask is retained; image containment cannot own points."""
    H, W, n = 64, 64, 5
    car = Detection(box_xyxy=(10, 10, 50, 50), cls="car", score=0.9)
    driver = Detection(box_xyxy=(25, 18, 35, 30), cls="pedestrian", score=0.6)
    pred = _FakePredictor(H, W, strength={(25, 18, 35, 30): 5.0})
    masklets = _track(
        pred,
        _ConstDetector([car, driver]),
        n,
        H,
        W,
        redetect_every_k=3,
    )
    by_cls = {m.cls: m for m in masklets}
    assert set(by_cls) == {"car", "pedestrian"}
    assert all(m.frames_present == list(range(n)) for m in masklets)
    assert _mask(by_cls["car"]).sum() == 40 * 40
    assert _mask(by_cls["pedestrian"]).sum() == 10 * 12
    assert pred.removed == []


def test_pedestrian_in_front_of_a_car_is_not_contained():
    """A pedestrian in front of a car reaches the road below the car's bottom."""
    H, W, n = 64, 64, 4
    car = Detection(box_xyxy=(10, 10, 50, 50), cls="car", score=0.9)
    ped = Detection(box_xyxy=(25, 18, 35, 52), cls="pedestrian", score=0.8)
    pred = _FakePredictor(H, W, strength={(25, 18, 35, 52): 5.0})
    masklets = _track(
        pred,
        _PerCallDetector([[car, ped]]),
        n,
        H,
        W,
        redetect_every_k=100,
    )
    assert {m.cls: m.frames_present for m in masklets} == {
        "car": [0, 1, 2, 3],
        "pedestrian": [0, 1, 2, 3],
    }


# ---------------------------------------------------------------------------
# Detector support: confirmation, keep-alive, class vote, re-prompting
# ---------------------------------------------------------------------------

_SUPPORT = dict(
    redetect_every_k=2, birth_score=0.5, confirm_keyframes=3, support_keyframes=2
)


def test_weak_detection_is_written_only_once_confirmed(tmp_path):
    """A weak car boxed at three keyframes in a row is confirmed and written
    from its birth; weak junk the detector stops boxing is dropped with its
    held masks."""
    H, W, n = 32, 32, 8
    car = Detection(box_xyxy=(4, 4, 14, 14), cls="car", score=0.3)
    junk = Detection(box_xyxy=(18, 18, 30, 30), cls="truck", score=0.3)
    pred = _FakePredictor(H, W)
    (m,) = _track(
        pred,
        _PerCallDetector([[car, junk], [car, junk], [car], [car]]),
        n,
        H,
        W,
        **_SUPPORT,
    )
    assert (m.cls, m.frames_present) == ("car", list(range(n)))
    written = [f for _, _, files in os.walk(tmp_path / "raw") for f in files]
    assert len(written) == n
    assert pred.removed == [1]


def test_track_the_detector_stops_seeing_ends():
    """SAM2 keeps a large mask going after the detector stops boxing anything
    there: the track ends once its keep-alive runs out."""
    car = Detection(box_xyxy=(4, 4, 20, 20), cls="car", score=0.9)
    (m,) = _track(
        _FakePredictor(32, 32), _PerCallDetector([[car]]), 10, 32, 32, **_SUPPORT
    )
    assert m.frames_present == [0, 1, 2, 3]  # unsupported at keyframes 2 and 4


def test_distant_object_the_detector_misses_is_kept():
    """A mask under support_min_area of the image needs no support: the
    detector boxes a distant car once, SAM2 tracks it fine."""
    car = Detection(box_xyxy=(4, 4, 12, 10), cls="car", score=0.27)
    (m,) = _track(
        _FakePredictor(200, 200),
        _PerCallDetector([[car]]),
        10,
        200,
        200,
        support_min_area=0.01,
        **{**_SUPPORT, "birth_score": 0.0},
    )
    assert m.frames_present == list(range(10))


def test_class_is_the_vote_of_the_supporting_detections():
    box = (4, 4, 20, 20)
    per_call = [
        [Detection(box_xyxy=box, cls="truck", score=0.6)],
        [Detection(box_xyxy=box, cls="bus", score=0.5)],
        [Detection(box_xyxy=box, cls="bus", score=0.3)],
    ]
    (m,) = _track(
        _FakePredictor(32, 32), _PerCallDetector(per_call), 6, 32, 32, **_SUPPORT
    )
    assert m.cls == "bus" and m.score == pytest.approx(0.4)


def test_drifted_mask_is_reprompted_with_a_confident_box():
    """SAM2's mask slides off the car (box IoU 0.67 with the detector's box at
    the next keyframe): the box re-prompts that track there."""
    box = (10, 10, 30, 30)
    car = Detection(box_xyxy=box, cls="car", score=0.9)
    pred = _FakePredictor(40, 40, trajectories={box: lambda f: (14, 10, 34, 30)})
    _track(
        pred,
        _ConstDetector([car]),
        4,
        40,
        40,
        reprompt_score=0.8,
        reprompt_below_iou=0.7,
        **_SUPPORT,
    )
    assert pred.prompts == [(0, 0, box), (2, 0, box)]


def test_no_reprompt_from_a_box_backing_two_tracks():
    """Two tracks each holding half of what the detector boxes as one car:
    re-prompting either with the whole box would make it a duplicate."""
    left, right = (0, 0, 6, 6), (34, 34, 40, 40)
    pred = _FakePredictor(
        40,
        40,
        trajectories={
            left: lambda f: left if f == 0 else (10, 10, 20, 30),
            right: lambda f: right if f == 0 else (20, 10, 30, 30),
        },
    )
    seeds = [Detection(box_xyxy=b, cls="car", score=0.9) for b in (left, right)]
    car = Detection(box_xyxy=(10, 10, 30, 30), cls="car", score=0.9)
    _track(
        pred,
        _PerCallDetector([seeds, [car]]),
        4,
        40,
        40,
        iou_match_threshold=0.4,  # each half's bbox has IoU 0.5 with the box
        reprompt_score=0.8,
        reprompt_below_iou=0.7,
        **_SUPPORT,
    )
    assert [frame for frame, _, _ in pred.prompts] == [0, 0]


class _MaskPromptPredictor(_FakePredictor):
    def __init__(self, H, W):
        super().__init__(H, W)
        self.mask_prompts = []
        self.seed_masks = {}

    def add_new_mask(self, inference_state, frame_idx, obj_id, mask):
        from wato_perception_2d.models.sam2_tracker import _mask_bbox

        self.mask_prompts.append((frame_idx, obj_id, mask.copy()))
        self._objs[obj_id] = (frame_idx, _mask_bbox(mask))
        self.seed_masks[obj_id] = mask.copy()
        return self._prompt_output(frame_idx)

    def propagate_in_video(self, *args, **kwargs):
        for f, ids, logits in super().propagate_in_video(*args, **kwargs):
            for i, oid in enumerate(ids):
                if oid in self.seed_masks:
                    logits[i, 0] = np.where(self.seed_masks[oid], 4, -4)
            yield f, ids, logits


def _script_refiner(monkeypatch, candidates):
    from wato_perception_2d.models import sam2_tracker

    instances = []

    class ScriptedRefiner:
        def __init__(self, *args):
            self.candidates = iter(candidates)
            self.images = 0
            instances.append(self)

        def set_image(self, image):
            self.images += 1

        def reset(self):
            pass

        def propose(self, *args):
            return next(self.candidates)

    monkeypatch.setattr(sam2_tracker, "KeyframeMaskRefiner", ScriptedRefiner)
    return instances


def test_mask_seeds_share_keyframe_encoding_and_are_marked_in_diagnostics(
    monkeypatch, tmp_path
):
    masks = []
    for det in _two_cars():
        mask = np.zeros((16, 16), dtype=bool)
        x1, y1, x2, y2 = det.box_xyxy
        mask[y1:y2, x1:x2] = True
        masks.append(mask)
    refiners = _script_refiner(
        monkeypatch, [RefinedMask(m, 0.9, 1.0, "accepted") for m in masks]
    )
    pred = _MaskPromptPredictor(16, 16)
    events: list[dict] = []
    result = _track(
        pred,
        _ConstDetector(_two_cars()),
        2,
        16,
        16,
        refinement=MaskRefinementConfig(enabled=True),
        diagnostics=events,
    )
    assert len(result) == 2 and len(pred.mask_prompts) == 2
    assert pred.prompts == []  # accepted masks enter the video API directly
    assert refiners[0].images == 1
    frame0 = [e for e in events if e["kind"] == "mask" and e["camera_seq"] == 0]
    assert all(e["logit_source"] == "mask_prompt" for e in frame0)


def test_bad_seed_quality_preserves_a_weak_distant_detection(monkeypatch, tmp_path):
    _script_refiner(
        monkeypatch, [RefinedMask(None, None, None, "no_acceptable_candidate")]
    )
    det = Detection(box_xyxy=(1, 1, 4, 4), cls="car", score=0.26)
    pred = _MaskPromptPredictor(16, 16)
    events: list[dict] = []
    (m,) = _track(
        pred,
        _ConstDetector([det]),
        2,
        16,
        16,
        refinement=MaskRefinementConfig(enabled=True),
        diagnostics=events,
    )
    assert m.score == pytest.approx(0.26)
    assert m.frames_present == [0, 1]
    assert not pred.mask_prompts
    assert events[0]["applied"] is False


@pytest.mark.parametrize("major_disagreement", [False, True])
def test_shape_correction_with_identical_bboxes_has_an_identity_guard(
    monkeypatch, tmp_path, major_disagreement
):
    # The fresh silhouette and propagated rectangle have the same tight bbox.
    # A hole changes pixel IoU without changing association or bbox-IoU drift.
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    if major_disagreement:
        mask[11:29, 11:29] = False  # IoU .19: abstain, potential identity error
    else:
        mask[14:26, 14:26] = False  # IoU .64: eligible for correction
    _script_refiner(
        monkeypatch,
        [
            RefinedMask(None, None, None, "no_acceptable_candidate"),
            RefinedMask(mask, 0.95, 1.0, "accepted"),
        ],
    )
    pred = _MaskPromptPredictor(40, 40)
    det = Detection(box_xyxy=(10, 10, 30, 30), cls="car", score=0.9)
    events: list[dict] = []
    (m,) = _track(
        pred,
        _ConstDetector([det]),
        4,
        40,
        40,
        refinement=MaskRefinementConfig(enabled=True, correct_shape_disagreement=True),
        reprompt_below_iou=0.7,
        redetect_every_k=2,
        diagnostics=events,
    )
    assert m.frames_present == [0, 1, 2, 3]
    assert m.score == pytest.approx(0.9)  # mask score never replaces detector score
    assert len(pred.mask_prompts) == (0 if major_disagreement else 1)
    if not major_disagreement:
        np.testing.assert_array_equal(_mask(m, 2), mask)
    prompts = [e for e in events if e["kind"] == "prompt"]
    assert prompts[-1]["applied"] is (not major_disagreement)
    assert prompts[-1]["masklet_id"] == m.masklet_id


# ---------------------------------------------------------------------------
# Birth gate: births are tested against tracked mask pixels
# ---------------------------------------------------------------------------


def test_part_box_inside_a_tracked_car_does_not_start_a_track():
    """YOLO boxes a corner of a car already tracked. Its box IoU with the car's
    mask bbox is 0.14, so it supports nothing, but all of it lies inside the
    car's mask: under SAM2MOT's test (> 50% untracked) it starts no track."""
    car = Detection(box_xyxy=(10, 10, 50, 40), cls="car", score=0.9)
    corner = Detection(box_xyxy=(10, 10, 24, 22), cls="car", score=0.3)
    events: list[dict] = []
    (m,) = _track(
        _FakePredictor(64, 64),
        _PerCallDetector([[car], [car, corner]]),
        4,
        64,
        64,
        redetect_every_k=2,
        birth_gate="untracked_area",
        diagnostics=events,
    )
    assert m.frames_present == [0, 1, 2, 3]
    births = [e for e in events if e["kind"] == "birth"]
    assert [(e["camera_seq"], e["accepted"]) for e in births] == [
        (0, True),
        (2, False),
    ]
    assert births[0]["masklet_id"] == m.masklet_id
    assert births[1]["untracked_fraction"] == 0.0
    assert births[1]["masklet_id"] is None


def test_part_box_behind_its_cars_box_at_the_same_keyframe_is_refused():
    """Births at one keyframe go highest score first, each tested against the
    masks of the births before it: the corner fails behind the whole car."""
    car = Detection(box_xyxy=(10, 10, 50, 40), cls="car", score=0.84)
    corner = Detection(box_xyxy=(36, 28, 50, 40), cls="truck", score=0.30)
    masklets = _track(
        _FakePredictor(64, 64),
        _PerCallDetector([[corner, car]]),
        2,
        64,
        64,
        redetect_every_k=100,
        birth_gate="untracked_area",
    )
    assert [m.cls for m in masklets] == ["car"]


@pytest.mark.parametrize(
    "box,born",
    [
        ((30, 30, 60, 40), False),  # a third of it untracked
        ((40, 30, 64, 40), True),  # 58% untracked
    ],
)
def test_box_is_born_when_most_of_it_is_untracked(box, born):
    car = Detection(box_xyxy=(10, 10, 50, 50), cls="car", score=0.9)
    other = Detection(box_xyxy=box, cls="truck", score=0.5)
    masklets = _track(
        _FakePredictor(64, 64),
        _PerCallDetector([[car], [car, other]]),
        4,
        64,
        64,
        redetect_every_k=2,
        birth_gate="untracked_area",
    )
    assert sorted(m.cls for m in masklets) == (["car", "truck"] if born else ["car"])


def test_box_on_one_car_of_a_mask_bled_over_two_is_refused():
    """A parked car's mask bleeds onto the car beside it. A confident box on
    the second car has box IoU 0.45 with the bled bbox, so it matched nothing
    and started a duplicate; it lies inside the bled mask, so it is refused."""
    first = Detection(box_xyxy=(4, 20, 20, 32), cls="car", score=0.6)
    second = Detection(box_xyxy=(26, 20, 44, 32), cls="car", score=0.72)
    pred = _FakePredictor(
        64,
        64,
        trajectories={(4, 20, 20, 32): lambda f: (4, 20, 44, 32) if f else (4, 20, 20, 32)},
    )
    masklets = _track(
        pred,
        _PerCallDetector([[first], [second]]),
        4,
        64,
        64,
        redetect_every_k=2,
        birth_gate="untracked_area",
    )
    assert [m.score for m in masklets] == [pytest.approx(0.6)]


def _box_mask_refiner(monkeypatch, H, W):
    """The image predictor as the mask_iou gate uses it: a box's mask is the box."""
    from wato_perception_2d.models import sam2_tracker

    encoded: list[int] = []

    class BoxMasks:
        def __init__(self, *args):
            pass

        def set_image(self, image):
            encoded.append(1)

        def reset(self):
            pass

        def box_mask(self, box):
            mask = np.zeros((H, W), dtype=bool)
            x1, y1, x2, y2 = (int(v) for v in box)
            mask[y1:y2, x1:x2] = True
            return mask

    monkeypatch.setattr(sam2_tracker, "KeyframeMaskRefiner", BoxMasks)
    return encoded


@pytest.mark.parametrize(
    "gate,born",
    [
        # SAM2MOT refuses both boxes inside the car; SAM 3's mask IoU (0.02)
        # lets the small one through and refuses the larger overlap (0.12).
        ("untracked_area", ["car", "truck"]),
        ("mask_iou", ["car", "motorcycle", "truck"]),
    ],
)
def test_untracked_area_and_mask_iou_gates_differ_on_a_small_box_inside(
    monkeypatch, gate, born
):
    encoded = _box_mask_refiner(monkeypatch, 64, 64)
    car = Detection(box_xyxy=(10, 10, 50, 50), cls="car", score=0.9)
    overlap = Detection(box_xyxy=(30, 30, 60, 40), cls="bus", score=0.6)
    small = Detection(box_xyxy=(20, 20, 26, 26), cls="motorcycle", score=0.5)
    clear = Detection(box_xyxy=(45, 52, 64, 62), cls="truck", score=0.4)
    masklets = _track(
        _FakePredictor(64, 64),
        _PerCallDetector([[car], [car, overlap, small, clear]]),
        4,
        64,
        64,
        redetect_every_k=2,
        birth_gate=gate,
    )
    assert sorted(m.cls for m in masklets) == born
    # Encoded once, at the keyframe with something to overlap.
    assert len(encoded) == (1 if gate == "mask_iou" else 0)


def test_mask_events_carry_sam2_quality_when_the_predictor_reports_it():
    class Reporting(_FakePredictor):
        def frame_quality(self, state, obj_id):
            return {"predicted_iou": 0.7, "object_score_logit": 3.0}

    events: list[dict] = []
    _track(Reporting(16, 16), _ConstDetector(_two_cars()), 2, 16, 16, diagnostics=events)
    masks = [e for e in events if e["kind"] == "mask"]
    assert len(masks) == 4
    assert all(
        e["predicted_iou"] == 0.7 and e["object_score_logit"] == 3.0 for e in masks
    )


# ---------------------------------------------------------------------------
# Reverse back-fill from each birth
# ---------------------------------------------------------------------------


class _StampedPredictor(_FakePredictor):
    """Reads each session frame's window frame from its pixels
    (``_track(stamped=True)``), so trajectories follow window frames in the
    back-fill's reversed session."""

    def init_state(self, video_path, **kwargs):
        state = super().init_state(video_path, **kwargs)
        names = sorted(
            (f for f in os.listdir(video_path) if f.endswith(".jpg")),
            key=lambda f: int(f[:-4]),
        )
        self.sessions = getattr(self, "sessions", [])
        self._window_frame = [
            round(np.asarray(PILImage.open(os.path.join(video_path, f))).mean() / 8)
            for f in names
        ]
        self.sessions.append(list(self._window_frame))
        return state

    def _box_logit(self, box, frame):
        return super()._box_logit(box, self._window_frame[frame])


def _late_car(trajectory, *, n=12, **kw):
    """A car tracked from image 0, and a distant one YOLO first boxes at
    keyframe 8 whose mask SAM2 places by ``trajectory`` (window frame → box)."""
    near = Detection(box_xyxy=(2, 2, 8, 8), cls="car", score=0.9)
    far = Detection(box_xyxy=(20, 20, 28, 28), cls="truck", score=0.4)
    pred = _StampedPredictor(32, 32, trajectories={far.box_xyxy: trajectory})
    events: list[dict] = []
    masklets = _track(
        pred,
        _PerCallDetector([[near], [near], [near, far]]),
        n,
        32,
        32,
        stamped=True,
        redetect_every_k=4,
        diagnostics=events,
        **kw,
    )
    by_cls = {m.cls: m for m in masklets}
    (backfill,) = [e for e in events if e["kind"] == "backfill"]
    return by_cls, backfill, pred


def test_backfill_recovers_a_car_before_its_first_detection():
    by_cls, backfill, pred = _late_car(
        lambda f: None if f == 0 else (20, 20, 28, 28),
        backfill_frames=20,
        drop_lost_after_frames=1,
    )
    assert by_cls["car"].frames_present == list(range(12))
    assert by_cls["truck"].frames_present == list(range(1, 12))
    assert all(_mask(by_cls["truck"], i).sum() == 64 for i in range(11))
    assert backfill["frames_added"] == 7 and backfill["stop"] == "lost"
    assert backfill["masklet_id"] == by_cls["truck"].masklet_id
    # The reverse pass reads the window backwards from the birth.
    assert pred.sessions[-1] == list(range(8, -1, -1))


def test_backfill_stops_after_its_span():
    by_cls, backfill, _ = _late_car(lambda f: (20, 20, 28, 28), backfill_frames=3)
    assert by_cls["truck"].frames_present == list(range(5, 12))
    assert backfill["stop"] == "span"


def test_backfill_stops_where_the_mask_runs_onto_a_tracked_object():
    """Before image 4 SAM2 latches the far car's id onto the near car, which
    its own track already covers there: those masks are not added."""
    by_cls, backfill, _ = _late_car(
        lambda f: (2, 2, 8, 8) if f < 4 else (20, 20, 28, 28), backfill_frames=20
    )
    assert by_cls["truck"].frames_present == list(range(4, 12))
    assert by_cls["car"].frames_present == list(range(12))
    assert backfill["stop"] == "tracked"


def test_a_birth_at_the_windows_first_image_is_not_backfilled():
    near = Detection(box_xyxy=(2, 2, 8, 8), cls="car", score=0.9)
    pred = _StampedPredictor(32, 32)
    events: list[dict] = []
    (m,) = _track(
        pred,
        _PerCallDetector([[near]]),
        6,
        32,
        32,
        stamped=True,
        redetect_every_k=4,
        backfill_frames=20,
        diagnostics=events,
    )
    assert m.frames_present == list(range(6))
    assert not [e for e in events if e["kind"] == "backfill"]
    assert len(pred.sessions) == 1  # no reverse session was opened
