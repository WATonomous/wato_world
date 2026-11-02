"""Tests for sam2_tracker — detect on keyframes, box-prompt SAM2, propagate, and
group per-object masks into Masklets — using a fake SAM2 video predictor and a
fake detector so no `sam2` / `transformers` install or GPU is required.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image as PILImage

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


def _frames(n: int) -> list[CameraFrameInfo]:
    return [
        CameraFrameInfo(
            frame_id=f"f{i}",
            bag_id="bag0",
            chunk_id="chunk0",
            sweep_id=i,
            cam_id="cam_front",
            image_path=f"/tmp/{i}.png",
            camera_seq=i,
            camera_timestamp_ns=None,
            valid_camera=True,
        )
        for i in range(n)
    ]


def _track(predictor, detector, n, H, W, **kw):
    base = dict(
        bag_id="bag0",
        chunk_id="chunk0",
        cam_id="cam_front",
        dino_model="dinov2_vitl14",
        dino_every_k=0,  # skip DINOv2 (no torch in test env)
        device="cpu",
        # Detector support and re-prompting off unless a test turns them on,
        # so each test exercises its own rule.
        birth_score=0.0,
        support_keyframes=0,
        reprompt_below_iou=0.0,
    )
    base.update(kw)
    images = [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(n)]
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


def test_track_that_jumps_onto_another_tracked_object_ends():
    """An oncoming car's mask snaps onto a parked car with no empty frame in
    between: the track that jumped ends, the parked car's carries on."""
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
    assert by_score[0.8] == list(range(7))


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
# Pixel ownership
# ---------------------------------------------------------------------------


def test_overlapping_masks_give_each_pixel_to_one_object():
    """A bus's mask wraps around a car half-hidden beside it. Each pixel goes
    to the object SAM2 is surest of there, or the car's LiDAR points would be
    lifted as both."""
    H, W, n = 32, 32, 2
    bus = Detection(box_xyxy=(2, 2, 20, 20), cls="bus", score=0.9)
    car = Detection(box_xyxy=(16, 10, 24, 18), cls="car", score=0.8)
    pred = _FakePredictor(H, W, strength={(16, 10, 24, 18): 3.0})
    masklets = _track(
        pred, _PerCallDetector([[bus, car]]), n, H, W, redetect_every_k=100
    )
    m = {mk.cls: _mask(mk) for mk in masklets}
    assert not (m["bus"] & m["car"]).any()
    assert m["car"].sum() == 8 * 8
    assert m["bus"].sum() == 18 * 18 - 4 * 8


def test_track_left_owning_almost_none_of_its_mask_ends():
    """A lost pedestrian's id latched onto a car that holds those pixels more
    surely: it keeps a 1-row sliver (3% of its mask), far below
    duplicate_mask_iou with the car. It is never recorded, and ends after
    collapse_after_frames."""
    H, W, n = 64, 64, 5
    car = Detection(box_xyxy=(10, 10, 50, 50), cls="car", score=0.9)
    ped = Detection(box_xyxy=(20, 20, 30, 51), cls="pedestrian", score=0.5)
    pred = _FakePredictor(H, W, strength={(10, 10, 50, 50): 5.0})
    masklets = _track(
        pred,
        _PerCallDetector([[car, ped]]),
        n,
        H,
        W,
        redetect_every_k=3,
        collapse_after_frames=3,
    )
    assert [(m.cls, m.frames_present) for m in masklets] == [("car", [0, 1, 2, 3, 4])]
    assert pred.removed == [1]  # removed at keyframe 3


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


def test_contested_specks_go_to_the_runner_up():
    """Where two masks' logits are nearly equal, B wins isolated pixels inside
    A's region. Dropped as B's specks, they go back to A instead of becoming
    holes."""
    from wato_perception_2d.models.sam2_tracker import _owned_masks

    H, W = 40, 40
    a = np.zeros((H, W), dtype=bool)
    a[:20] = True
    b = np.zeros((H, W), dtype=bool)
    b[10:] = True
    logits = np.full((2, 1, H, W), -5.0, dtype=np.float32)
    logits[0, 0][a] = 2.0
    logits[1, 0][b] = 1.9
    logits[1, 0, 10:18:4, ::4] = 2.1  # B narrowly wins a sparse grid in A's rows

    owned, collapsed = _owned_masks(
        logits, [0, 1], [(0, a), (1, b)], island_fraction=0.1, collapse_fraction=0.2
    )
    got = dict(owned)
    assert collapsed == []
    assert (got[0] == a).all()
    assert (got[1] == (b & ~a)).all()


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
# Containment: what is seen through a vehicle's windows is part of it
# ---------------------------------------------------------------------------

_CONTAINED = dict(contained_classes=("pedestrian",), container_classes=("car",))


def test_driver_seen_through_the_windows_is_not_a_pedestrian():
    """The driver's mask lies inside the car's and ends well above its bottom:
    never recorded, ended, and its pixels stay the car's."""
    H, W, n = 64, 64, 5
    car = Detection(box_xyxy=(10, 10, 50, 50), cls="car", score=0.9)
    driver = Detection(box_xyxy=(25, 18, 35, 30), cls="pedestrian", score=0.6)
    pred = _FakePredictor(H, W, strength={(25, 18, 35, 30): 5.0})
    (car_m,) = _track(
        pred,
        _PerCallDetector([[car, driver]]),
        n,
        H,
        W,
        redetect_every_k=3,
        containment_after_frames=3,
        **_CONTAINED,
    )
    assert car_m.cls == "car" and car_m.frames_present == [0, 1, 2, 3, 4]
    assert _mask(car_m).sum() == 40 * 40
    assert pred.removed == [1]


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
        **_CONTAINED,
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
