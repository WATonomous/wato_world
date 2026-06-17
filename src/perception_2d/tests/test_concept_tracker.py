"""Tests for sam3_concept_tracker — grouping the predictor's per-frame outputs
into Masklets by (concept, obj_id), using a fake Sam3MultiplexVideoPredictor.
"""

from __future__ import annotations

import numpy as np
import pytest

from wato_perception_2d.io import CameraFrameInfo
from wato_perception_2d.models.sam3_concept_tracker import track_camera_concepts


def _outputs(n_obj: int, H: int, W: int) -> dict:
    masks = np.zeros((n_obj, H, W), dtype=bool)
    for i in range(n_obj):
        masks[i, 10:30, 10 + i * 25 : 30 + i * 25] = True
    return {
        "out_obj_ids": np.arange(1, n_obj + 1, dtype=np.int64),
        "out_probs": np.full(n_obj, 0.9, dtype=np.float32),
        "out_binary_masks": masks,
    }


class _FakeModel:
    """Stand-in for predictor.model — only init_state is exercised here.

    init_state mirrors the real multiplex signature (accepts
    offload_video_to_cpu) and records it so the wrapper's threading is tested.
    """

    def __init__(self) -> None:
        self.last_offload_video_to_cpu = None

    def init_state(
        self, resource_path, offload_video_to_cpu=False, async_loading_frames=False
    ):
        self.last_offload_video_to_cpu = offload_video_to_cpu
        return {"num_frames": len(resource_path)}


class _FakePredictor:
    """Minimal stand-in: 'car' yields 2 instances/frame, anything else yields 1."""

    def __init__(self, H: int, W: int, n_frames: int) -> None:
        self.H, self.W, self.n = H, W, n_frames
        self.cur_text = None
        self.model = _FakeModel()
        self._all_inference_states: dict = {}
        self.async_loading_frames = False

    def _n_obj(self) -> int:
        return 2 if self.cur_text == "car" else 1

    def handle_request(self, req: dict) -> dict:
        t = req["type"]
        if t == "reset_session":
            return {"is_success": True}
        if t == "add_prompt":
            self.cur_text = req["text"]
            return {"frame_index": 0, "outputs": _outputs(self._n_obj(), self.H, self.W)}
        if t == "close_session":
            return {}
        return {}

    def handle_stream_request(self, req: dict):
        for fi in range(self.n):
            yield {"frame_index": fi, "outputs": _outputs(self._n_obj(), self.H, self.W)}


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


def test_groups_instances_by_concept_and_obj_id(tmp_path):
    H, W, n = 64, 64, 3
    frames = _frames(n)
    images = [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(n)]
    predictor = _FakePredictor(H, W, n)

    masklets = track_camera_concepts(
        predictor,
        frames,
        images,
        concept_prompts=[("car", "car"), ("person", "pedestrian")],
        bag_id="bag0",
        chunk_id="chunk0",
        cam_id="cam_front",
        masks_2d_base_dir=str(tmp_path),
        dino_every_k=0,  # skip DINOv2 (no torch in test env)
        device="cpu",
        offload_video_to_cpu=True,
    )

    # The VRAM fix must reach init_state, not just sit in our wrapper.
    assert predictor.model.last_offload_video_to_cpu is True

    # car → 2 instances, pedestrian → 1 instance = 3 masklets.
    assert len(masklets) == 3
    by_cls = sorted(m.cls for m in masklets)
    assert by_cls == ["car", "car", "pedestrian"]

    for m in masklets:
        assert m.frames_present == [0, 1, 2]
        assert len(m.mask_paths) == 3
        assert m.tracker_backend == "sam3"
        assert m.score == pytest.approx(0.9)


def test_resizes_mask_when_shape_differs_from_frame(tmp_path):
    """SAM 3.1 can return masks at its internal (square image_size) resolution
    rather than the original frame size — most visibly on wide / panoramic
    cameras. Those masks must be resized to the frame, not silently dropped
    (which previously surfaced as '0 masklets' from a clean propagation).
    """
    from PIL import Image as PILImage

    img_H, img_W, n = 48, 96, 3  # wide (panoramic-ish) frame
    mask_H, mask_W = 64, 64  # square model resolution the predictor emits
    frames = _frames(n)
    images = [np.zeros((img_H, img_W, 3), dtype=np.uint8) for _ in range(n)]
    predictor = _FakePredictor(mask_H, mask_W, n)  # emits 64x64 masks

    masklets = track_camera_concepts(
        predictor,
        frames,
        images,
        concept_prompts=[("car", "car")],
        bag_id="bag0",
        chunk_id="chunk0",
        cam_id="cam_pano",
        masks_2d_base_dir=str(tmp_path),
        dino_every_k=0,
        device="cpu",
        offload_video_to_cpu=True,
    )

    # Detections are kept, not dropped: car -> 2 instances across 3 frames.
    assert len(masklets) == 2
    for m in masklets:
        assert m.frames_present == [0, 1, 2]
        # Saved masks land at the frame resolution, not the model's mask size.
        arr = np.array(PILImage.open(m.mask_paths[0]))
        assert arr.shape == (img_H, img_W)


class _OOM(RuntimeError):
    """Named to satisfy _is_cuda_oom (matches OutOfMemoryError by class name)."""


_OOM.__name__ = "OutOfMemoryError"


class _OOMOnLongClipPredictor(_FakePredictor):
    """Propagation OOMs once a session holds more than ``oom_above`` frames.

    Mirrors the real failure: the full clip OOMs mid-propagation, but a
    short-enough window (a fresh, smaller session) completes. ``cur_n`` tracks
    the frame count of the session currently open (set at start_session).
    """

    def __init__(self, H, W, n_frames, oom_above) -> None:
        super().__init__(H, W, n_frames)
        self.oom_above = oom_above
        self.cur_n = n_frames

    def handle_stream_request(self, req):
        if self.cur_n > self.oom_above:
            raise _OOM("CUDA out of memory")
        yield from super().handle_stream_request(req)


def test_oom_falls_back_to_sub_clips(tmp_path):
    H, W, n = 16, 16, 10
    frames = _frames(n)
    images = [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(n)]
    predictor = _OOMOnLongClipPredictor(H, W, n, oom_above=4)

    # init_state records the open session's frame count so the fake knows whether
    # the *current* window should OOM.
    orig_init = predictor.model.init_state

    def init_state(resource_path, **kw):
        predictor.cur_n = len(resource_path)
        predictor.n = len(resource_path)  # bound the per-window yield range
        return orig_init(resource_path, **kw)

    predictor.model.init_state = init_state

    masklets = track_camera_concepts(
        predictor,
        frames,
        images,
        concept_prompts=[("car", "car")],  # 2 instances/frame in the fake
        bag_id="bag0",
        chunk_id="chunk0",
        cam_id="cam_front",
        masks_2d_base_dir=str(tmp_path),
        dino_every_k=0,
        device="cpu",
        sub_clip_frames=4,  # 10 frames → windows [0:4], [4:8], [8:10]
    )

    # Full clip (10 > 4) OOMs, so windowing runs. Each window's object ids reset,
    # so "car" yields 2 masklets per window × 3 windows = 6.
    assert len(masklets) == 6
    assert {m.cls for m in masklets} == {"car"}
    # Each masklet spans only its window's frames (≤ 4), and frame ranges tile the
    # clip — proving absolute camera_seq is preserved across windows.
    covered = sorted(s for m in masklets for s in m.frames_present)
    assert covered == sorted([0, 1, 2, 3] * 2 + [4, 5, 6, 7] * 2 + [8, 9] * 2)


def _oom_predictor(H, W, n, oom_above):
    """An _OOMOnLongClipPredictor wired so each fresh session reports its own
    frame count (so the fake OOMs per-window, mirroring real propagation)."""
    predictor = _OOMOnLongClipPredictor(H, W, n, oom_above=oom_above)
    orig_init = predictor.model.init_state

    def init_state(resource_path, **kw):
        predictor.cur_n = len(resource_path)
        predictor.n = len(resource_path)
        return orig_init(resource_path, **kw)

    predictor.model.init_state = init_state
    return predictor


def test_oom_window_recursively_halves(tmp_path):
    # 8 frames, initial window 8, but a session only fits <= 2 frames. The full
    # clip OOMs, then each 8-frame window OOMs and is halved 8 -> 4 -> 2 until it
    # fits. Nothing is skipped: every frame ends up covered.
    H, W, n = 16, 16, 8
    frames = _frames(n)
    images = [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(n)]
    predictor = _oom_predictor(H, W, n, oom_above=2)

    masklets = track_camera_concepts(
        predictor,
        frames,
        images,
        concept_prompts=[("car", "car")],  # 2 instances/frame in the fake
        bag_id="bag0",
        chunk_id="chunk0",
        cam_id="cam_front",
        masks_2d_base_dir=str(tmp_path),
        dino_every_k=0,
        device="cpu",
        sub_clip_frames=8,      # one window for the whole clip, then halve it
        min_sub_clip_frames=2,  # floor the halving fits at
    )

    # Halving lands on 2-frame windows (4 of them); 2 instances each → 8 masklets,
    # and the frame ranges still tile the whole clip (no swath dropped).
    assert len(masklets) == 8
    covered = sorted(s for m in masklets for s in m.frames_present)
    assert covered == sorted([0, 1] * 2 + [2, 3] * 2 + [4, 5] * 2 + [6, 7] * 2)


def test_oom_at_floor_skips_window_without_raising(tmp_path):
    # Every session OOMs (oom_above=0), so even floor-sized windows fail. The
    # camera must degrade to zero masklets, not raise.
    H, W, n = 16, 16, 4
    frames = _frames(n)
    images = [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(n)]
    predictor = _oom_predictor(H, W, n, oom_above=0)

    masklets = track_camera_concepts(
        predictor,
        frames,
        images,
        concept_prompts=[("car", "car")],
        bag_id="bag0",
        chunk_id="chunk0",
        cam_id="cam_front",
        masks_2d_base_dir=str(tmp_path),
        dino_every_k=0,
        device="cpu",
        sub_clip_frames=4,
        min_sub_clip_frames=1,
    )

    assert masklets == []


def test_empty_inputs_return_empty(tmp_path):
    assert track_camera_concepts(
        _FakePredictor(8, 8, 0),
        [],
        [],
        concept_prompts=[("car", "car")],
        bag_id="b",
        chunk_id="c",
        cam_id="cam",
        masks_2d_base_dir=str(tmp_path),
        dino_every_k=0,
    ) == []
