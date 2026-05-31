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
    """Stand-in for predictor.model — only init_state is exercised here."""

    def init_state(self, resource_path, async_loading_frames=False):
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
    )

    # car → 2 instances, pedestrian → 1 instance = 3 masklets.
    assert len(masklets) == 3
    by_cls = sorted(m.cls for m in masklets)
    assert by_cls == ["car", "car", "pedestrian"]

    for m in masklets:
        assert m.frames_present == [0, 1, 2]
        assert len(m.mask_paths) == 3
        assert m.tracker_backend == "sam3"
        assert m.score == pytest.approx(0.9)


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
