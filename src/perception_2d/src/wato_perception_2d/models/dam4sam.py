"""Forward-only SAM2 video API adapter around the authors' DAM4SAM tracker.

Each live object has an official tracker context with its own birth-relative
clock. Contexts share weights and one frame's image features. A correction
creates a fresh context for that object; no stale memory survives it. Repeated
keyframe yields reuse logits and do not advance the memory twice.

The memory selection algorithm lives in _vendor/d4sm, not in this adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from wato_perception_2d._vendor.d4sm.tracking_wrapper_mot import DAM4SAMMOT


class _Context(DAM4SAMMOT):
    def __init__(self, model, shared):
        super().__init__(sam=model, device=model.device)
        self._shared = shared

    def _prepare_image(self, image):
        return self._shared["prepared"]

    def _get_features(self, image, num_obj=1):
        if num_obj != 1:
            raise ValueError("DAM4SAM contexts contain exactly one object")
        return self._shared["features"]


@dataclass
class _Object:
    context: DAM4SAMMOT
    frame: int
    logits: torch.Tensor


class DAM4SAMPredictor:
    """The subset of SAM2VideoPredictor used by track_camera.

    Frame JPEGs are read lazily on CPU; the memory banks stay on the model's
    device, as in the authors' implementation. Reverse propagation and arbitrary
    seeking are deliberately rejected rather than silently altering memory.
    """

    def __init__(self, model):
        self.image_model = model
        self.device = model.device
        self._preprocessor = DAM4SAMMOT(sam=model, device=self.device)

    def init_state(self, video_path, **kwargs):
        paths = sorted(Path(video_path).glob("*.jpg"), key=lambda p: int(p.stem))
        if not paths:
            raise ValueError("DAM4SAM needs at least one JPEG frame")
        return {"paths": paths, "objects": {}, "shared": {}, "frame": None}

    @torch.inference_mode()
    def _prepare(self, state, frame):
        if not 0 <= frame < len(state["paths"]):
            raise IndexError(frame)
        if state["frame"] != frame:
            with Image.open(state["paths"][frame]) as source:
                image = source.convert("RGB")
            prepared = self._preprocessor._prepare_image(image)
            features = self._preprocessor._get_features(prepared.unsqueeze(0))
            state["shared"].update(image=image, prepared=prepared, features=features)
            state["frame"] = frame
        return state["shared"]["image"]

    @torch.inference_mode()
    def _seed(self, state, frame, obj_id, region):
        image = self._prepare(state, frame)
        if any(obj.frame > frame for obj in state["objects"].values()):
            raise ValueError("DAM4SAM corrections must be on the current frame")
        context = _Context(self.image_model, state["shared"])
        context.initialize(image, [region])
        logits = F.interpolate(
            context.per_object_outputs_all[1][0]["pred_masks"],
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )
        state["objects"][obj_id] = _Object(context, frame, logits)
        # Callers use prompts for their state update; expose the prompted object.
        return frame, [obj_id], logits

    def add_new_points_or_box(self, inference_state, frame_idx, obj_id, box):
        x1, y1, x2, y2 = [float(v) for v in box]
        return self._seed(
            inference_state, frame_idx, obj_id, {"bbox": [x1, y1, x2 - x1, y2 - y1]}
        )

    def add_new_mask(self, inference_state, frame_idx, obj_id, mask):
        return self._seed(inference_state, frame_idx, obj_id, {"mask": mask})

    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=0,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        if reverse:
            raise ValueError("DAM4SAM adapter supports forward propagation only")
        objects = inference_state["objects"]
        if not objects:
            raise ValueError("No objects to propagate")
        stop = len(inference_state["paths"]) - 1
        if max_frame_num_to_track is not None:
            stop = min(stop, start_frame_idx + max_frame_num_to_track)
        for frame in range(start_frame_idx, stop + 1):
            with torch.inference_mode():
                if any(
                    o.frame > frame or o.frame < frame - 1 for o in objects.values()
                ):
                    raise ValueError("DAM4SAM propagation must visit frames in order")
                image = self._prepare(inference_state, frame)
                for obj in objects.values():
                    if obj.frame != frame:
                        obj.logits = obj.context.track(image)["mask_logits"]
                        obj.frame = frame
                ids = list(objects)
                logits = torch.cat([objects[oid].logits for oid in ids], dim=0)
            yield frame, ids, logits

    def remove_object(self, inference_state, obj_id, strict=False, need_output=False):
        if need_output:
            raise ValueError("DAM4SAM removal supports need_output=False")
        if strict and obj_id not in inference_state["objects"]:
            raise KeyError(obj_id)
        inference_state["objects"].pop(obj_id, None)

    def reset_state(self, inference_state):
        inference_state["objects"].clear()
        inference_state["shared"].clear()
        inference_state["frame"] = None

    def memory_diagnostics(self, inference_state, obj_id):
        """Small policy trace; no tensors or model confidence masquerading as truth."""
        context = inference_state["objects"][obj_id].context
        memory = context.per_object_outputs_all[1]
        return {
            "relative_frame": context.frame_index,
            "ram_frames": [
                m["frame_idx"] for m in memory if not m["is_init"] and not m["is_drm"]
            ],
            "drm_frames": [m["frame_idx"] for m in memory if m["is_drm"]],
            "pointer_count": len(context.per_object_obj_ptr[1]),
        }


def build_dam4sam(config_file, checkpoint, device):
    from wato_perception_2d._vendor.d4sm.tracking_wrapper_mot import build_sam

    model = build_sam(
        config_file,
        checkpoint,
        device=device,
        hydra_overrides_extra=[
            "++model._target_=wato_perception_2d._vendor.d4sm.sam2_base.SAM2Base"
        ],
    )
    return DAM4SAMPredictor(model)
