"""Forward-only SAM2 video API adapter around the authors' DAM4SAM tracker.

Each live object has an official tracker context with its own birth-relative
clock. Contexts share weights and one frame's image features. A correction
creates a fresh context for that object; no stale memory survives it. Repeated
keyframe yields reuse logits and do not advance the memory twice.

The memory selection algorithm lives in _vendor/d4sm, not in this adapter. The
adapter sets two of its parameters per session and records SAM2's own quality
outputs, without editing the vendored files:

* ``update_delta`` is DAM4SAM's memory update spacing Δ in images. The authors
  use 5 at 30 fps and 1 at 5 fps (DAM4SAM extended, arXiv 2509.13864, §3.2.1 /
  §5.5): one update every ~170-200 ms. Their code default (5) stays the
  adapter default.
* ``memory_min_predicted_iou`` ports SAMURAI's memory-bank gate (arXiv
  2411.11922; code ``memory_bank_iou_threshold`` 0.5 with object score > 0, at
  commit 76ba195): the previous frame is always attended to, but only a frame
  whose predicted IoU exceeds the gate is kept as an older recent-appearance
  (RAM) memory. An object score <= 0 already empties SAM2's mask, and DAM4SAM
  never stores an empty frame, so the IoU is the only test left to add. Object
  pointers are not gated (SAMURAI does not gate them), and the distractor
  memory keeps the authors' stricter IoU > 0.8 gate. 0 disables.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from wato_perception_2d._vendor.d4sm.tracking_wrapper_mot import DAM4SAMMOT

# The authors' code defaults (tracking_wrapper_mot.py): Δ = 5, no RAM gate.
AUTHOR_UPDATE_DELTA = 5


class _StepView:
    """The shared SAM2 model as one context sees it.

    Every attribute is the model's; ``track_step`` also hands its output to the
    context, after memory attention and before the vendored memory update.
    """

    def __init__(self, model, on_step):
        self._model = model
        self._on_step = on_step

    def __getattr__(self, name):
        return getattr(self._model, name)

    def track_step(self, **kwargs):
        out = self._model.track_step(**kwargs)
        self._on_step(out)
        return out


class _Context(DAM4SAMMOT):
    def __init__(self, model, shared, *, update_delta, memory_min_predicted_iou):
        super().__init__(sam=_StepView(model, self._stepped), device=model.device)
        self._shared = shared
        self.update_delta = update_delta
        self._min_iou = memory_min_predicted_iou
        # (predicted IoU of the chosen mask, object-score logit) of the newest
        # frame, and how many frames the RAM gate has refused.
        self.quality: tuple[float, float] | None = None
        self.gated_frames = 0

    def _prepare_image(self, image):
        return self._shared["prepared"]

    def _get_features(self, image, num_obj=1):
        if num_obj != 1:
            raise ValueError("DAM4SAM contexts contain exactly one object")
        return self._shared["features"]

    def _stepped(self, out):
        self.quality = (
            float(out["ious"].detach().float().max()),
            float(out["object_score_logits"].detach().float().max()),
        )
        # The previous frame was this frame's last-frame memory. If it failed
        # the gate it leaves RAM now, as if the vendored update had replaced it
        # with this frame instead of keeping it. A frame promoted to the
        # distractor memory is swapped by the vendored update itself.
        memory = self.per_object_outputs_all.get(1)
        if memory and not self.add_to_drm_next.get(1):
            memory[:] = [m for m in memory if not m.get("below_memory_gate")]

    def track(self, image):
        out = super().track(image)
        if self._min_iou > 0 and self.quality[0] <= self._min_iou:
            for m in self.per_object_outputs_all[1]:
                if (
                    m["frame_idx"] == self.frame_index
                    and not m["is_init"]
                    and not m["is_drm"]
                ):
                    m["below_memory_gate"] = True
                    self.gated_frames += 1
        return out


@dataclass
class _Object:
    context: DAM4SAMMOT
    frame: int
    logits: torch.Tensor


class DAM4SAMPredictor:
    """The subset of SAM2VideoPredictor used by track_camera.

    Frame JPEGs are read lazily on CPU; the memory banks stay on the model's
    device, as in the authors' implementation. Reverse propagation and arbitrary
    seeking are deliberately rejected rather than silently altering memory;
    track_camera's back-fill feeds earlier frames forward in reverse order.
    """

    def __init__(
        self,
        model,
        *,
        update_delta: int = AUTHOR_UPDATE_DELTA,
        memory_min_predicted_iou: float = 0.0,
    ):
        if update_delta < 1:
            raise ValueError("DAM4SAM update_delta must be >= 1")
        self.image_model = model
        self.device = model.device
        self.update_delta = update_delta
        self.memory_min_predicted_iou = memory_min_predicted_iou
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
        context = _Context(
            self.image_model,
            state["shared"],
            update_delta=self.update_delta,
            memory_min_predicted_iou=self.memory_min_predicted_iou,
        )
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

    def frame_quality(self, inference_state, obj_id):
        """SAM2's own confidence in the object's newest mask: the chosen mask's
        predicted IoU and the object-score logit (> 0: present). Published
        trackers gate memory and end tracks on these (SAMURAI, SAM2Long,
        Seg2Track-SAM2, SAM2MOT); they are model confidence, not accuracy."""
        quality = inference_state["objects"][obj_id].context.quality
        if quality is None:
            return {}
        return {"predicted_iou": quality[0], "object_score_logit": quality[1]}

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
            "gated_frames": context.gated_frames,
        }


def build_dam4sam(
    config_file,
    checkpoint,
    device,
    *,
    update_delta: int = AUTHOR_UPDATE_DELTA,
    memory_min_predicted_iou: float = 0.0,
):
    from wato_perception_2d._vendor.d4sm.tracking_wrapper_mot import build_sam

    model = build_sam(
        config_file,
        checkpoint,
        device=device,
        hydra_overrides_extra=[
            "++model._target_=wato_perception_2d._vendor.d4sm.sam2_base.SAM2Base"
        ],
    )
    return DAM4SAMPredictor(
        model,
        update_delta=update_delta,
        memory_min_predicted_iou=memory_min_predicted_iou,
    )
