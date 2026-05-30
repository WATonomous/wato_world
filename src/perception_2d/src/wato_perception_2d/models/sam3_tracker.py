"""SAM 3.1 video tracker wrapper — the "sam3" tracker backend.

Uses SAM 3.1's Object Multiplex update for multi-object temporal tracking
within a single camera stream.  Raises RuntimeError when SAM3 is not installed:
there is NO automatic fallback.  To run the IoU tracker instead, select it
explicitly with tracker.backend: "deva" (Tracker2D).

The SAM3Tracker is stateful across frames: call reset() between camera streams
or chunks.  finalize() closes all active tracks and returns Masklet objects
compatible with the rest of the pipeline.

DINOv2 appearance embeddings are extracted every `dino_every_k` frames via
embeddings.py (for downstream re-identification; not used for tracking here).
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Optional

import numpy as np

from wato_perception_2d.fusion.tracker_2d import Masklet
from wato_perception_2d.models.embeddings import extract_dino_feature
from wato_perception_2d.models.segmenter import SegmentedDetection

log = logging.getLogger(__name__)


class SAM3Tracker:
    """SAM 3.1 Object Multiplex temporal tracker for one camera stream.

    Usage:
        tracker = SAM3Tracker(checkpoint="facebook/sam3.1", device=None)
        tracker.reset()  # call before each new camera stream
        for frame in camera_frames:
            seg_dets = segmenter.segment(image, proposals)
            tracker.update(frame.camera_seq, image, seg_dets)
        masklets = tracker.finalize()

    Raises RuntimeError when SAM3 is unavailable — no IoU fallback (use
    tracker.backend: "deva" for the IoU tracker explicitly).
    """

    def __init__(
        self,
        bag_id: str,
        chunk_id: str,
        cam_id: str,
        masks_2d_base_dir: str,
        checkpoint: str = "facebook/sam3.1",
        device: Optional[str] = None,
        dino_model: str = "dinov2_vitl14",
        dino_every_k: int = 5,
    ) -> None:
        self.bag_id = bag_id
        self.chunk_id = chunk_id
        self.cam_id = cam_id
        self.masks_2d_base_dir = masks_2d_base_dir
        self._checkpoint = checkpoint
        self._device = device or self._default_device()
        self.dino_model = dino_model
        self.dino_every_k = dino_every_k

        self._video_predictor = None  # lazy-loaded
        self._state = None  # SAM3 inference state (per stream)
        self._track_meta: dict[int, _TrackMeta] = {}  # sam3_obj_id → metadata
        self._frame_counter = 0

    @staticmethod
    def _default_device() -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load(self) -> None:
        if self._video_predictor is not None:
            return
        try:
            from sam3.build_sam import build_sam3_video_predictor

            self._video_predictor = build_sam3_video_predictor(
                self._checkpoint, device=self._device
            )
            log.info("SAM3Tracker loaded (%s) on %s", self._checkpoint, self._device)
        except Exception as exc:
            raise RuntimeError(
                f"SAM 3.1 video predictor unavailable (checkpoint {self._checkpoint!r}): "
                f"{exc}. Install 'sam3', or set tracker.backend: 'deva' to use the IoU "
                "tracker (Tracker2D) explicitly. perception_2d does not auto-fall-back."
            ) from exc

    def reset(self) -> None:
        """Reset state for a new camera stream.  Call before each camera."""
        self._state = None
        self._track_meta = {}
        self._frame_counter = 0

    def update(
        self,
        camera_seq: int,
        image_rgb: np.ndarray,
        seg_detections: list[SegmentedDetection],
    ) -> None:
        """Process one frame.  Call in camera_seq order.

        Raises RuntimeError if SAM3 is unavailable; SAM3 step errors propagate.
        """
        self._frame_counter += 1
        self._load()
        self._sam3_update(camera_seq, image_rgb, seg_detections)

    def _sam3_update(
        self,
        camera_seq: int,
        image_rgb: np.ndarray,
        seg_detections: list[SegmentedDetection],
    ) -> None:
        """SAM 3.1 Object Multiplex tracking step.

        FIXME(sam3-api): this streaming flow (re-register every detection each
        frame, then propagate a single frame) does NOT perform real cross-frame
        association — `obj_id = id(sd)` is a fresh id every frame, so each
        detection becomes a brand-new track.  A correct implementation must
        carry a stable per-object id across frames (matching new detections to
        existing tracks) and use SAM 3.1's propagation to maintain identity.
        Fixing this requires pinning the actual SAM 3.1 video-predictor API
        (init_state / add_new_mask / propagate_in_video signatures are
        currently unverified — the package is not installed in CI).  Until then
        the IoU fallback (Tracker2D) is the path that produces correct masklets.
        """
        if self._state is None:
            self._state = self._video_predictor.init_state(images_dir=None)

        # Register new detections as tracked objects.
        for sd in seg_detections:
            obj_id = id(sd)  # FIXME(sam3-api): not stable across frames — see above
            self._video_predictor.add_new_mask(
                inference_state=self._state,
                frame_idx=camera_seq,
                obj_id=obj_id,
                mask=sd.mask,
            )
            if obj_id not in self._track_meta:
                masklet_id = str(uuid.uuid4())[:12]
                self._track_meta[obj_id] = _TrackMeta(
                    masklet_id=masklet_id,
                    bag_id=self.bag_id,
                    chunk_id=self.chunk_id,
                    cam_id=self.cam_id,
                    cls=sd.phrase,
                    score=sd.sam3_score,
                )

        # Propagate all tracked objects forward.
        for obj_id, masks, scores in self._video_predictor.propagate_in_video(
            self._state, start_frame_idx=camera_seq, max_frame_num_to_track=1
        ):
            if obj_id not in self._track_meta:
                continue
            meta = self._track_meta[obj_id]
            mask = (
                masks[0].cpu().numpy().astype(bool)
                if hasattr(masks, "cpu")
                else masks[0].astype(bool)
            )
            mask_path = self._save_mask(meta.masklet_id, camera_seq, mask)
            meta.frames_present.append(camera_seq)
            meta.mask_paths.append(mask_path)

            if self._frame_counter % self.dino_every_k == 0:
                feat = extract_dino_feature(
                    image_rgb, mask, self.dino_model, self._device
                )
                if feat is not None:
                    meta.dino_accum.append(feat)

    def finalize(self) -> list[Masklet]:
        """Close all SAM3-tracked objects and return Masklet objects."""
        closed: list[Masklet] = []
        for meta in self._track_meta.values():
            if not meta.frames_present:
                continue
            dino = (
                np.mean(meta.dino_accum, axis=0).astype(np.float32)
                if meta.dino_accum
                else None
            )
            closed.append(
                Masklet(
                    masklet_id=meta.masklet_id,
                    bag_id=meta.bag_id,
                    chunk_id=meta.chunk_id,
                    cam_id=meta.cam_id,
                    cls=meta.cls,
                    score=meta.score,
                    frames_present=meta.frames_present,
                    mask_paths=meta.mask_paths,
                    dino_feature=dino,
                )
            )
        self._track_meta = {}
        return closed

    def _save_mask(self, masklet_id: str, camera_seq: int, mask: np.ndarray) -> str:
        from PIL import Image as PILImage

        d = os.path.join(self.masks_2d_base_dir, masklet_id)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{camera_seq:06d}.png")
        PILImage.fromarray(mask.astype(np.uint8) * 255).save(path)
        return path


class _TrackMeta:
    """Internal metadata for one SAM3-tracked object."""

    __slots__ = (
        "masklet_id",
        "bag_id",
        "chunk_id",
        "cam_id",
        "cls",
        "score",
        "frames_present",
        "mask_paths",
        "dino_accum",
    )

    def __init__(
        self,
        masklet_id: str,
        bag_id: str,
        chunk_id: str,
        cam_id: str,
        cls: str,
        score: float,
    ) -> None:
        self.masklet_id = masklet_id
        self.bag_id = bag_id
        self.chunk_id = chunk_id
        self.cam_id = cam_id
        self.cls = cls
        self.score = score
        self.frames_present: list[int] = []
        self.mask_paths: list[str] = []
        self.dino_accum: list[np.ndarray] = []
