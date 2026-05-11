"""Per-camera 2D temporal tracker.

Associates per-frame detections into masklets (temporally consistent tracks)
using IoU-based matching.  Extracts DINOv2 embeddings for each tracklet at
configurable intervals for downstream cross-camera ReID.

DEVA (the paper's choice for propagation) is optionally used if installed;
IoU matching is the reliable fallback that requires no extra dependencies.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from wato_perception_2d.segmenter import SegmentedDetection

log = logging.getLogger(__name__)

_dino_warned = False


@dataclass
class Masklet:
    """One temporally-associated object track within a single camera view."""

    masklet_id: str
    bag_id: str
    chunk_id: str
    cam_id: str
    cls: str
    score: float
    frames_present: list[int]    # camera_seq values where the mask exists
    mask_paths: list[str]        # parallel to frames_present
    dino_feature: Optional[np.ndarray] = None   # (D,) float32 DINOv2 embedding
    global_object_id: Optional[str] = None


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _extract_dino_feature(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    model_name: str = "dinov2_vitl14",
    device: str = "cpu",
) -> Optional[np.ndarray]:
    """Extract a DINOv2 embedding for the masked region.

    Returns (D,) float32 or None if torch/DINOv2 is unavailable.
    """
    global _dino_warned
    try:
        import torch
        import torchvision.transforms as T
        from PIL import Image as PILImage

        # Crop the bounding box of the mask.
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            return None
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        crop = image_rgb[r0:r1 + 1, c0:c1 + 1]

        model = torch.hub.load("facebookresearch/dinov2", model_name)
        model.eval().to(device)

        transform = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tensor = transform(PILImage.fromarray(crop)).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(tensor)[0].cpu().numpy().astype(np.float32)
        return feat
    except Exception as exc:  # noqa: BLE001
        if not _dino_warned:
            log.warning("DINOv2 unavailable (%s) — skipping feature extraction.", exc)
            _dino_warned = True
        return None


class Tracker2D:
    """IoU-based per-camera frame-to-frame tracker with DINOv2 embeddings.

    Processing model:
    - For each new frame, greedily match each active track's last mask to the
      new detections by IoU (Hungarian matching could replace this).
    - Unmatched detections start new tracklets.
    - Tracks that go N frames without a match are closed.
    - Every `dino_every_k` frames, extract a DINOv2 embedding for each active
      track and keep the mean embedding.
    """

    IOU_THRESHOLD = 0.25
    MAX_GAP_FRAMES = 3

    def __init__(
        self,
        bag_id: str,
        chunk_id: str,
        cam_id: str,
        masks_2d_base_dir: str,
        dino_model: str = "dinov2_vitl14",
        dino_every_k: int = 5,
        device: str = "cpu",
    ) -> None:
        self.bag_id = bag_id
        self.chunk_id = chunk_id
        self.cam_id = cam_id
        self.masks_2d_base_dir = masks_2d_base_dir
        self.dino_model = dino_model
        self.dino_every_k = dino_every_k
        self.device = device

        self._active: list[_ActiveTrack] = []
        self._closed: list[Masklet] = []
        self._frame_counter = 0

    def update(
        self,
        camera_seq: int,
        image_rgb: np.ndarray,
        seg_detections: list[SegmentedDetection],
    ) -> None:
        """Process one frame.  Call in camera_seq order."""
        self._frame_counter += 1

        if not seg_detections:
            # Age all active tracks.
            for trk in self._active:
                trk.gap += 1
            self._prune()
            return

        new_masks = [sd.mask for sd in seg_detections]
        n_new = len(new_masks)
        n_active = len(self._active)

        # Build IoU matrix: active × new.
        if n_active > 0:
            iou_mat = np.zeros((n_active, n_new), dtype=np.float32)
            for i, trk in enumerate(self._active):
                for j, mask in enumerate(new_masks):
                    iou_mat[i, j] = _iou(trk.last_mask, mask)

            matched_active = set()
            matched_new = set()
            # Greedy matching by descending IoU.
            for idx in np.argsort(iou_mat.ravel())[::-1]:
                i, j = divmod(int(idx), n_new)
                if iou_mat[i, j] < self.IOU_THRESHOLD:
                    break
                if i in matched_active or j in matched_new:
                    continue
                matched_active.add(i)
                matched_new.add(j)
                sd = seg_detections[j]
                mask_path = self._save_mask(
                    self._active[i].masklet_id, camera_seq, sd.mask
                )
                self._active[i].update(camera_seq, sd.mask, mask_path)

                if self._frame_counter % self.dino_every_k == 0:
                    feat = _extract_dino_feature(
                        image_rgb, sd.mask, self.dino_model, self.device
                    )
                    if feat is not None:
                        self._active[i].accumulate_dino(feat)
        else:
            matched_active = set()
            matched_new = set()

        for j, sd in enumerate(seg_detections):
            if j in matched_new:
                continue
            masklet_id = str(uuid.uuid4())[:12]
            mask_path = self._save_mask(masklet_id, camera_seq, sd.mask)
            trk = _ActiveTrack(
                masklet_id=masklet_id,
                bag_id=self.bag_id,
                chunk_id=self.chunk_id,
                cam_id=self.cam_id,
                cls=sd.detection.class_name,
                score=float(sd.detection.score),
                first_frame=camera_seq,
                last_mask=sd.mask,
                frames_present=[camera_seq],
                mask_paths=[mask_path],
            )
            self._active.append(trk)

        for i, trk in enumerate(self._active):
            if i not in matched_active:
                trk.gap += 1
        self._prune()

    def finalize(self) -> list[Masklet]:
        """Close all remaining active tracks and return all masklets."""
        for trk in self._active:
            self._closed.append(trk.to_masklet())
        self._active.clear()
        return list(self._closed)

    def _prune(self) -> None:
        still_active = []
        for trk in self._active:
            if trk.gap > self.MAX_GAP_FRAMES:
                self._closed.append(trk.to_masklet())
            else:
                still_active.append(trk)
        self._active = still_active

    def _save_mask(self, masklet_id: str, camera_seq: int, mask: np.ndarray) -> str:
        from PIL import Image as PILImage
        d = os.path.join(self.masks_2d_base_dir, masklet_id)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{camera_seq:06d}.png")
        PILImage.fromarray(mask.astype(np.uint8) * 255).save(path)
        return path


@dataclass
class _ActiveTrack:
    masklet_id: str
    bag_id: str
    chunk_id: str
    cam_id: str
    cls: str
    score: float
    first_frame: int
    last_mask: np.ndarray
    frames_present: list[int]
    mask_paths: list[str]
    gap: int = 0
    _dino_accum: list[np.ndarray] = field(default_factory=list)

    def update(self, seq: int, mask: np.ndarray, mask_path: str) -> None:
        self.last_mask = mask
        self.frames_present.append(seq)
        self.mask_paths.append(mask_path)
        self.score = max(self.score, 0.0)  # keep initial score
        self.gap = 0

    def accumulate_dino(self, feat: np.ndarray) -> None:
        self._dino_accum.append(feat)

    def to_masklet(self) -> Masklet:
        dino = (
            np.mean(self._dino_accum, axis=0).astype(np.float32)
            if self._dino_accum else None
        )
        return Masklet(
            masklet_id=self.masklet_id,
            bag_id=self.bag_id,
            chunk_id=self.chunk_id,
            cam_id=self.cam_id,
            cls=self.cls,
            score=self.score,
            frames_present=self.frames_present,
            mask_paths=self.mask_paths,
            dino_feature=dino,
        )
