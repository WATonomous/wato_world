"""SAM2 segmentation wrapper.

Given per-frame bounding boxes (from detector.py) and optional LiDAR point
prompts (cross-modal, SAM4D-style), produces per-detection binary masks.

Lazy-imports sam2 so the module can be imported without it installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from wato_perception_2d.detector import Detection

log = logging.getLogger(__name__)

_warned_missing = False


@dataclass
class SegmentedDetection:
    """Detection extended with a binary pixel mask."""

    detection: Detection
    mask: np.ndarray  # (H, W) bool


class SAM2Segmenter:
    """SAM2 wrapper for prompt-based segmentation.

    Accepts:
    - bounding-box prompts (from GroundingDINO)
    - optional point prompts (projected LiDAR dynamic points, SAM4D cross-modal)

    Falls back to filled bounding-box masks when sam2 is not installed.
    """

    def __init__(
        self,
        checkpoint: str = "sam2_hiera_large",
        device: Optional[str] = None,
    ) -> None:
        self._checkpoint = checkpoint
        self._device = device or self._default_device()
        self._predictor = None  # lazy-loaded

    @staticmethod
    def _default_device() -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load(self) -> bool:
        global _warned_missing
        if self._predictor is not None:
            return True
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            model = build_sam2(self._checkpoint, device=self._device)
            self._predictor = SAM2ImagePredictor(model)
            log.info("SAM2 loaded (%s) on %s", self._checkpoint, self._device)
            return True
        except Exception as exc:  # noqa: BLE001
            if not _warned_missing:
                log.warning(
                    "SAM2 unavailable (%s) — using bbox-fill fallback masks. "
                    "Install: pip install sam2",
                    exc,
                )
                _warned_missing = True
            return False

    def segment(
        self,
        image_rgb: np.ndarray,
        detections: list[Detection],
        lidar_point_prompts: Optional[np.ndarray] = None,
    ) -> list[SegmentedDetection]:
        """Segment each detection in image_rgb.

        Args:
            image_rgb: (H, W, 3) uint8 RGB image.
            detections: list of Detection with bbox_xyxy in pixels.
            lidar_point_prompts: optional (M, 2) float32 pixel coordinates of
                projected LiDAR dynamic points (all treated as foreground hints).

        Returns list of SegmentedDetection (same order as detections).
        """
        if not detections:
            return []

        H, W = image_rgb.shape[:2]

        if not self._load():
            return self._bbox_fill_fallback(detections, H, W)

        self._predictor.set_image(image_rgb)

        results: list[SegmentedDetection] = []
        for det in detections:
            boxes = det.bbox_xyxy[None].astype(np.float32)  # (1, 4)

            # Merge LiDAR points into the point prompt set for this detection.
            point_coords: Optional[np.ndarray] = None
            point_labels: Optional[np.ndarray] = None
            if lidar_point_prompts is not None and lidar_point_prompts.shape[0] > 0:
                # Keep only points inside this box.
                x1, y1, x2, y2 = det.bbox_xyxy
                inside = (
                    (lidar_point_prompts[:, 0] >= x1)
                    & (lidar_point_prompts[:, 0] <= x2)
                    & (lidar_point_prompts[:, 1] >= y1)
                    & (lidar_point_prompts[:, 1] <= y2)
                )
                pts = lidar_point_prompts[inside]
                if pts.shape[0] > 0:
                    point_coords = pts[:, :2].astype(np.float32)
                    point_labels = np.ones(pts.shape[0], dtype=np.int32)

            masks_t, scores_t, _ = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=boxes,
                multimask_output=False,
            )
            # masks_t: (1, H, W) bool tensor
            mask = (
                masks_t[0].cpu().numpy().astype(bool)
                if hasattr(masks_t, "cpu")
                else masks_t[0].astype(bool)
            )
            results.append(SegmentedDetection(detection=det, mask=mask))

        return results

    @staticmethod
    def _bbox_fill_fallback(
        detections: list[Detection], H: int, W: int
    ) -> list[SegmentedDetection]:
        """Fill the bounding box rectangle as the mask (used when SAM2 is absent)."""
        results: list[SegmentedDetection] = []
        for det in detections:
            mask = np.zeros((H, W), dtype=bool)
            x1, y1, x2, y2 = det.bbox_xyxy.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            mask[y1:y2, x1:x2] = True
            results.append(SegmentedDetection(detection=det, mask=mask))
        return results
