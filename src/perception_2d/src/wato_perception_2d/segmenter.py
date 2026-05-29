"""Segmentation backends for perception_2d.

Given per-frame bounding boxes (from detector.py) and optional LiDAR point
prompts (cross-modal, SAM4D-style), produces per-detection binary masks.

Backends
--------
SAM2Segmenter  — Meta SAM 2 (production default).
SAM3Segmenter  — Meta SAM 3 stub; fill in _load() / segment() once SAM3 ships.

Use build_segmenter(backend, checkpoint) to select at runtime via config.

All backends lazy-import their model libraries so this module can be imported
even when the heavy ML packages are not installed. Missing packages fall back
to bbox-fill masks with a one-time warning.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from wato_perception_2d.detector import Detection

log = logging.getLogger(__name__)

_sam2_warned_missing = False
_sam3_warned_missing = False


@dataclass
class SegmentedDetection:
    """Detection extended with a binary pixel mask."""

    detection: Detection
    mask: np.ndarray  # (H, W) bool


class BaseSegmenter(ABC):
    """Common interface for all segmentation backends."""

    @abstractmethod
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

    @staticmethod
    def _bbox_fill_fallback(
        detections: list[Detection], H: int, W: int
    ) -> list[SegmentedDetection]:
        """Fill the bounding box rectangle as the mask (used when model is absent)."""
        results: list[SegmentedDetection] = []
        for det in detections:
            mask = np.zeros((H, W), dtype=bool)
            x1, y1, x2, y2 = det.bbox_xyxy.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            mask[y1:y2, x1:x2] = True
            results.append(SegmentedDetection(detection=det, mask=mask))
        return results

    @staticmethod
    def _default_device() -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


class SAM2Segmenter(BaseSegmenter):
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

    def _load(self) -> bool:
        global _sam2_warned_missing
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
            if not _sam2_warned_missing:
                log.warning(
                    "SAM2 unavailable (%s) — using bbox-fill fallback. "
                    "Install: pip install sam2",
                    exc,
                )
                _sam2_warned_missing = True
            return False

    def segment(
        self,
        image_rgb: np.ndarray,
        detections: list[Detection],
        lidar_point_prompts: Optional[np.ndarray] = None,
    ) -> list[SegmentedDetection]:
        if not detections:
            return []

        H, W = image_rgb.shape[:2]

        if not self._load():
            return self._bbox_fill_fallback(detections, H, W)

        self._predictor.set_image(image_rgb)

        results: list[SegmentedDetection] = []
        for det in detections:
            boxes = det.bbox_xyxy[None].astype(np.float32)  # (1, 4)

            point_coords: Optional[np.ndarray] = None
            point_labels: Optional[np.ndarray] = None
            if lidar_point_prompts is not None and lidar_point_prompts.shape[0] > 0:
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
            mask = (
                masks_t[0].cpu().numpy().astype(bool)
                if hasattr(masks_t, "cpu")
                else masks_t[0].astype(bool)
            )
            results.append(SegmentedDetection(detection=det, mask=mask))

        return results


class SAM3Segmenter(BaseSegmenter):
    """SAM3 backend stub.

    Structure mirrors SAM2Segmenter exactly. Fill in _load() and the
    predict() call in segment() once SAM3's package and API are known.
    """

    def __init__(
        self,
        checkpoint: str = "sam3_hiera_large",
        device: Optional[str] = None,
    ) -> None:
        self._checkpoint = checkpoint
        self._device = device or self._default_device()
        self._predictor = None  # lazy-loaded

    def _load(self) -> bool:
        global _sam3_warned_missing
        if self._predictor is not None:
            return True
        try:
            # TODO: replace with actual SAM3 imports when the package ships.
            from sam3.build_sam import build_sam3  # type: ignore[import]  # noqa: F401
            from sam3.sam3_image_predictor import SAM3ImagePredictor  # type: ignore[import]  # noqa: F401

            raise NotImplementedError(
                "SAM3 _load() stub — wire up actual build_sam3 + SAM3ImagePredictor calls here"
            )
        except ImportError:
            if not _sam3_warned_missing:
                log.warning(
                    "SAM3 unavailable — using bbox-fill fallback. "
                    "Install sam3 when it ships and implement _load()."
                )
                _sam3_warned_missing = True
            return False

    def segment(
        self,
        image_rgb: np.ndarray,
        detections: list[Detection],
        lidar_point_prompts: Optional[np.ndarray] = None,
    ) -> list[SegmentedDetection]:
        if not detections:
            return []

        H, W = image_rgb.shape[:2]

        if not self._load():
            return self._bbox_fill_fallback(detections, H, W)

        # TODO: fill in SAM3 predictor.set_image() + predict() call once API is known.
        raise NotImplementedError(
            "SAM3 segment() stub — implement predict() call when SAM3 API is available"
        )


def build_segmenter(
    backend: str,
    checkpoint: str,
    device: Optional[str] = None,
) -> BaseSegmenter:
    """Factory: return the correct segmenter for the configured backend.

    Args:
        backend: "sam2" or "sam3".
        checkpoint: model checkpoint name or path.
        device: "cuda", "cpu", or None (auto-detect).
    """
    if backend == "sam3":
        return SAM3Segmenter(checkpoint=checkpoint, device=device)
    return SAM2Segmenter(checkpoint=checkpoint, device=device)
