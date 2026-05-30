"""SAM3 text-prompted segmentation wrapper.

Given per-frame noun phrases (from discovery.py / Florence-2) and optional
LiDAR point prompts (cross-modal, SAM4D-style), produces per-phrase binary
masks with SAM3's presence-token scores.

SAM3 is text-prompted: each phrase is passed directly as a text query rather
than requiring a bounding box from a separate detector.  The presence-token
score discriminates "concept actually in image" from "concept forced into image",
which filters Florence-2 hallucinations without a separate CLIP re-ranking step.

Lazy-imports sam3 so the module can be *imported* without it installed; calling
segment() without sam3 raises loudly.  There is no degraded fallback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from wato_perception_2d.models.discovery import RegionProposal

log = logging.getLogger(__name__)


@dataclass
class SegmentedDetection:
    """A single segmented instance."""

    phrase: str
    rough_box: tuple[float, float, float, float]  # (x1, y1, x2, y2) pixels
    mask: np.ndarray  # (H, W) bool
    sam3_score: float = 0.0  # presence-token confidence
    discovery_score: float = 0.0  # Florence-2 confidence


class SAM3Segmenter:
    """SAM3 wrapper for text-prompted segmentation.

    Accepts:
    - text phrase prompts (from Florence-2 / discovery.py)
    - optional point prompts (projected LiDAR dynamic points, SAM4D cross-modal)

    Raises RuntimeError when sam3 is not installed — no degraded fallback.
    """

    def __init__(
        self,
        checkpoint: str = "sam3_hiera_large",
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

    def _load(self) -> None:
        if self._predictor is not None:
            return
        try:
            from sam3.build_sam import build_sam3
            from sam3.sam3_image_predictor import SAM3ImagePredictor

            model = build_sam3(self._checkpoint, device=self._device)
            self._predictor = SAM3ImagePredictor(model)
            log.info("SAM3 loaded (%s) on %s", self._checkpoint, self._device)
        except Exception as exc:
            raise RuntimeError(
                f"SAM 3.1 segmenter unavailable (checkpoint {self._checkpoint!r}): "
                f"{exc}. Install the 'sam3' package and ensure the checkpoint is "
                "present. perception_2d does not fall back."
            ) from exc

    def segment(
        self,
        image_rgb: np.ndarray,
        proposals: list[RegionProposal],
        lidar_point_prompts: Optional[np.ndarray] = None,
    ) -> list[SegmentedDetection]:
        """Segment each proposal in image_rgb using text prompts.

        Args:
            image_rgb: (H, W, 3) uint8 RGB image.
            proposals: list of RegionProposal from Florence-2.
            lidar_point_prompts: optional (M, 2) float32 pixel coordinates of
                projected LiDAR dynamic points (treated as foreground hints).

        Returns list of SegmentedDetection (same order as proposals).
        """
        if not proposals:
            return []

        self._load()
        self._predictor.set_image(image_rgb)

        results: list[SegmentedDetection] = []
        for prop in proposals:
            # Build optional LiDAR point prompts restricted to this proposal's box.
            point_coords: Optional[np.ndarray] = None
            point_labels: Optional[np.ndarray] = None
            if lidar_point_prompts is not None and lidar_point_prompts.shape[0] > 0:
                x1, y1, x2, y2 = prop.rough_box
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
                text=prop.phrase,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )
            mask = (
                masks_t[0].cpu().numpy().astype(bool)
                if hasattr(masks_t, "cpu")
                else masks_t[0].astype(bool)
            )
            sam3_score = float(scores_t[0]) if scores_t is not None else 0.0

            results.append(
                SegmentedDetection(
                    phrase=prop.phrase,
                    rough_box=prop.rough_box,
                    mask=mask,
                    sam3_score=sam3_score,
                    discovery_score=prop.confidence,
                )
            )

        return results
