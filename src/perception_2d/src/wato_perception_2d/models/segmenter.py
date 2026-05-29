"""SAM3 text-prompted segmentation wrapper.

Given per-frame noun phrases (from discovery.py / Florence-2) and optional
LiDAR point prompts (cross-modal, SAM4D-style), produces per-phrase binary
masks with SAM3's presence-token scores.

SAM3 is text-prompted: each phrase is passed directly as a text query rather
than requiring a bounding box from a separate detector.  The presence-token
score discriminates "concept actually in image" from "concept forced into image",
which filters Florence-2 hallucinations without a separate CLIP re-ranking step.

Lazy-imports sam3 so the module can be imported without it installed.
Falls back to filled bounding-box masks using the rough_box from each proposal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from wato_perception_2d.models.discovery import RegionProposal

log = logging.getLogger(__name__)

_warned_missing = False


@dataclass
class SegmentedDetection:
    """A single segmented instance."""

    phrase: str
    rough_box: tuple[float, float, float, float]  # (x1, y1, x2, y2) pixels
    mask: np.ndarray  # (H, W) bool
    sam3_score: float = 0.0    # presence-token confidence
    discovery_score: float = 0.0  # Florence-2 confidence


class SAM3Segmenter:
    """SAM3 wrapper for text-prompted segmentation.

    Accepts:
    - text phrase prompts (from Florence-2 / discovery.py)
    - optional point prompts (projected LiDAR dynamic points, SAM4D cross-modal)

    Falls back to filled bounding-box masks when sam3 is not installed.
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

    def _load(self) -> bool:
        global _warned_missing
        if self._predictor is not None:
            return True
        try:
            from sam3.build_sam import build_sam3
            from sam3.sam3_image_predictor import SAM3ImagePredictor

            model = build_sam3(self._checkpoint, device=self._device)
            self._predictor = SAM3ImagePredictor(model)
            log.info("SAM3 loaded (%s) on %s", self._checkpoint, self._device)
            return True
        except Exception as exc:  # noqa: BLE001
            if not _warned_missing:
                log.warning(
                    "SAM3 unavailable (%s) — using bbox-fill fallback masks. "
                    "Install: pip install sam3",
                    exc,
                )
                _warned_missing = True
            return False

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

        H, W = image_rgb.shape[:2]

        if not self._load():
            return self._bbox_fill_fallback(proposals, H, W)

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

            try:
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
            except Exception as exc:  # noqa: BLE001
                log.debug("SAM3 predict failed for phrase %r: %s", prop.phrase, exc)
                mask = self._box_mask(prop.rough_box, H, W)
                sam3_score = 0.0

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

    @staticmethod
    def _box_mask(
        rough_box: tuple[float, float, float, float], H: int, W: int
    ) -> np.ndarray:
        mask = np.zeros((H, W), dtype=bool)
        x1, y1, x2, y2 = (int(v) for v in rough_box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        mask[y1:y2, x1:x2] = True
        return mask

    @staticmethod
    def _bbox_fill_fallback(
        proposals: list[RegionProposal], H: int, W: int
    ) -> list[SegmentedDetection]:
        """Fill the rough bounding box as the mask (used when SAM3 is absent)."""
        return [
            SegmentedDetection(
                phrase=p.phrase,
                rough_box=p.rough_box,
                mask=SAM3Segmenter._box_mask(p.rough_box, H, W),
                sam3_score=0.0,
                discovery_score=p.confidence,
            )
            for p in proposals
        ]
