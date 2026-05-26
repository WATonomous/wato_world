"""Depth Anything V2 wrapper.

Produces per-frame relative depth maps from RGB images.  The relative depth
is subsequently aligned to metric scale via LiDAR anchor pairs in depth_align.py.

Lazy-imports depth_anything_v2 so the module can be imported without it installed.
Confidence is derived from depth-gradient magnitude: flat regions get high
confidence, depth discontinuities get low confidence.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

_warned_missing = False


class DepthAnythingV2:
    """Depth Anything V2 relative-depth estimator.

    Returns (relative_depth, confidence) per image:
    - relative_depth: (H, W) float32, arbitrary positive scale.
    - confidence: (H, W) float32 in [0, 1], derived from gradient magnitude.

    Falls back to zero arrays when the package is not installed.
    """

    def __init__(
        self,
        model_size: str = "large",
        device: Optional[str] = None,
    ) -> None:
        self._model_size = model_size
        self._device = device or self._default_device()
        self._model = None  # lazy-loaded

    @staticmethod
    def _default_device() -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load(self) -> bool:
        global _warned_missing
        if self._model is not None:
            return True
        try:
            from depth_anything_v2.dpt import DepthAnythingV2 as _DA

            encoder = {"small": "vits", "base": "vitb", "large": "vitl"}.get(
                self._model_size, "vitl"
            )
            self._model = _DA(encoder=encoder, features=256, out_channels=[256, 512, 1024, 1024])
            self._model.load_pretrained()
            self._model.to(self._device)
            self._model.eval()
            log.info("DepthAnythingV2 loaded (%s) on %s", self._model_size, self._device)
            return True
        except Exception as exc:  # noqa: BLE001
            if not _warned_missing:
                log.warning(
                    "DepthAnythingV2 unavailable (%s) — returning zero depth maps. "
                    "Install: pip install depth-anything-v2",
                    exc,
                )
                _warned_missing = True
            return False

    def infer(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run inference on one RGB image.

        Args:
            image_rgb: (H, W, 3) uint8 RGB image.

        Returns:
            relative_depth: (H, W) float32, arbitrary positive scale.
            confidence: (H, W) float32 in [0, 1].
        """
        H, W = image_rgb.shape[:2]
        if not self._load():
            return np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32)

        try:
            import torch

            with torch.no_grad():
                depth = self._model.infer_image(image_rgb)  # (H, W) float32
            if hasattr(depth, "cpu"):
                depth = depth.cpu().numpy()
            depth = np.asarray(depth, dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            log.warning("DepthAnythingV2 inference failed: %s", exc)
            return np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32)

        confidence = _gradient_confidence(depth)
        return depth, confidence


def _gradient_confidence(depth: np.ndarray) -> np.ndarray:
    """Derive per-pixel confidence from depth-gradient magnitude.

    Confidence = 1 / (1 + gradient_magnitude).  Flat regions → ~1.0;
    depth discontinuities → near 0.
    """
    gy = np.gradient(depth, axis=0)
    gx = np.gradient(depth, axis=1)
    grad_mag = np.sqrt(gx**2 + gy**2)
    return (1.0 / (1.0 + grad_mag)).astype(np.float32)
