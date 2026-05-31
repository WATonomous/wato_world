"""Depth Anything V2 wrapper.

Produces per-frame relative depth maps from RGB images.  The relative depth
is subsequently aligned to metric scale via LiDAR anchor pairs in depth_align.py.

Lazy-imports depth_anything_v2 so the module can be imported without it installed.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

_warned_missing = False


class DepthAnythingV2:
    """Depth Anything V2 relative-depth estimator.

    Returns relative_depth (H, W) float32, arbitrary positive scale, per image.
    Falls back to a zero array when the package is not installed.
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

    # encoder → (DPT head config, HF repo holding the .pth checkpoint).
    _ENCODER_CONFIGS = {
        "vits": ({"features": 64, "out_channels": [48, 96, 192, 384]},
                 "depth-anything/Depth-Anything-V2-Small"),
        "vitb": ({"features": 128, "out_channels": [96, 192, 384, 768]},
                 "depth-anything/Depth-Anything-V2-Base"),
        "vitl": ({"features": 256, "out_channels": [256, 512, 1024, 1024]},
                 "depth-anything/Depth-Anything-V2-Large"),
    }

    def _encoder(self) -> str:
        s = self._model_size.lower()
        if "vits" in s or "small" in s:
            return "vits"
        if "vitb" in s or "base" in s:
            return "vitb"
        return "vitl"

    def _load(self) -> bool:
        global _warned_missing
        if self._model is not None:
            return True
        try:
            import time

            import torch
            from depth_anything_v2.dpt import DepthAnythingV2 as _DA
            from huggingface_hub import hf_hub_download

            encoder = self._encoder()
            head_cfg, repo_id = self._ENCODER_CONFIGS[encoder]
            log.info(
                "DepthAnythingV2: loading %s (%s) on %s …",
                encoder, repo_id, self._device,
            )
            t0 = time.perf_counter()
            ckpt = hf_hub_download(
                repo_id=repo_id, filename=f"depth_anything_v2_{encoder}.pth"
            )
            model = _DA(encoder=encoder, **head_cfg)
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            # Assign only after a fully successful load: a partially-built model
            # left in self._model would make the guard above wrongly short-circuit.
            self._model = model.to(self._device).eval()
            log.info(
                "DepthAnythingV2 ready in %.1fs: %s on %s",
                time.perf_counter() - t0, encoder, self._device,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            self._model = None
            if not _warned_missing:
                log.warning(
                    "DepthAnythingV2 unavailable (%s) — returning zero depth maps. "
                    "Install: pip install depth-anything-v2",
                    exc,
                )
                _warned_missing = True
            return False

    def infer(self, image_rgb: np.ndarray) -> np.ndarray:
        """Run inference on one RGB image.

        Args:
            image_rgb: (H, W, 3) uint8 RGB image.

        Returns:
            relative_depth: (H, W) float32, arbitrary positive scale.
        """
        H, W = image_rgb.shape[:2]
        if not self._load():
            return np.zeros((H, W), dtype=np.float32)

        try:
            import torch

            with torch.no_grad():
                # infer_image expects a BGR (cv2-style) array; our input is RGB.
                image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
                depth = self._model.infer_image(image_bgr)  # (H, W) float32
            if hasattr(depth, "cpu"):
                depth = depth.cpu().numpy()
            depth = np.asarray(depth, dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            log.warning("DepthAnythingV2 inference failed: %s", exc)
            return np.zeros((H, W), dtype=np.float32)

        return depth
