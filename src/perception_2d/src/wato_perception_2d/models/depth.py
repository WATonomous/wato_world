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


class DepthAnythingV2:
    """Depth Anything V2 relative-depth estimator.

    Returns (relative_depth, confidence) per image:
    - relative_depth: (H, W) float32, arbitrary positive scale.
    - confidence: (H, W) float32 in [0, 1], derived from gradient magnitude.

    Raises RuntimeError when the package/checkpoint is unavailable — no
    degraded fallback.
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
        "vits": (
            {"features": 64, "out_channels": [48, 96, 192, 384]},
            "depth-anything/Depth-Anything-V2-Small",
        ),
        "vitb": (
            {"features": 128, "out_channels": [96, 192, 384, 768]},
            "depth-anything/Depth-Anything-V2-Base",
        ),
        "vitl": (
            {"features": 256, "out_channels": [256, 512, 1024, 1024]},
            "depth-anything/Depth-Anything-V2-Large",
        ),
    }

    def _encoder(self) -> str:
        s = self._model_size.lower()
        if "vits" in s or "small" in s:
            return "vits"
        if "vitb" in s or "base" in s:
            return "vitb"
        return "vitl"

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from depth_anything_v2.dpt import DepthAnythingV2 as _DA
            from huggingface_hub import hf_hub_download

            encoder = self._encoder()
            head_cfg, repo_id = self._ENCODER_CONFIGS[encoder]
            ckpt = hf_hub_download(
                repo_id=repo_id, filename=f"depth_anything_v2_{encoder}.pth"
            )
            model = _DA(encoder=encoder, **head_cfg)
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            # Assign only after a fully successful load: a partially-built model
            # left in self._model would make the guard above wrongly short-circuit.
            self._model = model.to(self._device).eval()
            log.info("DepthAnythingV2 loaded (%s) on %s", encoder, self._device)
        except Exception as exc:
            self._model = None
            raise RuntimeError(
                f"Depth Anything V2 ({self._encoder()}) could not be loaded: {exc}. "
                "Install 'depth_anything_v2' (+ huggingface_hub) or set depth.enabled: "
                "false. perception_2d does not fall back to zero depth."
            ) from exc

    def infer(
        self, image_rgb: np.ndarray, with_confidence: bool = False
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Run inference on one RGB image.

        Args:
            image_rgb: (H, W, 3) uint8 RGB image.
            with_confidence: when True, also compute the gradient-magnitude
                confidence map.  The pipeline does not consume it, so it
                defaults off to avoid the per-frame gradient computation.

        Returns:
            relative_depth: (H, W) float32, arbitrary positive scale.
            confidence: (H, W) float32 in [0, 1], or None when not requested.
        """
        self._load()

        import torch

        with torch.no_grad():
            # infer_image expects a BGR (cv2-style) array; our input is RGB.
            image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
            depth = self._model.infer_image(image_bgr)  # (H, W) float32
        if hasattr(depth, "cpu"):
            depth = depth.cpu().numpy()
        depth = np.asarray(depth, dtype=np.float32)

        confidence = _gradient_confidence(depth) if with_confidence else None
        return depth, confidence

    def infer_batch(
        self,
        images_rgb: list[np.ndarray],
        *,
        batch_size: int = 1,
        with_confidence: bool = False,
    ) -> tuple[list[np.ndarray], list[Optional[np.ndarray]]]:
        """Run inference on many same-resolution RGB images, GPU-batched.

        All images must share one (H, W) — true when they come from a single
        camera stream, which is how the pipeline calls this.  Frames are pushed
        through the backbone ``batch_size`` at a time: a larger batch trades VRAM
        for throughput (the per-frame backbone forward is the depth pass's
        bottleneck).  Equivalent to looping ``infer`` when batch_size == 1.

        Args:
            images_rgb: list of (H, W, 3) uint8 RGB images, all the same size.
            batch_size: frames per GPU forward (clamped to >= 1).
            with_confidence: also compute the gradient-magnitude confidence map.

        Returns:
            relative_depth: list of (H, W) float32 maps, aligned with images_rgb.
            confidence: list of (H, W) float32 maps in [0, 1], or None entries
                when with_confidence is False.
        """
        self._load()

        import torch
        import torch.nn.functional as F

        bs = max(1, batch_size)
        depths: list[np.ndarray] = []
        confidences: list[Optional[np.ndarray]] = []

        with torch.no_grad():
            for start in range(0, len(images_rgb), bs):
                chunk = images_rgb[start : start + bs]
                tensors = []
                sizes = []
                for img in chunk:
                    # image2tensor expects a BGR (cv2-style) array; ours is RGB.
                    img_bgr = np.ascontiguousarray(img[:, :, ::-1])
                    tensor, (h, w) = self._model.image2tensor(img_bgr)
                    tensors.append(tensor)
                    sizes.append((h, w))
                # Same input (h, w) → same resized tensor shape, so this cat is
                # valid (and would raise loudly if a caller mixed resolutions).
                batch = torch.cat(tensors, dim=0)  # (B, 3, H', W')
                pred = self._model.forward(batch)  # (B, H', W')
                # All frames share (h, w), so one interpolate covers the batch.
                h, w = sizes[0]
                pred = F.interpolate(
                    pred[:, None], (h, w), mode="bilinear", align_corners=True
                )[:, 0]
                pred_np = np.asarray(pred.cpu().numpy(), dtype=np.float32)
                for depth in pred_np:
                    depths.append(depth)
                    confidences.append(
                        _gradient_confidence(depth) if with_confidence else None
                    )

        return depths, confidences

    def unload(self) -> None:
        """Drop the GPU-resident model so its VRAM can be reclaimed.

        The pipeline runs depth as its own pass and unloads DA-V2 before the
        detector + SAM2 are built, so the two heavy model sets are never
        co-resident (critical on small-VRAM GPUs). Caller should follow with
        torch.cuda.empty_cache().
        """
        self._model = None


def _gradient_confidence(depth: np.ndarray) -> np.ndarray:
    """Derive per-pixel confidence from depth-gradient magnitude.

    Confidence = 1 / (1 + gradient_magnitude).  Flat regions → ~1.0;
    depth discontinuities → near 0.
    """
    gy = np.gradient(depth, axis=0)
    gx = np.gradient(depth, axis=1)
    grad_mag = np.sqrt(gx**2 + gy**2)
    return (1.0 / (1.0 + grad_mag)).astype(np.float32)
