"""DINOv2 appearance-embedding extraction.

Extracts a per-mask DINOv2 embedding and persists it on each masklet.  This
stage only *produces* embeddings — the actual re-identification happens
downstream in the `tracking` component, which consumes them.

Shared by both tracker_2d.py (IoU fallback tracker) and sam3_tracker.py
(SAM 3.1 primary tracker).  Lazy-imports torch so the module can be *imported*
without it installed; extracting a feature without torch/DINOv2 raises loudly.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# Cache loaded DINOv2 models so torch.hub.load (model reconstruction + weight
# load + device transfer) runs ONCE per (model, device) rather than on every
# extract_dino_feature call (which happens every k frames per track).
_MODEL_CACHE: dict[tuple[str, str], object] = {}


def _get_model(model_name: str, device: str):
    """Load (or fetch from cache) a DINOv2 model on the given device."""
    key = (model_name, device)
    model = _MODEL_CACHE.get(key)
    if model is None:
        import torch

        model = torch.hub.load("facebookresearch/dinov2", model_name)
        model.eval().to(device)
        _MODEL_CACHE[key] = model
    return model


def extract_dino_feature(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    model_name: str = "dinov2_vitl14",
    device: str = "cpu",
) -> Optional[np.ndarray]:
    """Extract a DINOv2 embedding for the masked region in image_rgb.

    Crops the mask bounding box, resizes to 224×224, normalizes, and runs
    the DINOv2 CLS token through the model.

    Args:
        image_rgb: (H, W, 3) uint8 RGB image.
        mask: (H, W) bool mask selecting the foreground region.
        model_name: torch.hub DINOv2 model name.
        device: "cuda" or "cpu".

    Returns:
        (D,) float32 embedding, or None when the mask is empty (no region to
        embed).

    Raises:
        RuntimeError: if torch / torchvision / DINOv2 cannot be loaded.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    r0, r1 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    c0, c1 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    crop = image_rgb[r0 : r1 + 1, c0 : c1 + 1]

    try:
        import torch
        import torchvision.transforms as T
        from PIL import Image as PILImage
    except Exception as exc:
        raise RuntimeError(
            f"DINOv2 embedding extraction requires torch + torchvision: {exc}. "
            "Install them. perception_2d does not skip embeddings silently."
        ) from exc

    model = _get_model(model_name, device)
    transform = T.Compose(
        [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor = transform(PILImage.fromarray(crop)).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(tensor)[0].cpu().numpy().astype(np.float32)
    return feat
