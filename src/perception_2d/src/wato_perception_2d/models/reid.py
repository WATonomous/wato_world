"""DINOv2 ReID feature extraction.

Used by sam3_concept_tracker.py to attach a per-masklet embedding (for
downstream cross-camera ReID).  Lazy-imports torch so the module can be
imported without it installed.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

_warned_missing = False
# (model_name, device) → (model, transform). Cached across calls so we
# don't pay torch.hub.load + state_dict load + .to(device) every call.
_model_cache: dict[tuple[str, str], tuple] = {}


def _get_dino(model_name: str, device: str):
    """Return (model, transform), constructing + caching on first use."""
    key = (model_name, device)
    cached = _model_cache.get(key)
    if cached is not None:
        return cached

    import time

    import torch
    import torchvision.transforms as T

    log.info("DINOv2: loading %s on %s …", model_name, device)
    t0 = time.perf_counter()
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().to(device)
    transform = T.Compose(
        [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    _model_cache[key] = (model, transform)
    log.info(
        "DINOv2 ready in %.1fs: %s on %s",
        time.perf_counter() - t0, model_name, device,
    )
    return _model_cache[key]


def extract_dino_feature(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    model_name: str = "dinov2_vitl14",
    device: str = "cpu",
) -> Optional[np.ndarray]:
    """Extract a DINOv2 embedding for the masked region in image_rgb.

    Crops the mask bounding box, resizes to 224×224, normalizes, and runs
    the DINOv2 CLS token through the model. The model is loaded once and
    cached per (model_name, device).
    """
    global _warned_missing
    try:
        import torch
        from PIL import Image as PILImage

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            return None
        r0, r1 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
        c0, c1 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
        crop = image_rgb[r0 : r1 + 1, c0 : c1 + 1]

        model, transform = _get_dino(model_name, device)
        tensor = transform(PILImage.fromarray(crop)).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(tensor)[0].cpu().numpy().astype(np.float32)
        return feat
    except Exception as exc:  # noqa: BLE001
        if not _warned_missing:
            log.warning("DINOv2 unavailable (%s) — skipping feature extraction.", exc)
            _warned_missing = True
        return None
