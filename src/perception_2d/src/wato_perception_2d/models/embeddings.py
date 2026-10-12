"""DINOv2 appearance-embedding extraction.

Extracts a per-mask DINOv2 embedding and persists it on each masklet.  This
stage only *produces* embeddings — the actual re-identification happens
downstream in the `tracking` component, which consumes them.

Used by the SAM2 tracker (sam2_tracker.py), which calls extract_dino_features
every k frames with all of that frame's tracked masks, batched into one
forward.  Lazy-imports torch so the module can be *imported* without it
installed; extracting a feature without torch/DINOv2 raises loudly.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from wato_perception_2d.model_registry import torch_hub_repo

log = logging.getLogger(__name__)

# Cache loaded DINOv2 models so torch.hub.load (model reconstruction + weight
# load + device transfer) runs ONCE per (model, device) rather than on every
# extract_dino_features call (which happens every k frames).
_MODEL_CACHE: dict[tuple[str, str], object] = {}


def _get_model(model_name: str, device: str):
    """Load (or fetch from cache) a DINOv2 model on the given device."""
    key = (model_name, device)
    model = _MODEL_CACHE.get(key)
    if model is None:
        import torch

        # Pinned ref, not the default branch. Two reasons: upstream `main`
        # moves (so an unpinned load is a silently changing ReID model), and
        # the ref names the TORCH_HOME cache directory
        # (facebookresearch_dinov2_<ref>) that fetch_models.py populated — a
        # bare "facebookresearch/dinov2" looks for ..._main, misses the
        # pre-fetched cache, and tries to reach the network from a container
        # whose /data/models is mounted read-only.
        model = torch.hub.load(torch_hub_repo(), model_name)
        model.eval().to(device)
        _MODEL_CACHE[key] = model
    return model


def _mask_crop(image_rgb: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    """The image cropped to the mask's bounding box, or None for an empty mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    r0, r1 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    c0, c1 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    return image_rgb[r0 : r1 + 1, c0 : c1 + 1]


def extract_dino_features(
    image_rgb: np.ndarray,
    masks: list[np.ndarray],
    model_name: str = "dinov2_vitl14",
    device: str = "cpu",
) -> list[Optional[np.ndarray]]:
    """DINOv2 embeddings for several masked regions of one image, in one forward.

    Each mask's bounding-box crop is resized to 224 (short side), centre-cropped
    to 224×224, normalized, and the batch goes through the model together; the
    CLS output is the embedding.

    Args:
        image_rgb: (H, W, 3) uint8 RGB image.
        masks: (H, W) bool masks selecting each foreground region.
        model_name: torch.hub DINOv2 model name.
        device: "cuda" or "cpu".

    Returns:
        One (D,) float32 embedding per mask, aligned with ``masks``; None for an
        empty mask (no region to embed).

    Raises:
        RuntimeError: if torch / torchvision / DINOv2 cannot be loaded.
    """
    crops = [_mask_crop(image_rgb, m) for m in masks]
    keep = [i for i, c in enumerate(crops) if c is not None]
    out: list[Optional[np.ndarray]] = [None] * len(masks)
    if not keep:
        return out

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
    batch = torch.stack([transform(PILImage.fromarray(crops[i])) for i in keep]).to(
        device
    )
    with torch.no_grad():
        feats = model(batch).float().cpu().numpy().astype(np.float32)
    for i, feat in zip(keep, feats):
        out[i] = feat
    return out


def extract_dino_feature(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    model_name: str = "dinov2_vitl14",
    device: str = "cpu",
) -> Optional[np.ndarray]:
    """DINOv2 embedding for one masked region; see extract_dino_features."""
    return extract_dino_features(image_rgb, [mask], model_name, device)[0]
