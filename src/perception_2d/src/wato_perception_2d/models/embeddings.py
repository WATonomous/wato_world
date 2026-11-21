"""DINOv2 appearance descriptors of masklets, for `tracking`'s ReID.

One descriptor per (masklet, embedded image), by Foreground Feature Averaging
(Kotar et al., "Are These the Same Apple?", NeurIPS 2023; the code path is
NIDS-Net's utils/inference_utils.py): the mask's bounding-box crop is resized
to a square, DINOv2's patch tokens under the mask are averaged, and the mean
is L2-normalised, so two descriptors compare by a dot product. The mask picks
the tokens, so two objects sharing a bounding box get different descriptors,
and the square resize keeps the ends of a wide or tall object that a centre
crop cuts off. Every embedded image keeps its own descriptor: a consumer
compares two objects by their closest pair (DeepSORT's gallery), which
separates nearby same-class objects better than one mean per masklet
(docs/tracking_measurements.md).

embed_chunk reads the chunk's masks back from disk, so it runs after the
tracker (or alone, on a chunk tracked earlier) with only DINOv2 on the GPU.
Lazy-imports torch; extraction without torch/DINOv2 raises.
"""

from __future__ import annotations

import functools
import logging
from typing import Optional

import numpy as np
from tqdm import tqdm

from wato_common.artifacts import masklets
from wato_common.artifacts.frame_index import CameraFrame
from wato_common.artifacts.masklets import MaskletEmbeddings, MaskletRecord
from wato_perception_2d.model_registry import torch_hub_repo

# How a mask's tokens become one descriptor; recorded with the descriptors,
# since only rows of one pooling compare.
POOLING = "ffa"
# The square the crop is resized to. FFA and NIDS-Net use 336; 224 separated
# identities as well on the nuScenes GT scenes at 2.25x fewer tokens
# (docs/tracking_measurements.md).
INPUT_SIZE = 224
_PATCH = 14  # DINOv2 ViT-*/14
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
# Crops per forward; one image's masks are split over forwards past this.
_BATCH = 64

log = logging.getLogger(__name__)

# One torch.hub.load per (model, device), not per call.
_MODEL_CACHE: dict[tuple[str, str], object] = {}


def _get_model(model_name: str, device: str):
    """Load (or fetch from cache) a DINOv2 model on the given device."""
    key = (model_name, device)
    model = _MODEL_CACHE.get(key)
    if model is None:
        import torch

        # Pinned ref: upstream `main` moves, and the ref names the TORCH_HOME
        # cache dir fetch_models.py populated (a bare repo looks for ..._main
        # and tries the network from a read-only mount).
        model = torch.hub.load(torch_hub_repo(), model_name)
        model.eval().to(device)
        _MODEL_CACHE[key] = model
    return model


@functools.cache
def check_model(model_name: str) -> None:
    """Raise unless the pinned DINOv2 and its weights load from TORCH_HOME.

    Loads on the CPU and drops the model, so it can run before the depth pass
    without taking VRAM. A missing pinned repo otherwise surfaces as a GitHub
    download error (HTTP 403 once rate-limited) at the first embedding.
    """
    import torch

    try:
        torch.hub.load(torch_hub_repo(), model_name)
    except Exception as exc:
        raise RuntimeError(
            f"DINOv2 {model_name!r} ({torch_hub_repo()}) did not load from "
            f"TORCH_HOME: {exc}. Fetch it with scripts/fetch_models.py."
        ) from exc


def _mask_crop(
    image_rgb: np.ndarray, mask: np.ndarray
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """The image and the mask cropped to the mask's bounding box, or None for
    an empty mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    r0, r1 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    c0, c1 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    return image_rgb[r0 : r1 + 1, c0 : c1 + 1], mask[r0 : r1 + 1, c0 : c1 + 1]


def model_input(crop_rgb: np.ndarray) -> np.ndarray:
    """The crop as the model sees it: (INPUT_SIZE, INPUT_SIZE, 3) uint8,
    resized bicubically to the square without keeping its aspect (FFA)."""
    from PIL import Image

    size = (INPUT_SIZE, INPUT_SIZE)
    return np.asarray(Image.fromarray(crop_rgb).resize(size, Image.BICUBIC))


def patch_mask(crop_mask: np.ndarray) -> np.ndarray:
    """(g, g) float32 0/1, g = INPUT_SIZE / 14: the patch tokens the mask
    covers. NIDS-Net's get_foreground_mask: the mask resized bilinearly to the
    token grid and cut at one half; a mask too thin to cover any token keeps
    them all, which is the plain crop's mean."""
    from PIL import Image

    g = INPUT_SIZE // _PATCH
    m = Image.fromarray(crop_mask.astype(np.uint8) * 255).resize((g, g), Image.BILINEAR)
    out = (np.asarray(m, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
    if not out.any():
        out[:] = 1.0
    return out


def pool(patch_tokens: np.ndarray, grid_masks: np.ndarray) -> np.ndarray:
    """(B, D) unit-norm descriptors: each crop's (g*g, D) patch tokens
    averaged over its (g, g) mask, then L2-normalised."""
    b = patch_tokens.shape[0]
    w = grid_masks.reshape(b, -1, 1).astype(np.float32)
    mean = (patch_tokens.astype(np.float32) * w).sum(axis=1) / w.sum(axis=1)
    return mean / np.linalg.norm(mean, axis=1, keepdims=True)


def extract_dino_features(
    image_rgb: np.ndarray,
    masks: list[np.ndarray],
    model_name: str = "dinov2_vitl14",
    device: str = "cpu",
) -> list[Optional[np.ndarray]]:
    """One FFA descriptor per masked region of one image (module docstring).

    Args:
        image_rgb: (H, W, 3) uint8 RGB image.
        masks: (H, W) bool masks selecting each foreground region.
        model_name: torch.hub DINOv2 model name.
        device: "cuda" or "cpu".

    Returns:
        One (D,) float32 unit-norm descriptor per mask, aligned with
        ``masks``; None for an empty mask (no region to embed).

    Raises:
        RuntimeError: if torch / DINOv2 cannot be loaded.
    """
    crops = [_mask_crop(image_rgb, m) for m in masks]
    keep = [i for i, c in enumerate(crops) if c is not None]
    out: list[Optional[np.ndarray]] = [None] * len(masks)
    if not keep:
        return out

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            f"DINOv2 embedding extraction requires torch: {exc}. "
            "Install it. perception_2d does not skip embeddings silently."
        ) from exc

    model = _get_model(model_name, device)
    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
    for start in range(0, len(keep), _BATCH):
        idx = keep[start : start + _BATCH]
        images = np.stack([model_input(crops[i][0]) for i in idx])
        grids = np.stack([patch_mask(crops[i][1]) for i in idx])
        batch = torch.from_numpy(images).to(device).permute(0, 3, 1, 2).float()
        batch = (batch / 255.0 - mean) / std
        with torch.no_grad():
            tokens = model.forward_features(batch)["x_norm_patchtokens"]
        feats = pool(tokens.float().cpu().numpy(), grids)
        for i, feat in zip(idx, feats):
            out[i] = feat.astype(np.float32)
    return out


def _load_rgb(path: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def embed_chunk(
    bag_id: str,
    chunk_id: str,
    records: list[MaskletRecord],
    frames_by_cam: dict[str, list[CameraFrame]],
    *,
    model_name: str,
    every_k_frames: int,
    device: str,
) -> MaskletEmbeddings:
    """The chunk's descriptors, from its masks on disk.

    Per camera, every ``every_k_frames``-th image (by position in the
    camera's images) is embedded: each masklet with a mask there gets one row.
    An unreadable image raises: a masklet silently without descriptors would
    look like one tracking has nothing to compare.
    """
    if every_k_frames <= 0:
        raise ValueError(f"every_k_frames {every_k_frames}: embeddings are off")
    index = masklets.MaskletIndex(bag_id, chunk_id, records)
    ids: list[str] = []
    cams: list[str] = []
    seqs: list[int] = []
    times: list[int] = []
    feats: list[np.ndarray] = []
    for cam_id, cam_frames in frames_by_cam.items():
        sampled = [f for k, f in enumerate(cam_frames) if k % every_k_frames == 0]
        for frame in tqdm(
            sampled, desc=f"{chunk_id}/{cam_id} embed", unit="frame", leave=False
        ):
            present = index.at(cam_id, frame.camera_seq)
            if not present:
                continue
            image = _load_rgb(frame.image_path)
            found = extract_dino_features(
                image,
                [index.mask(r, frame.camera_seq) for r in present],
                model_name,
                device,
            )
            for record, feat in zip(present, found):
                if feat is None:
                    continue
                ids.append(record.masklet_id)
                cams.append(cam_id)
                seqs.append(frame.camera_seq)
                times.append(frame.timestamp_ns)
                feats.append(feat)
    dim = feats[0].shape[0] if feats else 0
    return MaskletEmbeddings(
        model=model_name,
        pooling=POOLING,
        masklet_ids=np.array(ids, dtype=str),
        cam_ids=np.array(cams, dtype=str),
        camera_seq=np.array(seqs, dtype=np.int64),
        camera_timestamp_ns=np.array(times, dtype=np.int64),
        features=np.stack(feats) if feats else np.zeros((0, dim), dtype=np.float32),
    )
