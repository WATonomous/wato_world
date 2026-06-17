"""Shared SAM2 video predictor loader (cache).

SAM2.1 is loaded through Meta's official `sam2` package (facebookresearch/sam2)
via ``SAM2VideoPredictor.from_pretrained``, which pulls the checkpoint + hydra
config from the HuggingFace repo ``facebook/sam2.1-hiera-large`` (pre-fetched
into HF_HOME by scripts/fetch_models.py).

Unlike the SAM 3.1 multiplex tracker this replaces, the SAM2 video predictor
needs no upstream monkeypatches: it offloads to CPU through documented
``init_state`` kwargs and bounds its own memory.  The predictor is GPU-resident,
so this module caches one instance shared across cameras and chunks.

Lazy-imports `sam2` so this module can be imported without it installed; callers
treat a None return as "SAM2 unavailable" (the pipeline turns that into a hard
error for the chunk — there is no hand-rolled tracking fallback).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

log = logging.getLogger(__name__)

# (model_id, device) → predictor
_cache: dict[tuple, object] = {}


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def get_sam2_predictor(
    model_id: str = "facebook/sam2.1-hiera-large",
    device: Optional[str] = None,
) -> Optional[object]:
    """Return a cached SAM2VideoPredictor, or None if unavailable.

    None means the `sam2` package or its checkpoint isn't installed/fetched.
    The pipeline treats None as a hard error for the chunk (fail loud); there is
    no hand-rolled tracking fallback.
    """
    dev = device or _default_device()
    key = (model_id, dev)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    try:
        from sam2.sam2_video_predictor import SAM2VideoPredictor

        log.info("SAM2: building video predictor (%s) on %s …", model_id, dev)
        t0 = time.perf_counter()
        predictor = SAM2VideoPredictor.from_pretrained(model_id, device=dev)
        _cache[key] = predictor
        log.info(
            "SAM2 predictor ready in %.1fs (%s)", time.perf_counter() - t0, model_id
        )
        return predictor
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "SAM2 video predictor unavailable (%s) — install the `sam2` package "
            "and fetch %s into the HF cache (scripts/fetch_models.py).",
            exc,
            model_id,
        )
        return None


def sam2_importable() -> bool:
    """Cheap check that `sam2` imports, WITHOUT building the heavy predictor.

    The pipeline calls this before the (expensive) depth pass so a missing
    `sam2` install / dependency fails fast instead of after the whole depth pass.
    Actually executes the import (not importlib.util.find_spec) so a broken
    transitive dep is caught, not just an absent top-level package.
    """
    import importlib

    try:
        importlib.import_module("sam2.sam2_video_predictor")
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("SAM2 (`sam2.sam2_video_predictor`) not importable: %s", exc)
        return False


def release_sam2_predictor() -> None:
    """Drop the cached predictor and free its GPU memory (best-effort).

    Called at the end of a bag so a long-lived process doesn't keep SAM2 parked
    in VRAM after the run.
    """
    _cache.clear()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
