"""Shared SAM2 video predictor loader (cache).

SAM2.1 is built with the `sam2` package's ``build_sam2_video_predictor`` from
a local checkpoint (fetched by scripts/fetch_models.py; the hydra config ships
in the package) — not ``from_pretrained``, since the container runs
``HF_HUB_OFFLINE=1``. One GPU-resident predictor is cached across cameras and
chunks. Lazy-imports `sam2`; a None return means unavailable, which the
pipeline turns into a hard error.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

log = logging.getLogger(__name__)

# (config_file, ckpt_path, device) → predictor
_cache: dict[tuple, object] = {}

# Hydra config name bundled in the `sam2` package for the 2.1 hiera-large model.
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def get_sam2_predictor(
    checkpoint: str,
    config_file: str = DEFAULT_CONFIG,
    device: Optional[str] = None,
) -> Optional[object]:
    """Return a cached SAM2 video predictor, or None if unavailable.

    Args:
        checkpoint: path to the SAM2.1 .pt checkpoint (e.g.
            /data/models/sam2.1_hiera_large.pt).
        config_file: hydra config name bundled in the `sam2` package.
        device: "cuda"/"cpu"; defaults to cuda when available.

    None means the `sam2` package isn't installed or the checkpoint is missing.
    """
    dev = device or _default_device()
    key = (config_file, checkpoint, dev)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    try:
        from sam2.build_sam import build_sam2_video_predictor

        log.info(
            "SAM2: building video predictor (config=%s, ckpt=%s) on %s …",
            config_file,
            checkpoint,
            dev,
        )
        t0 = time.perf_counter()
        predictor = build_sam2_video_predictor(config_file, checkpoint, device=dev)
        _cache[key] = predictor
        log.info(
            "SAM2 predictor ready in %.1fs (ckpt=%s)",
            time.perf_counter() - t0,
            checkpoint,
        )
        return predictor
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "SAM2 video predictor unavailable (%s) — install the `sam2` package "
            "and place the checkpoint at %s (scripts/fetch_models.py).",
            exc,
            checkpoint,
        )
        return None


def sam2_importable() -> bool:
    """Cheap check that `sam2` imports, WITHOUT building the heavy predictor.

    Called before the depth pass. Executes the import (not find_spec), so a
    broken transitive dependency is caught too.
    """
    import importlib

    try:
        importlib.import_module("sam2.build_sam")
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("SAM2 (`sam2.build_sam`) not importable: %s", exc)
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
