"""Shared SAM2 / DAM4SAM predictor loader (cache).

Both backends reuse a local SAM2.1 checkpoint and the installed SAM2 Hydra
config. DAM4SAM loads the vendored author model class; stock SAM2 uses
``build_sam2_video_predictor``. Models are cached by backend, config, checkpoint
and device across cameras/chunks. Neither path downloads weights. Stock SAM2
returns None on loading failure; DAM4SAM propagates its error. Neither failure
can silently select a different backend.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import time
from typing import Optional

from wato_perception_2d._vendor import d4sm

log = logging.getLogger(__name__)

# (backend, config_file, ckpt_path, device) → predictor
_cache: dict[tuple, object] = {}

# Hydra config name bundled in the `sam2` package for the 2.1 hiera-large model.
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
BACKENDS = ("sam2", "dam4sam")


def installed_sam2_commit() -> Optional[str]:
    """The commit the installed `sam-2` package was built from (its PEP 610
    direct_url.json; docker/perception_2d.Dockerfile's SAM2_COMMIT is the one
    pin), or None when sam2 isn't installed from git."""
    try:
        text = importlib.metadata.distribution("sam-2").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    if not text:
        return None
    return json.loads(text).get("vcs_info", {}).get("commit_id")


def propagation_provenance(backend: str, device: str) -> dict:
    """What produced the masks, for manifests and resume-cache checks.

    Both backends run Meta's SAM2 modules (`sam2_commit`); DAM4SAM adds the
    vendored authors' memory policy (`dam4sam_commit`). SAM2's hole filling
    runs only on CUDA (check_sam2_cuda_extension proves it there); on CPU,
    SAM2 warns and returns the masks unfilled. The two produce different
    masks, so the device's effect is part of the policy.
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown mask propagation backend: {backend}")
    return {
        "backend": backend,
        "sam2_commit": installed_sam2_commit(),
        "dam4sam_commit": d4sm.COMMIT if backend == "dam4sam" else None,
        "adapter_version": 1,
        "cuda_hole_filling": device.startswith("cuda"),
    }


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
    backend: str = "sam2",
) -> Optional[object]:
    """Return a cached predictor for the requested memory policy.

    Args:
        checkpoint: path to the SAM2.1 .pt checkpoint (e.g.
            /data/models/sam2.1_hiera_large.pt).
        config_file: hydra config name bundled in the `sam2` package.
        device: "cuda"/"cpu"; defaults to cuda when available.
        backend: "sam2" (stock API default) or "dam4sam" (pipeline default).

    Stock SAM2 returns None when loading fails; DAM4SAM raises its loading error.
    """
    dev = device or _default_device()
    if backend not in BACKENDS:
        raise ValueError(f"Unknown mask propagation backend: {backend}")
    key = (backend, config_file, checkpoint, dev)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    if backend == "dam4sam":
        from wato_perception_2d.models.dam4sam import build_dam4sam

        predictor = build_dam4sam(config_file, checkpoint, dev)
        _cache[key] = predictor
        return predictor
    try:
        from sam2.build_sam import build_sam2_video_predictor

        log.info(
            "SAM2: building video predictor (config=%s, ckpt=%s) on %s …",
            config_file,
            checkpoint,
            dev,
        )
        t0 = time.perf_counter()
        # A box re-prompting an object on a frame it was already tracked through
        # (sam2_tracker's re-prompting) must be kept as a conditioning frame:
        # otherwise propagate_in_video re-infers that frame from memory and
        # discards the correction. A new object's first prompt is a
        # conditioning frame either way.
        predictor = build_sam2_video_predictor(
            config_file,
            checkpoint,
            device=dev,
            hydra_overrides_extra=["++model.add_all_frames_to_correct_as_cond=true"],
        )
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


def check_sam2_cuda_extension(device: str) -> None:
    """Exercise official hole filling before a CUDA pipeline starts expensive work.

    Import alone cannot detect a kernel built for the wrong GPU. SAM2 normally
    warns and returns unchanged masks on failure; that would silently disable
    the configured cleanup. Check the actual <=8-pixel boundary on this device.
    CPU inference remains available without this CUDA-only extension.
    """
    if not device.startswith("cuda"):
        return
    try:
        _probe_hole_filling(device)
    except Exception as exc:
        raise RuntimeError(
            "SAM2 CUDA hole filling is unavailable or failed its kernel check. "
            "Rebuild the perception_2d dependency and runtime images with "
            "docker/perception_2d.Dockerfile (SAM2_BUILD_ALLOW_ERRORS=0)."
        ) from exc


def _probe_hole_filling(device: str) -> None:
    """Raise unless SAM2's fill_holes_in_mask_scores fills an eight-pixel hole
    and keeps a nine-pixel one on `device`, without warning (SAM2's warning is
    its silent fallback)."""
    import warnings

    import torch
    from sam2.utils.misc import fill_holes_in_mask_scores

    with torch.inference_mode(), warnings.catch_warnings():
        warnings.simplefilter("error")
        scores = torch.ones((1, 1, 16, 16), device=device)
        scores[:, :, 2:4, 2:6] = -1  # eight pixels: fill
        scores[:, :, 8:11, 8:11] = -1  # nine pixels: retain
        expected = scores.clone()
        expected[:, :, 2:4, 2:6] = 0.1
        actual = fill_holes_in_mask_scores(scores, max_area=8)
        if not torch.equal(actual, expected):
            raise RuntimeError("SAM2 CUDA hole filling returned unexpected pixels")


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
