"""Shared SAM 3.1 multiplex video predictor loader (cache).

SAM 3.1's multiplex tracker is NOT available through HuggingFace transformers —
`facebook/sam3.1` hosts checkpoints only ("there is no Hugging Face Transformers
integration … visit the SAM 3 GitHub repository"). It is loaded through Meta's
official `sam3` package (facebookresearch/sam3) via
`build_sam3_multiplex_video_predictor`, which loads `sam3.1_multiplex.pt`
directly (the checkpoint is built for that code).

The predictor is heavy and GPU-resident; this module caches one instance so the
segmenter/tracker share it across cameras and chunks.

Lazy-imports `sam3` so this module can be imported without it installed; callers
treat a None return as "SAM 3.1 unavailable".
"""

from __future__ import annotations

import logging
import time
from typing import Optional

log = logging.getLogger(__name__)

# (version, use_fa3) → predictor
_cache: dict[tuple, object] = {}


def get_sam3_predictor(
    version: str = "sam3.1",
    use_fa3: bool = False,
) -> Optional[object]:
    """Return a cached Sam3MultiplexVideoPredictor, or None if unavailable.

    None means the `sam3` package or its checkpoint isn't installed/fetched.
    The pipeline treats None as a hard error for the chunk (empty output + log);
    there is no hand-rolled tracking fallback.
    """
    key = (version, use_fa3)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    try:
        from sam3.model_builder import (
            build_sam3_multiplex_video_predictor,
            download_ckpt_from_hf,
        )

        log.info("SAM 3.1: building multiplex video predictor (use_fa3=%s) …", use_fa3)
        t0 = time.perf_counter()
        ckpt = download_ckpt_from_hf(version=version)
        predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=ckpt,
            use_fa3=use_fa3,
            async_loading_frames=False,  # we hand it in-memory PIL frames
        )
        _cache[key] = predictor
        log.info(
            "SAM 3.1 predictor ready in %.1fs (checkpoint=%s)",
            time.perf_counter() - t0, ckpt,
        )
        return predictor
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "SAM 3.1 multiplex predictor unavailable (%s) — install the `sam3` "
            "package and fetch facebook/sam3.1 into the HF cache.",
            exc,
        )
        return None
