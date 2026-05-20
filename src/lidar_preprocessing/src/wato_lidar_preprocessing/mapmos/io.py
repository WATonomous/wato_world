"""MapMOS per-sweep logit sidecar I/O.

Logit files live next to the world NPZ as
`<sweep_id:06d>_mapmos_logit.npy` and are float32 arrays whose length
equals the world NPZ's xyz length (entries at ground positions are
filled with 0.0 — see plan non-negotiable #3).

Two distinct cases the caller must NOT collapse (plan non-negotiable #20):
- `read_logits` returns `None`     -> file missing  -> classify falls
                                       back to geometry-only (warning).
- `read_logits` returns `np.empty(0)` -> file present, sweep had zero
                                       points -> length-aligned, no warn.
Branch on `is None`, never on `len(...) == 0`.
"""

from __future__ import annotations

import os

import numpy as np

from wato_common.artifact_store import local_path


def write_logits(path_uri: str, logits: np.ndarray) -> None:
    """Persist float32 per-point logits at the well-known sidecar path."""
    if logits.dtype != np.float32:
        raise ValueError(
            f"mapmos logit dtype must be float32, got {logits.dtype} "
            "(callers must clamp + cast before write)"
        )
    np.save(local_path(path_uri), logits)


def read_logits(path_uri: str) -> np.ndarray | None:
    """Read a sidecar logit file. Returns None when the file is missing.

    A zero-length array (legitimate empty sweep) is distinct from None
    and must be handled separately by callers.
    """
    p = local_path(path_uri)
    if not os.path.exists(p):
        return None
    return np.load(p)
