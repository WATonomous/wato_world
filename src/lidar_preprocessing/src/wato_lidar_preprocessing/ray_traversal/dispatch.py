"""Public dispatch surface for ray traversal.

Imports numba and the JIT kernel lazily. When numba is unavailable, every
public function raises ImportError with a clear remediation message — no
silent slow fallback. The persistence classifier doesn't call into here,
so component setups that don't need log-odds work without numba installed.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

_NUMBA_IMPORT_ERROR: BaseException | None = None
try:
    import numba
    import numba.types as _nbtypes
    from ._numba_kernel import (
        _extract_arrays_numba,
        _extract_max_logit_numba,
        _update_sweep_numba,
    )

    _NUMBA_AVAILABLE = True
except Exception as _e:  # noqa: BLE001 — half-installed numba can raise AttributeError
    _NUMBA_AVAILABLE = False
    _NUMBA_IMPORT_ERROR = _e
    # Don't warn here: importing this package for non-log-odds reasons
    # (ground, viz, deskew, persistence-only classify) shouldn't print a
    # warning the caller can't act on. `_require_numba` raises a hard
    # ImportError with the original cause embedded as soon as anyone tries
    # to actually use the log-odds path. Module import stays silent.
    log.debug(
        "numba import failed at module load (%s: %s); will surface on "
        "first log-odds call",
        type(_e).__name__,
        _e,
    )


_MISSING_NUMBA_MSG = (
    "log-odds classification requires numba; install it "
    "(pip install 'numba>=0.59') or set classification_method=persistence "
    "in lidar_preprocessing.yaml"
)


def _require_numba() -> None:
    if not _NUMBA_AVAILABLE:
        if _NUMBA_IMPORT_ERROR is not None:
            raise ImportError(
                f"{_MISSING_NUMBA_MSG}\n"
                f"(original error: {type(_NUMBA_IMPORT_ERROR).__name__}: "
                f"{_NUMBA_IMPORT_ERROR})"
            )
        raise ImportError(_MISSING_NUMBA_MSG)


def make_log_odds_dicts() -> tuple:
    """Return (log_odds, n_obs, n_hits) as Numba typed dicts.

    Hard-fails with ImportError if numba is not installed, instead of
    returning plain dicts that would cause the pipeline to silently run
    ~10^4× slower on real data.
    """
    _require_numba()
    log_odds = numba.typed.Dict.empty(
        key_type=_nbtypes.int64,
        value_type=_nbtypes.float32,
    )
    n_obs = numba.typed.Dict.empty(
        key_type=_nbtypes.int64,
        value_type=_nbtypes.int32,
    )
    n_hits = numba.typed.Dict.empty(
        key_type=_nbtypes.int64,
        value_type=_nbtypes.int32,
    )
    return log_odds, n_obs, n_hits


def make_max_logit_dict():
    """Return a Numba typed dict (int64 -> float32) for MapMOS max-logit tracking.

    Separate constructor so callers on the geometry-only path don't pay
    the allocation cost. Plan non-negotiable #18 / #19: stored values are
    RAW logits aligned with the unique_keys export.
    """
    _require_numba()
    return numba.typed.Dict.empty(
        key_type=_nbtypes.int64,
        value_type=_nbtypes.float32,
    )


def update_sweep_log_odds(
    sweep_origin: np.ndarray,
    endpoints: np.ndarray,
    is_ground: np.ndarray | None,
    chunk_origin: np.ndarray,
    voxel_size: float,
    margin_voxels: float,
    max_length_m: float,
    log_odds,
    n_obs,
    n_hits,
    l_occ: float,
    l_free: float,
    log_odds_clamp: float,
    endpoint_priors: np.ndarray | None = None,
    alpha: float = 0.0,
    max_logit=None,
) -> None:
    """Update the per-voxel log-odds accumulators for one sweep's rays.

    Backwards-compatible signature: callers that don't pass
    `endpoint_priors`/`alpha`/`max_logit` get the geometry-only path
    bit-identically. When priors are present, the endpoint update folds
    in `+ alpha * prior[i]` and (if max_logit is provided) tracks the
    RAW max prior per voxel for the under-evidenced rescue path.
    """
    _require_numba()
    if is_ground is None:
        ig = np.zeros(len(endpoints), dtype=np.bool_)
    else:
        ig = np.asarray(is_ground, dtype=np.bool_)

    # Empty-shape sentinel means "no priors" inside the kernel. We pass a
    # length-0 float32 array (NOT None) because Numba kernel signatures
    # can't accept None for typed array arguments.
    if endpoint_priors is None:
        priors_arr = np.empty(0, dtype=np.float32)
    else:
        priors_arr = np.ascontiguousarray(endpoint_priors, dtype=np.float32)

    # max_logit dict is always passed; the kernel only writes when
    # track_max_logit is True. An empty placeholder dict keeps the
    # signature stable without paying the allocation cost on every call.
    track = max_logit is not None
    if track:
        ml = max_logit
    else:
        ml = make_max_logit_dict()

    _update_sweep_numba(
        float(sweep_origin[0]),
        float(sweep_origin[1]),
        float(sweep_origin[2]),
        endpoints.astype(np.float64),
        ig,
        float(chunk_origin[0]),
        float(chunk_origin[1]),
        float(chunk_origin[2]),
        float(voxel_size),
        float(margin_voxels),
        float(max_length_m),
        log_odds,
        n_obs,
        n_hits,
        float(l_occ),
        float(l_free),
        float(log_odds_clamp),
        priors_arr,
        float(alpha),
        ml,
        bool(track),
    )


def extract_log_odds_arrays(
    log_odds,
    n_obs,
    n_hits,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert the three typed dicts to sorted numpy arrays.

    Returns (unique_keys, lo_vals, n_obs_vals, n_hits_vals) sorted by key.
    """
    _require_numba()
    n = len(log_odds)
    if n == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )
    keys_buf = np.empty(n, dtype=np.int64)
    lo_buf = np.empty(n, dtype=np.float32)
    n_obs_buf = np.empty(n, dtype=np.int32)
    n_hits_buf = np.empty(n, dtype=np.int32)
    _extract_arrays_numba(log_odds, n_obs, n_hits, keys_buf, lo_buf, n_obs_buf, n_hits_buf)
    order = np.argsort(keys_buf)
    return keys_buf[order], lo_buf[order], n_obs_buf[order], n_hits_buf[order]


def extract_max_logit_array(
    unique_keys: np.ndarray,
    max_logit,
) -> np.ndarray:
    """Return max_logit values aligned with `unique_keys` (sorted by key).

    Plan non-negotiable #19: alignment with `unique_keys` is contract.
    Voxels that received no MapMOS prior float to a sentinel of -inf
    (representable in float32) so the under-evidenced rescue threshold
    cleanly excludes them — no MapMOS evidence means no rescue.

    Lookup happens inside `_extract_max_logit_numba` (mirrors
    _extract_arrays_numba): on chunks with ~10^6 occupied voxels the prior
    Python-side loop paid one Python↔Numba boundary crossing per typed-dict
    lookup, which dominates this step's runtime once MapMOS is shipping
    real logits. Pre-fill `out` with the sentinel; the JIT kernel only
    writes entries that exist in the dict.
    """
    if unique_keys.size == 0:
        return np.empty(0, dtype=np.float32)
    _require_numba()
    NEG_INF = np.float32(-1e18)
    out = np.full(unique_keys.shape[0], NEG_INF, dtype=np.float32)
    _extract_max_logit_numba(unique_keys, max_logit, out)
    return out
