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
        _apply_global_map_boost_numba,
        _extract_arrays_numba,
        _update_sweep_numba,
    )

    _NUMBA_AVAILABLE = True
except Exception as _e:  # noqa: BLE001 — half-installed numba can raise AttributeError
    _NUMBA_AVAILABLE = False
    _NUMBA_IMPORT_ERROR = _e
    log.warning(
        "numba unavailable (%s: %s); log-odds classification will hard-fail "
        "when invoked. Persistence-only classification still works.",
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
    r_max: float = 200.0,
    use_range_weight: bool = False,
) -> None:
    """Plumb a single sweep through the Numba DDA kernel."""
    _require_numba()
    if is_ground is None:
        ig = np.zeros(len(endpoints), dtype=np.bool_)
    else:
        ig = np.asarray(is_ground, dtype=np.bool_)
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
        float(r_max),
        bool(use_range_weight),
    )


def apply_global_map_boost(
    hit_keys: np.ndarray,
    hit_r_star: np.ndarray,
    l_occ_boost: float,
    clamp: float,
    log_odds,
) -> None:
    """UniLiPs IWU boost for endpoints matched in the global static map.

    For each unique voxel key the sweep hit, take the max r_star across the
    sweep's hits and add `l_occ_boost * max_r_star` to that voxel's
    log_odds, clamped at ±clamp.

    Only touches log_odds — n_hits stays backed by real sweep returns so
    the has_hits / min_occupied_hits gate isn't bypassed by the prior.
    """
    _require_numba()
    if hit_keys.size == 0:
        return
    unique_keys, inv = np.unique(hit_keys, return_inverse=True)
    max_r_star = np.zeros(len(unique_keys), dtype=np.float32)
    np.maximum.at(max_r_star, inv, hit_r_star.astype(np.float32))
    _apply_global_map_boost_numba(
        unique_keys.astype(np.int64),
        max_r_star,
        float(l_occ_boost),
        float(clamp),
        log_odds,
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
    _extract_arrays_numba(
        log_odds, n_obs, n_hits, keys_buf, lo_buf, n_obs_buf, n_hits_buf
    )
    order = np.argsort(keys_buf)
    return keys_buf[order], lo_buf[order], n_obs_buf[order], n_hits_buf[order]
