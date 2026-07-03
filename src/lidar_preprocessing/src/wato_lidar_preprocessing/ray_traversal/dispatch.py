"""Public dispatch surface for ray traversal.

Imports numba and the JIT kernel lazily. When numba is unavailable, every
public function raises ImportError with a clear remediation message — no
silent slow fallback.
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
        _accumulate_cov_numba,
        _apply_global_map_boost_numba,
        _build_normal_dicts_numba,
        _extract_arrays_numba,
        _extract_cov_numba,
        _update_sweep_numba,
    )

    _NUMBA_AVAILABLE = True
except Exception as _e:  # noqa: BLE001 — half-installed numba can raise AttributeError
    _NUMBA_AVAILABLE = False
    _NUMBA_IMPORT_ERROR = _e
    log.warning(
        "numba unavailable (%s: %s); log-odds classification will hard-fail "
        "when invoked.",
        type(_e).__name__,
        _e,
    )


_MISSING_NUMBA_MSG = (
    "log-odds classification requires numba; install it "
    "(pip install 'numba>=0.59')"
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


def make_cov_dicts() -> tuple:
    """Return (cnt, sx, sy, sz, sxx, syy, szz, sxy, sxz, syz) typed dicts for
    per-voxel surface-normal estimation. cnt is int32; the rest are float64."""
    _require_numba()
    cnt = numba.typed.Dict.empty(key_type=_nbtypes.int64, value_type=_nbtypes.int32)
    floats = [
        numba.typed.Dict.empty(key_type=_nbtypes.int64, value_type=_nbtypes.float64)
        for _ in range(9)
    ]
    return (cnt, *floats)


def make_normal_dicts() -> tuple:
    """Return (nx, ny, nz) typed dicts (int64 -> float32) for the carve gate."""
    _require_numba()
    return tuple(
        numba.typed.Dict.empty(key_type=_nbtypes.int64, value_type=_nbtypes.float32)
        for _ in range(3)
    )


def accumulate_cov(
    endpoints: np.ndarray,
    is_ground: np.ndarray | None,
    chunk_origin: np.ndarray,
    voxel_size: float,
    cov_dicts: tuple,
) -> None:
    """Accumulate per-voxel point moments from one sweep's non-ground endpoints."""
    _require_numba()
    if endpoints.shape[0] == 0:
        return
    if is_ground is None:
        ig = np.zeros(len(endpoints), dtype=np.bool_)
    else:
        ig = np.asarray(is_ground, dtype=np.bool_)
    _accumulate_cov_numba(
        endpoints.astype(np.float64),
        ig,
        float(chunk_origin[0]),
        float(chunk_origin[1]),
        float(chunk_origin[2]),
        float(voxel_size),
        *cov_dicts,
    )


def compute_normals(cov_dicts: tuple, min_pts: int = 3) -> tuple:
    """Per-voxel surface normal (smallest-eigenvector of the return covariance)
    as (nx, ny, nz) dicts. Only planar voxels with >= min_pts returns get a
    normal; the rest are omitted (the carve gate treats them as carvable).
    """
    _require_numba()
    cnt = cov_dicts[0]
    n = len(cnt)
    nx, ny, nz = make_normal_dicts()
    if n == 0:
        return nx, ny, nz

    keys = np.empty(n, dtype=np.int64)
    cnt_a = np.empty(n, dtype=np.int64)
    moments = [np.empty(n, dtype=np.float64) for _ in range(9)]
    _extract_cov_numba(*cov_dicts, keys, cnt_a, *moments)
    sx, sy, sz, sxx, syy, szz, sxy, sxz, syz = moments

    inv = 1.0 / np.maximum(cnt_a, 1)
    mx, my, mz = sx * inv, sy * inv, sz * inv
    # Covariance components (population): E[xx] - E[x]E[x], ...
    cxx = sxx * inv - mx * mx
    cyy = syy * inv - my * my
    czz = szz * inv - mz * mz
    cxy = sxy * inv - mx * my
    cxz = sxz * inv - mx * mz
    cyz = syz * inv - my * mz

    cov = np.empty((n, 3, 3), dtype=np.float64)
    cov[:, 0, 0] = cxx
    cov[:, 1, 1] = cyy
    cov[:, 2, 2] = czz
    cov[:, 0, 1] = cov[:, 1, 0] = cxy
    cov[:, 0, 2] = cov[:, 2, 0] = cxz
    cov[:, 1, 2] = cov[:, 2, 1] = cyz

    # eigh returns ascending eigenvalues; smallest-eigenvector is the normal.
    evals, evecs = np.linalg.eigh(cov)
    normals = evecs[:, :, 0]  # (n, 3)
    # Planarity: smallest eigenvalue much smaller than largest → a real plane.
    planar = evals[:, 0] <= 0.1 * np.maximum(evals[:, 2], 1e-12)
    valid = (cnt_a >= min_pts) & planar

    _build_normal_dicts_numba(
        keys,
        normals[:, 0].astype(np.float32),
        normals[:, 1].astype(np.float32),
        normals[:, 2].astype(np.float32),
        valid,
        nx,
        ny,
        nz,
    )
    return nx, ny, nz


def update_sweep_log_odds(
    sweep_origin: np.ndarray,
    endpoints: np.ndarray,
    is_ground: np.ndarray | None,
    chunk_origin: np.ndarray,
    voxel_size: float,
    margin_m: float,
    max_length_m: float,
    log_odds,
    n_obs,
    n_hits,
    l_occ: float,
    l_free: float,
    log_odds_clamp: float,
    d_star: float,
    normal_dicts: tuple | None = None,
    grazing_cos: float = 0.0,
) -> None:
    """Plumb a single sweep through the Numba DDA kernel.

    margin_m: carve stop-distance [m]. d_star: range-credibility crossover [m].
    normal_dicts: (nx, ny, nz) from compute_normals — occupied voxels grazed by
    the ray (|ray·n| < grazing_cos) are not carved. None disables the gate.
    """
    _require_numba()
    if is_ground is None:
        ig = np.zeros(len(endpoints), dtype=np.bool_)
    else:
        ig = np.asarray(is_ground, dtype=np.bool_)
    if normal_dicts is None:
        nx, ny, nz = make_normal_dicts()
    else:
        nx, ny, nz = normal_dicts
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
        float(margin_m),
        float(max_length_m),
        log_odds,
        n_obs,
        n_hits,
        float(l_occ),
        float(l_free),
        float(log_odds_clamp),
        float(d_star),
        nx,
        ny,
        nz,
        float(grazing_cos),
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
