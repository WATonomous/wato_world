"""Numba-jitted Amanatides-Woo 3D-DDA ray traversal.

Imported lazily by dispatch.py only when numba is installed.
"""

from __future__ import annotations

import math

import numpy as np
import numba

from ._keys import AXIS_RANGE, SHIFT_X, SHIFT_Y


@numba.njit(cache=True)
def _accumulate_cov_numba(
    endpoints: np.ndarray,  # (N, 3) float64
    is_ground: np.ndarray,  # (N,) bool
    cox: float,
    coy: float,
    coz: float,
    voxel_size: float,
    cnt: numba.typed.Dict,  # int64 -> int32
    sx: numba.typed.Dict,  # int64 -> float64 (Σx)
    sy: numba.typed.Dict,
    sz: numba.typed.Dict,
    sxx: numba.typed.Dict,  # int64 -> float64 (Σx²)
    syy: numba.typed.Dict,
    szz: numba.typed.Dict,
    sxy: numba.typed.Dict,  # int64 -> float64 (Σxy)
    sxz: numba.typed.Dict,
    syz: numba.typed.Dict,
) -> None:
    """Accumulate per-voxel count + 1st/2nd moments of non-ground returns, for
    per-voxel surface-normal (PCA) estimation ahead of the carving pass."""
    for i in range(endpoints.shape[0]):
        if is_ground[i]:
            continue
        ex = endpoints[i, 0]
        ey = endpoints[i, 1]
        ez = endpoints[i, 2]
        cx = int(math.floor((ex - cox) / voxel_size))
        cy = int(math.floor((ey - coy) / voxel_size))
        cz = int(math.floor((ez - coz) / voxel_size))
        if (
            cx < 0
            or cx >= AXIS_RANGE
            or cy < 0
            or cy >= AXIS_RANGE
            or cz < 0
            or cz >= AXIS_RANGE
        ):
            continue
        key = (cx << SHIFT_X) | (cy << SHIFT_Y) | cz
        cnt[key] = numba.int32(cnt.get(key, numba.int32(0)) + numba.int32(1))
        sx[key] = sx.get(key, 0.0) + ex
        sy[key] = sy.get(key, 0.0) + ey
        sz[key] = sz.get(key, 0.0) + ez
        sxx[key] = sxx.get(key, 0.0) + ex * ex
        syy[key] = syy.get(key, 0.0) + ey * ey
        szz[key] = szz.get(key, 0.0) + ez * ez
        sxy[key] = sxy.get(key, 0.0) + ex * ey
        sxz[key] = sxz.get(key, 0.0) + ex * ez
        syz[key] = syz.get(key, 0.0) + ey * ez


@numba.njit(cache=True)
def _update_sweep_numba(
    ox: float,
    oy: float,
    oz: float,
    endpoints: np.ndarray,  # (N, 3) float64
    is_ground: np.ndarray,  # (N,) bool
    cox: float,
    coy: float,
    coz: float,
    voxel_size: float,
    margin_m: float,
    max_length_m: float,
    log_odds: numba.typed.Dict,  # int64 -> float32
    n_obs: numba.typed.Dict,  # int64 -> int32
    n_hits: numba.typed.Dict,  # int64 -> int32
    l_occ: float,
    l_free: float,
    log_odds_clamp: float,
    d_star: float,
    nx: numba.typed.Dict,  # int64 -> float32 surface-normal x (occupied voxels)
    ny: numba.typed.Dict,
    nz: numba.typed.Dict,
    grazing_cos: float,
) -> None:
    """Update log_odds, n_obs, n_hits for one sweep's rays.

    Range weighting scales each update by min(1, d_star/d) (endpoint d=length,
    carve d=t_entry). Carving stops margin_m short of the endpoint. Incidence
    gate: a voxel with a surface normal (nx/ny/nz) is not carved when grazed
    (|ray·n| < grazing_cos); voxels without a normal carve normally.
    """
    INF = 1e18
    for i in range(endpoints.shape[0]):
        ex = endpoints[i, 0]
        ey = endpoints[i, 1]
        ez = endpoints[i, 2]

        dx = ex - ox
        dy = ey - oy
        dz = ez - oz
        length = math.sqrt(dx * dx + dy * dy + dz * dz)

        if length < 1e-9 or length > max_length_m:
            continue

        r_star_endpoint = d_star / length
        if r_star_endpoint > 1.0:
            r_star_endpoint = 1.0

        inv_len = 1.0 / length
        dxn = dx * inv_len
        dyn = dy * inv_len
        dzn = dz * inv_len

        cx = int(math.floor((ox - cox) / voxel_size))
        cy = int(math.floor((oy - coy) / voxel_size))
        cz = int(math.floor((oz - coz) / voxel_size))
        exi = int(math.floor((ex - cox) / voxel_size))
        eyi = int(math.floor((ey - coy) / voxel_size))
        ezi = int(math.floor((ez - coz) / voxel_size))

        # frac_*: origin's offset within its current voxel, in [0, voxel_size).
        frac_x = (ox - cox) - cx * voxel_size
        frac_y = (oy - coy) - cy * voxel_size
        frac_z = (oz - coz) - cz * voxel_size

        if dxn > 1e-12:
            sx = 1
            t_delta_x = voxel_size / dxn
            t_max_x = (voxel_size - frac_x) / dxn
        elif dxn < -1e-12:
            sx = -1
            t_delta_x = voxel_size / (-dxn)
            t_max_x = frac_x / (-dxn)
        else:
            sx = 0
            t_delta_x = INF
            t_max_x = INF

        if dyn > 1e-12:
            sy = 1
            t_delta_y = voxel_size / dyn
            t_max_y = (voxel_size - frac_y) / dyn
        elif dyn < -1e-12:
            sy = -1
            t_delta_y = voxel_size / (-dyn)
            t_max_y = frac_y / (-dyn)
        else:
            sy = 0
            t_delta_y = INF
            t_max_y = INF

        if dzn > 1e-12:
            sz = 1
            t_delta_z = voxel_size / dzn
            t_max_z = (voxel_size - frac_z) / dzn
        elif dzn < -1e-12:
            sz = -1
            t_delta_z = voxel_size / (-dzn)
            t_max_z = frac_z / (-dzn)
        else:
            sz = 0
            t_delta_z = INF
            t_max_z = INF

        # Stop carving margin_m metres before the endpoint so the measured
        # surface is never claimed as free space.
        stop_t = length - margin_m

        # t_entry must be captured BEFORE incrementing t_max_<axis> so it's
        # the parametric distance the ray enters the new voxel (what r*_t
        # needs). Sensor-origin voxel is never emitted.
        while True:
            if t_max_x <= t_max_y and t_max_x <= t_max_z:
                if t_max_x >= stop_t:
                    break
                t_entry = t_max_x
                cx += sx
                t_max_x += t_delta_x
            elif t_max_y <= t_max_z:
                if t_max_y >= stop_t:
                    break
                t_entry = t_max_y
                cy += sy
                t_max_y += t_delta_y
            else:
                if t_max_z >= stop_t:
                    break
                t_entry = t_max_z
                cz += sz
                t_max_z += t_delta_z

            if (
                cx < 0
                or cx >= AXIS_RANGE
                or cy < 0
                or cy >= AXIS_RANGE
                or cz < 0
                or cz >= AXIS_RANGE
            ):
                continue

            r_star_t = d_star / t_entry
            if r_star_t > 1.0:
                r_star_t = 1.0

            key = (cx << SHIFT_X) | (cy << SHIFT_Y) | cz

            # Incidence gate (2.0 sentinel = no normal → carve normally).
            vnx = nx.get(key, numba.float32(2.0))
            if vnx < numba.float32(1.5):
                dot = (
                    dxn * vnx
                    + dyn * ny.get(key, numba.float32(0.0))
                    + dzn * nz.get(key, numba.float32(0.0))
                )
                if dot < 0.0:
                    dot = -dot
                if dot < grazing_cos:
                    continue

            old_lo = log_odds.get(key, numba.float32(0.0))
            new_lo = old_lo - numba.float32(l_free * r_star_t)
            if new_lo < numba.float32(-log_odds_clamp):
                new_lo = numba.float32(-log_odds_clamp)
            log_odds[key] = new_lo
            n_obs[key] = numba.int32(n_obs.get(key, numba.int32(0)) + numba.int32(1))

        # Endpoint voxel: occupied observation (skipped when is_ground).
        if not is_ground[i] and (
            exi >= 0
            and exi < AXIS_RANGE
            and eyi >= 0
            and eyi < AXIS_RANGE
            and ezi >= 0
            and ezi < AXIS_RANGE
        ):
            key = (exi << SHIFT_X) | (eyi << SHIFT_Y) | ezi
            old_lo = log_odds.get(key, numba.float32(0.0))
            new_lo = old_lo + numba.float32(l_occ * r_star_endpoint)
            if new_lo > numba.float32(log_odds_clamp):
                new_lo = numba.float32(log_odds_clamp)
            log_odds[key] = new_lo
            n_obs[key] = numba.int32(n_obs.get(key, numba.int32(0)) + numba.int32(1))
            n_hits[key] = numba.int32(n_hits.get(key, numba.int32(0)) + numba.int32(1))


@numba.njit(cache=True)
def _apply_global_map_boost_numba(
    unique_keys: np.ndarray,  # (M,) int64, deduped
    max_r_star: np.ndarray,  # (M,) float32, per-key max r*
    l_occ_boost: float,
    clamp: float,
    log_odds: numba.typed.Dict,  # int64 -> float32
) -> None:
    """Add l_occ_boost * max_r_star[k] to log_odds[k] for each unique key, clamped."""
    for i in range(unique_keys.shape[0]):
        k = unique_keys[i]
        boost = numba.float32(l_occ_boost * max_r_star[i])
        old = log_odds.get(k, numba.float32(0.0))
        new_lo = old + boost
        if new_lo > numba.float32(clamp):
            new_lo = numba.float32(clamp)
        log_odds[k] = new_lo


@numba.njit(cache=True)
def _extract_cov_numba(
    cnt: numba.typed.Dict,
    sx: numba.typed.Dict,
    sy: numba.typed.Dict,
    sz: numba.typed.Dict,
    sxx: numba.typed.Dict,
    syy: numba.typed.Dict,
    szz: numba.typed.Dict,
    sxy: numba.typed.Dict,
    sxz: numba.typed.Dict,
    syz: numba.typed.Dict,
    keys_out: np.ndarray,
    cnt_out: np.ndarray,
    sx_out: np.ndarray,
    sy_out: np.ndarray,
    sz_out: np.ndarray,
    sxx_out: np.ndarray,
    syy_out: np.ndarray,
    szz_out: np.ndarray,
    sxy_out: np.ndarray,
    sxz_out: np.ndarray,
    syz_out: np.ndarray,
) -> None:
    """Flatten the per-voxel moment dicts into parallel arrays (cnt is canonical)."""
    i = 0
    for k, v in cnt.items():
        keys_out[i] = k
        cnt_out[i] = v
        sx_out[i] = sx[k]
        sy_out[i] = sy[k]
        sz_out[i] = sz[k]
        sxx_out[i] = sxx[k]
        syy_out[i] = syy[k]
        szz_out[i] = szz[k]
        sxy_out[i] = sxy[k]
        sxz_out[i] = sxz[k]
        syz_out[i] = syz[k]
        i += 1


@numba.njit(cache=True)
def _build_normal_dicts_numba(
    keys: np.ndarray,
    nxa: np.ndarray,
    nya: np.ndarray,
    nza: np.ndarray,
    valid: np.ndarray,
    nx: numba.typed.Dict,
    ny: numba.typed.Dict,
    nz: numba.typed.Dict,
) -> None:
    """Populate nx/ny/nz typed dicts from normal arrays for valid voxels only."""
    for i in range(keys.shape[0]):
        if valid[i]:
            k = keys[i]
            nx[k] = numba.float32(nxa[i])
            ny[k] = numba.float32(nya[i])
            nz[k] = numba.float32(nza[i])


@numba.njit(cache=True)
def _extract_arrays_numba(
    log_odds: numba.typed.Dict,
    n_obs: numba.typed.Dict,
    n_hits: numba.typed.Dict,
    keys_out: np.ndarray,
    lo_out: np.ndarray,
    n_obs_out: np.ndarray,
    n_hits_out: np.ndarray,
) -> None:
    """Fill pre-allocated arrays from the three typed dicts.

    INVARIANT: every key written into n_obs or n_hits is also written into
    log_odds in the same call. log_odds is the canonical key enumeration;
    n_obs/n_hits are looked up by key. A code path that increments n_obs
    without touching log_odds would silently drop those keys here.
    """
    i = 0
    for k, v in log_odds.items():
        keys_out[i] = k
        lo_out[i] = v
        n_obs_out[i] = n_obs.get(k, numba.int32(0))
        n_hits_out[i] = n_hits.get(k, numba.int32(0))
        i += 1
