"""Numba-jitted Amanatides-Woo 3D-DDA ray traversal.

Imported lazily by dispatch.py only when numba is installed.
"""

from __future__ import annotations

import math

import numpy as np
import numba

from ._keys import AXIS_RANGE, SHIFT_X, SHIFT_Y


@numba.njit(cache=True)
def _update_sweep_numba(
    ox: float, oy: float, oz: float,
    endpoints: np.ndarray,     # (N, 3) float64
    is_ground: np.ndarray,     # (N,) bool
    cox: float, coy: float, coz: float,
    voxel_size: float,
    margin_voxels: float,
    max_length_m: float,
    log_odds: numba.typed.Dict,   # int64 -> float32
    n_obs: numba.typed.Dict,      # int64 -> int32
    n_hits: numba.typed.Dict,     # int64 -> int32
    l_occ: float,
    l_free: float,
    log_odds_clamp: float,
    r_max: float,
    use_range_weight: bool,
) -> None:
    """Update log_odds, n_obs, n_hits for one sweep's rays.

    Range-credibility weighting (UniLiPs r*_j) is active when
    use_range_weight is True: the endpoint l_occ update is scaled by
    min(1, r_max/length), and each traversed voxel's l_free update is
    scaled by min(1, r_max/t_entry) where t_entry is the parametric distance
    at which the ray enters that voxel. When False, both factors are 1.0.
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

        if use_range_weight:
            r_star_endpoint = r_max / length
            if r_star_endpoint > 1.0:
                r_star_endpoint = 1.0
        else:
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

        # Stop carving margin_voxels before the endpoint so the measured
        # surface is never claimed as free space.
        stop_t = length - margin_voxels * voxel_size

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
                cx < 0 or cx >= AXIS_RANGE
                or cy < 0 or cy >= AXIS_RANGE
                or cz < 0 or cz >= AXIS_RANGE
            ):
                continue

            if use_range_weight:
                r_star_t = r_max / t_entry
                if r_star_t > 1.0:
                    r_star_t = 1.0
            else:
                r_star_t = 1.0

            key = (cx << SHIFT_X) | (cy << SHIFT_Y) | cz
            old_lo = log_odds.get(key, numba.float32(0.0))
            new_lo = old_lo - numba.float32(l_free * r_star_t)
            if new_lo < numba.float32(-log_odds_clamp):
                new_lo = numba.float32(-log_odds_clamp)
            log_odds[key] = new_lo
            n_obs[key] = numba.int32(n_obs.get(key, numba.int32(0)) + numba.int32(1))

        # Endpoint voxel: occupied observation (skipped when is_ground).
        if not is_ground[i] and (
            exi >= 0 and exi < AXIS_RANGE
            and eyi >= 0 and eyi < AXIS_RANGE
            and ezi >= 0 and ezi < AXIS_RANGE
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
    unique_keys: np.ndarray,   # (M,) int64, deduped
    max_r_star: np.ndarray,    # (M,) float32, per-key max r*
    l_occ_boost: float,
    clamp: float,
    log_odds: numba.typed.Dict,   # int64 -> float32
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
