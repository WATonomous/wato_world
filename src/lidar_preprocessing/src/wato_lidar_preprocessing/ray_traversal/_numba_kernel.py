"""Numba-jitted Amanatides-Woo 3D-DDA ray traversal.

Imported lazily by dispatch.py only when numba is installed. Inlines the
voxel-key packing math from _keys.py — Numba treats module-level ints as
compile-time constants so there's no perf cost vs. hard-coding 40/20.
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
) -> None:
    """Update log_odds, n_obs, n_hits for one sweep's rays.

    Canonical Amanatides-Woo init: derive t_max per axis from frac (position
    within current voxel), which is sign-symmetric and easier to audit than
    the (cx+1)*size - ox formulation.
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

        # Fractional offset of origin within its current voxel, in [0, voxel_size).
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

        # Stop carving margin_voxels before endpoint so the measured surface
        # is never claimed as free space.
        stop_t = length - margin_voxels * voxel_size

        # Traverse free-space voxels (sensor-origin voxel is never emitted).
        while True:
            if t_max_x <= t_max_y and t_max_x <= t_max_z:
                if t_max_x >= stop_t:
                    break
                cx += sx
                t_max_x += t_delta_x
            elif t_max_y <= t_max_z:
                if t_max_y >= stop_t:
                    break
                cy += sy
                t_max_y += t_delta_y
            else:
                if t_max_z >= stop_t:
                    break
                cz += sz
                t_max_z += t_delta_z

            if (
                cx < 0 or cx >= AXIS_RANGE
                or cy < 0 or cy >= AXIS_RANGE
                or cz < 0 or cz >= AXIS_RANGE
            ):
                continue

            key = (cx << SHIFT_X) | (cy << SHIFT_Y) | cz
            old_lo = log_odds.get(key, numba.float32(0.0))
            new_lo = old_lo - numba.float32(l_free)
            if new_lo < numba.float32(-log_odds_clamp):
                new_lo = numba.float32(-log_odds_clamp)
            log_odds[key] = new_lo
            n_obs[key] = n_obs.get(key, numba.int32(0)) + numba.int32(1)

        # Endpoint voxel: occupied observation (skipped for ground points).
        if not is_ground[i] and (
            exi >= 0 and exi < AXIS_RANGE
            and eyi >= 0 and eyi < AXIS_RANGE
            and ezi >= 0 and ezi < AXIS_RANGE
        ):
            key = (exi << SHIFT_X) | (eyi << SHIFT_Y) | ezi
            old_lo = log_odds.get(key, numba.float32(0.0))
            new_lo = old_lo + numba.float32(l_occ)
            if new_lo > numba.float32(log_odds_clamp):
                new_lo = numba.float32(log_odds_clamp)
            log_odds[key] = new_lo
            n_obs[key] = n_obs.get(key, numba.int32(0)) + numba.int32(1)
            n_hits[key] = n_hits.get(key, numba.int32(0)) + numba.int32(1)


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
    log_odds in the same call (see _update_sweep_numba). Iterating log_odds
    is therefore the canonical key enumeration; n_obs/n_hits are looked up
    by key. If a future code path increments n_obs without touching
    log_odds, those keys would be silently dropped here.
    """
    i = 0
    for k, v in log_odds.items():
        keys_out[i] = k
        lo_out[i] = v
        n_obs_out[i] = n_obs.get(k, numba.int32(0))
        n_hits_out[i] = n_hits.get(k, numba.int32(0))
        i += 1
