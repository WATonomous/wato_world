"""Pure-Python parity kernel for the Numba DDA.

Used ONLY by the parity test in tests/test_ray_traversal.py. Never imported
by dispatch.py at runtime — the Python path is too slow for real chunks.
Kept as a readable reference for the DDA arithmetic; the parity test holds
the Numba kernel to its semantics.
"""

from __future__ import annotations

import math

import numpy as np

from ._keys import AXIS_RANGE, SHIFT_X, SHIFT_Y


def _update_sweep_python(
    sweep_origin: np.ndarray,
    endpoints: np.ndarray,
    is_ground: np.ndarray | None,
    chunk_origin: np.ndarray,
    voxel_size: float,
    margin_voxels: float,
    max_length_m: float,
    log_odds: dict,
    n_obs: dict,
    n_hits: dict,
    l_occ: float,
    l_free: float,
    log_odds_clamp: float,
    r_max: float = 200.0,
    use_range_weight: bool = False,
) -> None:
    """Reference DDA — same arithmetic as _update_sweep_numba in pure Python."""
    ox = float(sweep_origin[0])
    oy = float(sweep_origin[1])
    oz = float(sweep_origin[2])
    cox = float(chunk_origin[0])
    coy = float(chunk_origin[1])
    coz = float(chunk_origin[2])

    INF = 1e18
    for i in range(len(endpoints)):
        is_g = bool(is_ground[i]) if is_ground is not None else False
        ex = float(endpoints[i, 0])
        ey = float(endpoints[i, 1])
        ez = float(endpoints[i, 2])

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

        stop_t = length - margin_voxels * voxel_size

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

            if not (0 <= cx < AXIS_RANGE and 0 <= cy < AXIS_RANGE and 0 <= cz < AXIS_RANGE):
                continue

            if use_range_weight:
                r_star_t = r_max / t_entry
                if r_star_t > 1.0:
                    r_star_t = 1.0
            else:
                r_star_t = 1.0

            key = (cx << SHIFT_X) | (cy << SHIFT_Y) | cz
            old_lo = log_odds.get(key, 0.0)
            new_lo = np.float32(old_lo) - np.float32(l_free * r_star_t)
            if new_lo < np.float32(-log_odds_clamp):
                new_lo = np.float32(-log_odds_clamp)
            log_odds[key] = new_lo
            n_obs[key] = n_obs.get(key, np.int32(0)) + np.int32(1)

        if not is_g and 0 <= exi < AXIS_RANGE and 0 <= eyi < AXIS_RANGE and 0 <= ezi < AXIS_RANGE:
            key = (exi << SHIFT_X) | (eyi << SHIFT_Y) | ezi
            old_lo = log_odds.get(key, 0.0)
            new_lo = np.float32(old_lo) + np.float32(l_occ * r_star_endpoint)
            if new_lo > np.float32(log_odds_clamp):
                new_lo = np.float32(log_odds_clamp)
            log_odds[key] = new_lo
            n_obs[key] = n_obs.get(key, np.int32(0)) + np.int32(1)
            n_hits[key] = n_hits.get(key, np.int32(0)) + np.int32(1)
