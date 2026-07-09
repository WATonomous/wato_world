"""Pass 2 — per-sweep dynamic-mask resolution + static/dynamic cloud build."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wato_common.artifact_store import (
    aw_dynamic_mask_path,
    dynamic_mask_path,
    local_path,
)
from wato_lidar_preprocessing.voxel import keys_in_sorted

from .io_helpers import load_world_xyz_intensity


@dataclass
class SweepMaskResult:
    """xyz/intensity slices are appended to the chunk-level static/dynamic
    clouds by the caller. Slice fields are None when the corresponding count
    is 0.
    """

    n_static: int
    n_dynamic: int
    mask_uri: str
    static_xyz: np.ndarray | None = None
    static_intensity: np.ndarray | None = None
    dyn_xyz: np.ndarray | None = None
    dyn_intensity: np.ndarray | None = None
    dyn_sweep_id: np.ndarray | None = None


def apply_classification_to_sweep(
    row: dict,
    sweep_id: int,
    keys: np.ndarray,
    static_arr: np.ndarray,
    not_dynamic_arr: np.ndarray,
    xyz_cache_i: np.ndarray | None,
    intensity_cache_i: np.ndarray | None,
    ground_mask_cache_i: np.ndarray | None,
    sweep_origin: np.ndarray | None,
    bag_id: str,
    chunk_id: str,
    any_intensity: bool,
    *,
    dynamic_min_range_m: float = 0.0,
    write_aw_snapshot: bool = False,
) -> SweepMaskResult:
    """Compute dynamic mask for one sweep, save it, return per-sweep stats.

    `keys` is always full-length (matches xyz from the world NPZ) so the
    saved mask is length-aligned with the downstream xyz array.

    Pure Amanatides-Woo: a point is dynamic iff its voxel is not in
    not_dynamic_arr (static + free_only + under_evidenced + ambiguous + the
    carved-noise bucket) and not flagged ground by Patchwork++. Then, when
    dynamic_min_range_m > 0, points within that horizontal range of the
    sensor (`sweep_origin`) are forced non-dynamic — near the ego the return
    is ego self-returns / near clutter and carving is maximal, so AW can't
    reliably call motion there. No MF-MOS involvement — that lives entirely
    in the mf_mos/ module (`--seg mos`).

    write_aw_snapshot additionally saves the mask to aw_dynamic_mask.npy.
    Set on the union path only: union overwrites dynamic_mask.npy with the
    fused verdict, so keep_aw_dynamic needs AW's own verdict preserved
    separately to stay re-fusable.
    """
    n = keys.shape[0]
    has_intensity = bool(row.get("has_intensity", False))
    dyn_uri = dynamic_mask_path(bag_id, chunk_id, sweep_id)

    if n == 0:
        mask = np.zeros(0, dtype=bool)
        np.save(local_path(dyn_uri), mask)
        if write_aw_snapshot:
            np.save(local_path(aw_dynamic_mask_path(bag_id, chunk_id, sweep_id)), mask)
        return SweepMaskResult(n_static=0, n_dynamic=0, mask_uri=dyn_uri)

    # not_dynamic_arr covers static + free-only + under-evidenced-with-hits
    # + ambiguous voxels (log_odds mode); static only (persistence mode).
    mask = ~keys_in_sorted(keys, not_dynamic_arr)

    # Patchwork++ ground mask is authoritative: ground points must never
    # appear in dynamic_map.npz. The not_dynamic_arr classification doesn't
    # reliably catch them — ground voxels can fall through whenever no
    # non-ground ray traverses them (skip_endpoint), aren't traversed at all
    # (skip_ray), or fall below the persistence threshold.
    if ground_mask_cache_i is not None:
        mask &= ~ground_mask_cache_i

    # Resolve xyz/intensity once — needed by the near-range gate below and by
    # the cloud build at the end. Uses the in-memory cache when available.
    xyz = xyz_cache_i
    intensity = intensity_cache_i

    # Near-range dynamic exclusion. Only loads xyz if a point is still a
    # dynamic candidate (mask.any()), so all-static sweeps skip the read.
    if dynamic_min_range_m > 0.0 and sweep_origin is not None and mask.any():
        if xyz is None:
            xyz, intensity = load_world_xyz_intensity(row["world_path"])
        d_xy = np.hypot(xyz[:, 0] - sweep_origin[0], xyz[:, 1] - sweep_origin[1])
        mask &= d_xy >= dynamic_min_range_m

    n_dyn = int(mask.sum())

    # is_static must use the static_arr lookup, NOT `~mask`: `~mask` would
    # include free-only and under-evidenced-with-hits voxels and pollute
    # static_map.npz with low-confidence returns.
    is_static = keys_in_sorted(keys, static_arr)
    n_static = int(is_static.sum())

    # Ground points belong in ground.npz only. Without this filter, road
    # surfaces (hit by every drive-over) pass the static-voxel test and
    # pollute static_map.npz.
    if ground_mask_cache_i is not None:
        is_static &= ~ground_mask_cache_i
        n_static = int(is_static.sum())

    np.save(local_path(dyn_uri), mask)
    if write_aw_snapshot:
        np.save(local_path(aw_dynamic_mask_path(bag_id, chunk_id, sweep_id)), mask)

    result = SweepMaskResult(n_static=n_static, n_dynamic=n_dyn, mask_uri=dyn_uri)

    if n_static == 0 and n_dyn == 0:
        return result

    # xyz may already be resolved (cache hit, or loaded by the near-range gate).
    if xyz is None:
        xyz, intensity = load_world_xyz_intensity(row["world_path"])

    static_mask = is_static
    if n_static > 0:
        result.static_xyz = xyz[static_mask]
        if any_intensity:
            if has_intensity and intensity is not None:
                result.static_intensity = intensity[static_mask].astype(np.float32)
            else:
                result.static_intensity = np.zeros(n_static, dtype=np.float32)

    if n_dyn > 0:
        result.dyn_xyz = xyz[mask]
        result.dyn_sweep_id = np.full(n_dyn, sweep_id, dtype=np.int32)
        if any_intensity:
            if has_intensity and intensity is not None:
                result.dyn_intensity = intensity[mask].astype(np.float32)
            else:
                result.dyn_intensity = np.zeros(n_dyn, dtype=np.float32)

    return result
