"""Pass 2 — per-sweep dynamic candidates + static/dynamic cloud build."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wato_common.artifacts.lidar_sweeps import write_dynamic_mask
from wato_common.artifact_store import dynamic_mask_path
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


def near_ego_mask(
    xyz: np.ndarray, sweep_origin: np.ndarray, min_range_m: float
) -> np.ndarray:
    """True for points within min_range_m horizontal range of the sensor.

    Within this radius returns are ego self-returns / near clutter, so a
    dynamic verdict is unreliable (cfg.dynamic_min_range_m).
    """
    d_xy = np.hypot(xyz[:, 0] - sweep_origin[0], xyz[:, 1] - sweep_origin[1])
    return d_xy < min_range_m


def dynamic_candidates(
    row: dict,
    keys: np.ndarray,
    dynamic_arr: np.ndarray,
    xyz_cache_i: np.ndarray | None,
    ground_mask_cache_i: np.ndarray | None,
    sweep_origin: np.ndarray | None,
    *,
    dynamic_min_range_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (mask, xyz): the points the voxel verdict calls dynamic.

    A point is a candidate IFF its voxel key is in `dynamic_arr`, it is not
    Patchwork++ ground, and it lies beyond dynamic_min_range_m of
    `sweep_origin`. Never-observed voxels are not dynamic. `keys` is always
    full-length, so the mask aligns with the world NPZ. xyz stays None for a
    sweep with no candidates and no cache, which skips the read.
    """
    mask = keys_in_sorted(keys, dynamic_arr)

    # Patchwork++ ground mask is authoritative: ground points must never
    # appear in dynamic_map.npz. Ground voxels can share keys with carved
    # voxels (e.g. a mover's wheels touching the road surface).
    if ground_mask_cache_i is not None:
        mask &= ~ground_mask_cache_i

    xyz = xyz_cache_i
    if mask.any() and xyz is None:
        xyz, _ = load_world_xyz_intensity(row["world_path"])

    if dynamic_min_range_m > 0.0 and sweep_origin is not None and mask.any():
        mask &= ~near_ego_mask(xyz, sweep_origin, dynamic_min_range_m)
    return mask, xyz


def apply_classification_to_sweep(
    row: dict,
    sweep_id: int,
    keys: np.ndarray,
    static_arr: np.ndarray,
    mask: np.ndarray,
    xyz_cache_i: np.ndarray | None,
    intensity_cache_i: np.ndarray | None,
    ground_mask_cache_i: np.ndarray | None,
    bag_id: str,
    chunk_id: str,
    any_intensity: bool,
) -> SweepMaskResult:
    """Save one sweep's final dynamic mask, return its stats and cloud slices."""
    n = keys.shape[0]
    has_intensity = bool(row.get("has_intensity", False))
    dyn_uri = dynamic_mask_path(bag_id, chunk_id, sweep_id)

    write_dynamic_mask(bag_id, chunk_id, sweep_id, mask)
    if n == 0:
        return SweepMaskResult(n_static=0, n_dynamic=0, mask_uri=dyn_uri)

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

    result = SweepMaskResult(n_static=n_static, n_dynamic=n_dyn, mask_uri=dyn_uri)

    if n_static == 0 and n_dyn == 0:
        return result

    xyz = xyz_cache_i
    intensity = intensity_cache_i
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
