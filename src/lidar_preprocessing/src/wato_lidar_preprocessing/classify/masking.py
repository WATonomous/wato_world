"""Pass 2 — per-sweep dynamic-mask resolution + static/dynamic cloud build."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from wato_common.artifact_store import dynamic_mask_path, local_path, static_mask_path
from wato_lidar_preprocessing.config import ComponentConfig

from .io_helpers import load_world_xyz_intensity

log = logging.getLogger(__name__)


@dataclass
class SweepMaskResult:
    """xyz/intensity slices are appended to the chunk-level static/dynamic
    clouds by the caller. Slice fields are None when the corresponding count
    is 0.
    """

    n_static: int
    n_dynamic: int
    mask_uri: str
    static_mask_uri: str
    static_xyz: np.ndarray | None = None
    static_intensity: np.ndarray | None = None
    dyn_xyz: np.ndarray | None = None
    dyn_intensity: np.ndarray | None = None
    dyn_sweep_id: np.ndarray | None = None
    dyn_lidar_id: np.ndarray | None = None


def apply_classification_to_sweep(
    row: dict,
    sweep_id: int,
    keys: np.ndarray,
    static_arr: np.ndarray,
    dynamic_arr: np.ndarray,
    xyz_cache_i: np.ndarray | None,
    intensity_cache_i: np.ndarray | None,
    ground_mask_cache_i: np.ndarray | None,
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
    any_intensity: bool,
    sweep_mf_mos_mask: np.ndarray | None = None,
) -> SweepMaskResult:
    """Compute dynamic mask for one sweep, save it, return per-sweep stats.

    `keys` is full-length (matches the world NPZ) so the saved mask stays
    length-aligned with the downstream xyz array.

    `sweep_mf_mos_mask` is point-aligned. Keeping fusion point-aligned avoids
    broadcasting one MF-MOS vote to unrelated points sharing an AW voxel.
    """
    n = keys.shape[0]
    has_intensity = bool(row.get("has_intensity", False))
    lidar_id = str(row["lidar_id"])
    dyn_uri = dynamic_mask_path(bag_id, chunk_id, lidar_id, sweep_id)
    static_uri = static_mask_path(bag_id, chunk_id, lidar_id, sweep_id)
    fusion_mode = cfg.profile_for(lidar_id).mf_mos.fusion_mode

    if n == 0:
        mask = np.zeros(0, dtype=bool)
        np.save(local_path(dyn_uri), mask)
        np.save(local_path(static_uri), mask)
        return SweepMaskResult(
            n_static=0,
            n_dynamic=0,
            mask_uri=dyn_uri,
            static_mask_uri=static_uri,
        )

    # Select only explicit AW dynamic evidence. Negating a conservative
    # not-dynamic set would incorrectly turn absent/unknown voxels dynamic.
    if dynamic_arr.size > 0:
        pos = np.searchsorted(dynamic_arr, keys)
        pos = np.clip(pos, 0, dynamic_arr.size - 1)
        mask = dynamic_arr[pos] == keys
    else:
        mask = np.zeros(n, dtype=bool)

    n_dyn = int(mask.sum())

    # is_static must use the static_arr lookup, NOT `~mask`: `~mask` would
    # include free-only and under-evidenced-with-hits voxels and pollute
    # static_map.npz with low-confidence returns.
    if static_arr.size > 0:
        pos_s = np.searchsorted(static_arr, keys)
        pos_s = np.clip(pos_s, 0, static_arr.size - 1)
        is_static = static_arr[pos_s] == keys
        n_static = int(is_static.sum())
    else:
        is_static = np.zeros(n, dtype=bool)
        n_static = 0

    # Ground points belong in ground.npz only. Without this filter, road
    # surfaces (hit by every drive-over) pass the static-voxel test and
    # pollute static_map.npz.
    if ground_mask_cache_i is not None:
        is_static &= ~ground_mask_cache_i
        n_static = int(is_static.sum())

    if sweep_mf_mos_mask is not None and fusion_mode != "independent":
        if sweep_mf_mos_mask.shape != (n,):
            raise ValueError(
                f"MF-MOS mask for ({lidar_id!r}, {sweep_id}) has shape "
                f"{sweep_mf_mos_mask.shape}, expected {(n,)}"
            )
        n_dyn_before_mf = n_dyn
        if fusion_mode == "union":
            mask = mask | sweep_mf_mos_mask
        else:  # mfmos_only
            mask = sweep_mf_mos_mask.copy()
        n_dyn = int(mask.sum())
        log.debug(
            "sweep %s mf_mos fusion: %d pts matched mf_mos voxels, "
            "%d pts flipped to dynamic (n_dyn %d→%d)",
            row.get("sweep_id"),
            int(sweep_mf_mos_mask.sum()),
            n_dyn - n_dyn_before_mf,
            n_dyn_before_mf,
            n_dyn,
        )
    # Final confident static excludes both selected motion and Patchwork's
    # candidate ground. Unknown/ambiguous/free-only points remain in neither
    # point mask.
    is_static &= ~mask
    if ground_mask_cache_i is not None:
        is_static &= ~ground_mask_cache_i
    n_static = int(is_static.sum())

    np.save(local_path(dyn_uri), mask)
    np.save(local_path(static_uri), is_static)

    result = SweepMaskResult(
        n_static=n_static,
        n_dynamic=n_dyn,
        mask_uri=dyn_uri,
        static_mask_uri=static_uri,
    )

    if n_static == 0 and n_dyn == 0:
        return result

    if xyz_cache_i is not None:
        xyz = xyz_cache_i
        intensity = intensity_cache_i
    else:
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
        result.dyn_lidar_id = np.full(n_dyn, lidar_id, dtype=f"<U{max(1, len(lidar_id))}")
        if any_intensity:
            if has_intensity and intensity is not None:
                result.dyn_intensity = intensity[mask].astype(np.float32)
            else:
                result.dyn_intensity = np.zeros(n_dyn, dtype=np.float32)

    return result
