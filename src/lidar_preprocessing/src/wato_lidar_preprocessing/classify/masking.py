"""Pass 2 — per-sweep dynamic-mask resolution + static/dynamic cloud build."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from wato_common.artifact_store import dynamic_mask_path, local_path
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
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
    any_intensity: bool,
    sweep_mf_mos_dynamic_arr: np.ndarray | None = None,
) -> SweepMaskResult:
    """Compute dynamic mask for one sweep, save it, return per-sweep stats.

    `keys` is always full-length (matches xyz from the world NPZ) so the
    saved mask is length-aligned with the downstream xyz array.

    `sweep_mf_mos_dynamic_arr`: optional sorted int64 voxel keys MF-MOS
    flagged dynamic FOR THIS SWEEP (built from the per-sweep mask in
    classify/pipeline.py and optionally AND-gated against the chunk-wide
    vote set). Fused via searchsorted, same pattern as `not_dynamic_arr`.

    fusion_mode="mfmos_only" drop behaviour: a point whose voxel is
    AW-DYNAMIC but NOT in sweep_mf_mos_dynamic_arr is dropped from both
    static_map.npz and dynamic_map.npz (intentional — MF-MOS is treated as
    the authoritative dynamic signal). Same drop already applies in all
    modes for points landing in AW free_only / under_evidenced / ambiguous
    voxels.
    """
    n = keys.shape[0]
    has_intensity = bool(row.get("has_intensity", False))
    dyn_uri = dynamic_mask_path(bag_id, chunk_id, sweep_id)

    if n == 0:
        mask = np.zeros(0, dtype=bool)
        np.save(local_path(dyn_uri), mask)
        return SweepMaskResult(n_static=0, n_dynamic=0, mask_uri=dyn_uri)

    # not_dynamic_arr covers static + free-only + under-evidenced-with-hits
    # + ambiguous voxels (log_odds mode); static only (persistence mode).
    if not_dynamic_arr.size > 0:
        pos = np.searchsorted(not_dynamic_arr, keys)
        pos = np.clip(pos, 0, not_dynamic_arr.size - 1)
        is_not_dynamic = not_dynamic_arr[pos] == keys
    else:
        is_not_dynamic = np.zeros(n, dtype=bool)
    mask = ~is_not_dynamic

    # Patchwork++ ground mask is authoritative: ground points must never
    # appear in dynamic_map.npz. The not_dynamic_arr classification doesn't
    # reliably catch them — ground voxels can fall through whenever no
    # non-ground ray traverses them (skip_endpoint), aren't traversed at all
    # (skip_ray), or fall below the persistence threshold.
    if ground_mask_cache_i is not None:
        mask &= ~ground_mask_cache_i

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

    if sweep_mf_mos_dynamic_arr is not None and sweep_mf_mos_dynamic_arr.size > 0:
        n_dyn_before_mf = n_dyn
        pos = np.searchsorted(sweep_mf_mos_dynamic_arr, keys)
        pos = np.clip(pos, 0, sweep_mf_mos_dynamic_arr.size - 1)
        is_mf_mos_dyn = sweep_mf_mos_dynamic_arr[pos] == keys
        if cfg.mf_mos.fusion_mode == "union":
            mask = mask | is_mf_mos_dyn
        else:  # mfmos_only
            mask = is_mf_mos_dyn
        # Re-apply ground filter: an MF-MOS vote applies to the whole voxel,
        # so without this re-AND, union/mfmos_only would re-introduce
        # co-voxel ground points that the earlier ground filter removed.
        if ground_mask_cache_i is not None:
            mask &= ~ground_mask_cache_i
        n_dyn = int(mask.sum())
        log.debug(
            "sweep %s mf_mos fusion: %d pts matched mf_mos voxels, "
            "%d pts flipped to dynamic (n_dyn %d→%d)",
            row.get("sweep_id"),
            int(is_mf_mos_dyn.sum()),
            n_dyn - n_dyn_before_mf,
            n_dyn_before_mf,
            n_dyn,
        )
        # A point now labelled dynamic can't also live in the static cloud.
        is_static = is_static & ~mask
        n_static = int(is_static.sum())

    np.save(local_path(dyn_uri), mask)

    result = SweepMaskResult(n_static=n_static, n_dynamic=n_dyn, mask_uri=dyn_uri)

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
        if any_intensity:
            if has_intensity and intensity is not None:
                result.dyn_intensity = intensity[mask].astype(np.float32)
            else:
                result.dyn_intensity = np.zeros(n_dyn, dtype=np.float32)

    return result
