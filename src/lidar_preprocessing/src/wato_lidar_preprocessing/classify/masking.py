"""Pass 2 — per-sweep dynamic-mask resolution + static/dynamic cloud build."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wato_common.artifact_store import dynamic_mask_path, local_path
from wato_lidar_preprocessing.config import ComponentConfig

from .io_helpers import load_world_xyz_intensity


@dataclass
class SweepMaskResult:
    n_static: int
    n_dynamic: int
    mask_uri: str
    # xyz/intensity slices to be appended to the chunk-level static/dynamic
    # clouds. May be None when n_static/n_dynamic is 0.
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
    mf_mos_dynamic_arr: np.ndarray | None = None,
) -> SweepMaskResult:
    """Compute dynamic mask for one sweep, save it, return per-sweep stats.

    `keys` is always full-length (matches xyz from the world NPZ), so the
    saved mask length matches the downstream xyz array — that's the
    contract proposal_generation and the occupancy export depend on.

    `mf_mos_dynamic_arr` is an optional sorted int64 array of voxel keys
    that MF-MOS voted dynamic (aggregated across sweeps in Pass 1).  When
    provided, the fusion is done via searchsorted lookup — the same pattern
    as `not_dynamic_arr` — so it is composable with the AW classifier.
    """
    n = keys.shape[0]
    has_intensity = bool(row.get("has_intensity", False))
    dyn_uri = dynamic_mask_path(bag_id, chunk_id, sweep_id)

    if n == 0:
        mask = np.zeros(0, dtype=bool)
        np.save(local_path(dyn_uri), mask)
        return SweepMaskResult(n_static=0, n_dynamic=0, mask_uri=dyn_uri)

    # not_dynamic_arr covers static voxels plus (in log_odds mode)
    # free-only + under-evidenced-with-hits voxels.  Searchsorted is O(N log K).
    if not_dynamic_arr.size > 0:
        pos = np.searchsorted(not_dynamic_arr, keys)
        pos = np.clip(pos, 0, not_dynamic_arr.size - 1)
        is_not_dynamic = not_dynamic_arr[pos] == keys
    else:
        is_not_dynamic = np.zeros(n, dtype=bool)
    mask = ~is_not_dynamic  # True == dynamic, length == n_total

    # Patchwork++ per-point ground mask is authoritative: a point flagged
    # ground in sensor frame must never end up in dynamic_map.npz.
    #
    # skip_ray needed this because ground rays aren't traversed at all → the
    # endpoint voxel never enters log_odds → searchsorted yields "not found"
    # → mask=True.
    #
    # skip_endpoint nominally routes ground voxels to free_only_arr, BUT
    # only if at least one other ray ever traversed the voxel.  Sparse
    # beams, peripheral ground returns, or any sweep whose ground voxel is
    # only ever its own endpoint never get an entry in unique_keys, and
    # those points fall straight through to the dynamic bucket too.
    #
    # Persistence mode has the same exposure: any ground voxel whose
    # static-sweep-fraction is below threshold isn't in not_dynamic_arr
    # either.  Applying the filter unconditionally (whenever a ground mask
    # is available) is the simplest correct fix.
    if ground_mask_cache_i is not None:
        mask &= ~ground_mask_cache_i

    n_dyn = int(mask.sum())
    # is_static covers confident-static voxels only.  Critically NOT `~mask`:
    # `~mask` would include free-only voxels (where ground points land in
    # skip_endpoint mode) and under-evidenced-with-hits voxels, polluting
    # static_map.npz with ground / low-confidence returns.
    if static_arr.size > 0:
        pos_s = np.searchsorted(static_arr, keys)
        pos_s = np.clip(pos_s, 0, static_arr.size - 1)
        is_static = static_arr[pos_s] == keys
        n_static = int(is_static.sum())
    else:
        is_static = np.zeros(n, dtype=bool)
        n_static = 0

    # Patchwork++ ground belongs in ground.npz exclusively. Without this,
    # ground voxels — hit by every drive-over — pass the static-voxel test
    # and pollute static_map.npz, defeating the whole ground-extraction step.
    if ground_mask_cache_i is not None:
        is_static &= ~ground_mask_cache_i
        n_static = int(is_static.sum())

    # MF-MOS voxel-level fusion via searchsorted (same pattern as not_dynamic_arr).
    if mf_mos_dynamic_arr is not None and mf_mos_dynamic_arr.size > 0:
        pos = np.searchsorted(mf_mos_dynamic_arr, keys)
        pos = np.clip(pos, 0, mf_mos_dynamic_arr.size - 1)
        is_mf_mos_dyn = mf_mos_dynamic_arr[pos] == keys
        if cfg.mf_mos.fusion_mode == "union":
            mask = mask | is_mf_mos_dyn
        else:  # mfmos_only
            mask = is_mf_mos_dyn
        # Re-apply the ground filter: a non-ground moving point's vote
        # applies to the whole voxel, so without this re-AND, union OR /
        # mfmos_only overwrite would re-introduce co-voxel ground points
        # that the earlier `mask &= ~ground_mask` had removed.
        if ground_mask_cache_i is not None:
            mask &= ~ground_mask_cache_i
        n_dyn = int(mask.sum())
        # A point fusion-labelled dynamic can't remain in the static cloud.
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
