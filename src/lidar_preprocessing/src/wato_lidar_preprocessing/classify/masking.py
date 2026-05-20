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
) -> SweepMaskResult:
    """Compute dynamic mask for one sweep, save it, return per-sweep stats.

    `keys` is always full-length (matches xyz from the world NPZ), so the
    saved mask length matches the downstream xyz array — that's the
    contract proposal_generation and the occupancy export depend on.
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

    # Defensive ground filter: voxel-level static/dynamic decisions don't
    # perfectly respect per-point ground status, so a per-point belt is
    # needed in BOTH endpoint strategies.
    #
    # skip_ray   — ground voxels never carved into log_odds at all, so they
    #              have no entry in not_dynamic_arr and would otherwise end
    #              up in the dynamic bucket above.
    # skip_endpoint — the ground voxel itself isn't hit by the ground point
    #              (endpoint skipped), but ANOTHER point landing in the same
    #              voxel (e.g. a passing car bumper that happens to share
    #              the ground voxel) does fire +l_occ. The voxel then has
    #              n_hits>0 + low p_occ + many observations, and falls
    #              through every not_dynamic gate — getting labeled dynamic
    #              and dragging the ground point with it.
    #
    # Patchwork-detected ground is never dynamic by definition, so force
    # mask=False at those points regardless of voxel-level decision. Only
    # the log-odds path needs this: the persistence path makes per-point
    # decisions and doesn't have the mixed-voxel issue.
    if (
        cfg.classification_method == "log_odds"
        and ground_mask_cache_i is not None
    ):
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
