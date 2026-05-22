"""Log-odds ray-casting classifier.

Two functions:
  - build_log_odds_grid:  loops sweeps through ray_traversal kernel,
                          returns sorted (keys, lo, n_obs, n_hits) arrays.
  - classify_from_log_odds: applies thresholds to return static_arr
                            and not_dynamic_arr + diagnostics.

Implements bug fix #2: under-evidenced voxels with hits go into
not_dynamic_arr (don't pollute the static cloud, don't false-positive
as dynamic).
"""

from __future__ import annotations

import logging

import numpy as np
from tqdm import tqdm

from wato_lidar_preprocessing.config import ComponentConfig
from wato_lidar_preprocessing.ray_traversal import (
    extract_log_odds_arrays,
    make_log_odds_dicts,
    update_sweep_log_odds,
)
from wato_lidar_preprocessing.voxel import voxel_indices

from .io_helpers import load_mf_mos_world_mask, load_world_full, sigmoid

log = logging.getLogger(__name__)


def build_log_odds_grid(
    meta_rows: list[dict],
    cfg: ComponentConfig,
    origin: np.ndarray,
    chunk_id: str,
    *,
    cache_xyz: bool,
    cache_intensity: bool = True,
    bag_id: str | None = None,
) -> tuple[
    list[np.ndarray | None],
    list[np.ndarray | None],
    list[np.ndarray | None],
    list[np.ndarray],
    dict[int, list[np.ndarray]],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """Pass 1 for the log-odds path.

    Loads each sweep once, populates the typed-dict log_odds accumulators
    via ray_traversal, builds sweep_keys for Pass 2, and (optionally) caches
    xyz/intensity/ground_mask in memory so Pass 2 doesn't re-read the NPZs.

    Returns:
        xyz_cache:           per-sweep xyz arrays (or None if not cached)
        intensity_cache:     per-sweep intensity arrays (or None)
        ground_mask_cache:   per-sweep ground masks (or None) — needed by
                             Pass 2 for bug fix #4 (skip_ray post-filter)
        sweep_keys:          per-sweep int64 voxel-key arrays (always full
                             length matching the world NPZ point count, so
                             dynamic_mask.npy stays length-aligned)
        frame_keys:          frame_id → list of key arrays for per-frame
                             voxel occupancy export
        log_odds_arrays:     (unique_keys, lo_vals, n_obs_vals, n_hits_vals,
                             mf_mos_votes_arr, n_sweep_hits_arr) sorted by key
    """
    log_odds_dict, n_obs_dict, n_hits_dict = make_log_odds_dicts()
    mf_mos_votes: dict[int, int] = {}
    n_sweep_hits: dict[int, int] = {}
    accumulate_votes = (
        bag_id is not None
        and cfg.mf_mos.enabled
        and cfg.mf_mos.fusion_mode != "independent"
    )

    xyz_cache: list[np.ndarray | None] = []
    intensity_cache: list[np.ndarray | None] = []
    ground_mask_cache: list[np.ndarray | None] = []
    sweep_keys: list[np.ndarray] = []
    frame_keys: dict[int, list[np.ndarray]] = {}

    for row in tqdm(
        meta_rows,
        desc=f"classify chunk {chunk_id} pass 1",
        unit="sweep",
    ):
        if row.get("valid") is False:
            xyz_cache.append(None)
            intensity_cache.append(None)
            ground_mask_cache.append(None)
            sweep_keys.append(np.empty(0, dtype=np.int64))
            continue

        xyz, intensity, sweep_origin, ground_mask = load_world_full(row["world_path"])

        if xyz.shape[0] == 0:
            xyz_cache.append(xyz if cache_xyz else None)
            intensity_cache.append(intensity if cache_xyz and cache_intensity else None)
            ground_mask_cache.append(ground_mask if cache_xyz else None)
            sweep_keys.append(np.empty(0, dtype=np.int64))
            continue

        # sweep_keys stays full length (matches xyz from world NPZ).  The
        # ground-aware filtering for ray traversal happens INSIDE the call
        # to update_sweep_log_odds via is_ground / endpoints filtering.
        keys = voxel_indices(xyz, origin, cfg.voxel_size_m, chunk_id=chunk_id)
        sweep_keys.append(keys)
        xyz_cache.append(xyz if cache_xyz else None)
        intensity_cache.append(intensity if cache_xyz and cache_intensity else None)
        ground_mask_cache.append(ground_mask if cache_xyz else None)

        fid = row.get("frame_id")
        if fid is not None:
            frame_keys.setdefault(int(fid), []).append(keys)

        if accumulate_votes:
            mf_mask = load_mf_mos_world_mask(
                bag_id, chunk_id, row, xyz.shape[0], cfg.filter_nonfinite_points
            )
            if mf_mask is not None:
                for k in np.unique(keys):
                    n_sweep_hits[int(k)] = n_sweep_hits.get(int(k), 0) + 1
                for k in np.unique(keys[mf_mask]):
                    mf_mos_votes[int(k)] = mf_mos_votes.get(int(k), 0) + 1

        if sweep_origin is None:
            log.warning(
                "chunk %s sweep %s: world NPZ missing 'origin' — "
                "ray traversal skipped; re-run deskew or use "
                "classification_method=persistence",
                chunk_id,
                row.get("sweep_id"),
            )
            continue

        if cfg.ground_endpoint_strategy == "skip_endpoint":
            # Traverse all rays; skip +l_occ at ground endpoints so air
            # voxels above the road accumulate free-space evidence while
            # road-surface voxels keep n_hits==0.
            endpoints_arr = xyz
            is_ground_arr = ground_mask  # None → all-False inside traversal
        else:  # "skip_ray"
            # Legacy: skip ground rays entirely. Bug fix #4 handles the
            # downstream consequence (ground voxels never enter log_odds)
            # via a post-filter in Pass 2 — sweep_keys stays full-length.
            endpoints_arr = xyz[~ground_mask] if ground_mask is not None else xyz
            is_ground_arr = None

        if endpoints_arr.shape[0] > 0:
            update_sweep_log_odds(
                sweep_origin,
                endpoints_arr,
                is_ground_arr,
                origin,
                cfg.voxel_size_m,
                cfg.free_space_margin_voxels,
                cfg.max_ray_length_m,
                log_odds_dict,
                n_obs_dict,
                n_hits_dict,
                cfg.l_occ,
                cfg.l_free,
                cfg.log_odds_clamp,
            )

    unique_keys, lo_vals, n_obs_vals, n_hits_vals = extract_log_odds_arrays(
        log_odds_dict, n_obs_dict, n_hits_dict
    )
    mf_mos_votes_arr = np.array(
        [mf_mos_votes.get(int(k), 0) for k in unique_keys], dtype=np.int32
    )
    n_sweep_hits_arr = np.array(
        [n_sweep_hits.get(int(k), 0) for k in unique_keys], dtype=np.int32
    )
    return (
        xyz_cache,
        intensity_cache,
        ground_mask_cache,
        sweep_keys,
        frame_keys,
        (unique_keys, lo_vals, n_obs_vals, n_hits_vals, mf_mos_votes_arr, n_sweep_hits_arr),
    )


def classify_from_log_odds(
    unique_keys: np.ndarray,
    lo_vals: np.ndarray,
    n_obs_vals: np.ndarray,
    n_hits_vals: np.ndarray,
    cfg: ComponentConfig,
    mf_mos_votes_arr: np.ndarray | None = None,
    n_sweep_hits_arr: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, int]]:
    """Return (static_arr, not_dynamic_arr, mf_mos_dynamic_arr, diag).

    static_arr           : voxels confident enough to label static (in the static cloud).
    not_dynamic_arr      : sorted union of (static + free-only + under-evidenced-with-hits);
                           Pass 2 uses this for the dynamic mask. Points in any of these
                           voxels get mask=False.
    mf_mos_dynamic_arr   : sorted voxel keys where MF-MOS votes exceed thresholds,
                           or None when vote aggregation was not active.
    """
    if unique_keys.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, None, {
            "n_evidenced": 0,
            "n_under_evidenced_with_hits": 0,
            "n_free_only": 0,
        }

    p_occ = sigmoid(lo_vals)
    evidenced = n_obs_vals >= cfg.min_observations
    has_hits = n_hits_vals >= cfg.min_occupied_hits

    # Confident static: passes all three gates.
    static_mask = evidenced & has_hits & (p_occ >= cfg.p_static_threshold)
    static_arr = unique_keys[static_mask]

    # Free-only: voxels only ever traversed (no endpoint hit).  Any point
    # landing here is ground noise and must not be labeled dynamic.
    free_only_mask = n_hits_vals == 0
    free_only_arr = unique_keys[free_only_mask]

    # BUG FIX #2: voxels with hits but too few observations get the benefit
    # of the doubt — into not_dynamic_arr (mask=False) but NOT into
    # static_arr (don't pollute the static cloud with low-confidence points).
    under_evidenced_with_hits_mask = (~evidenced) & has_hits
    under_arr = unique_keys[under_evidenced_with_hits_mask]

    # not_dynamic_arr is the union of (static + free_only + under-with-hits).
    parts = [a for a in (static_arr, free_only_arr, under_arr) if a.size > 0]
    if not parts:
        not_dynamic_arr = np.empty(0, dtype=np.int64)
    elif len(parts) == 1:
        not_dynamic_arr = parts[0]
    else:
        # np.unique sorts and dedupes — exactly what searchsorted needs.
        not_dynamic_arr = np.unique(np.concatenate(parts))

    # Voxel-level MF-MOS vote aggregation.
    mf_mos_dynamic_arr = None
    if (
        mf_mos_votes_arr is not None
        and n_sweep_hits_arr is not None
        and cfg.mf_mos.enabled
        and cfg.mf_mos.fusion_mode != "independent"
    ):
        denom = np.maximum(n_sweep_hits_arr, 1)
        vote_fraction = mf_mos_votes_arr / denom
        mf_mos_mask = (mf_mos_votes_arr >= cfg.min_mf_mos_votes) & (
            vote_fraction >= cfg.mf_mos_vote_fraction_threshold
        )
        mf_mos_dynamic_arr = np.sort(unique_keys[mf_mos_mask])

    diag = {
        "n_evidenced": int(evidenced.sum()),
        "n_under_evidenced_with_hits": int(under_evidenced_with_hits_mask.sum()),
        "n_free_only": int(free_only_mask.sum()),
    }
    return static_arr, not_dynamic_arr, mf_mos_dynamic_arr, diag
