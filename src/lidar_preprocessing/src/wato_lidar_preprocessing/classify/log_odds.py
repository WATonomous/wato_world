"""Log-odds ray-casting classifier.

build_log_odds_grid:    Pass 1 — loops sweeps through the ray_traversal
                        kernel, returns sorted per-voxel arrays.
classify_from_log_odds: applies thresholds to return static_arr,
                        not_dynamic_arr, mf_mos_dynamic_arr, classification,
                        and diagnostic counts.
"""

from __future__ import annotations

import logging

import numpy as np
from tqdm import tqdm

from wato_lidar_preprocessing.config import ComponentConfig
from wato_lidar_preprocessing.ray_traversal import (
    apply_global_map_boost,
    extract_log_odds_arrays,
    make_log_odds_dicts,
    update_sweep_log_odds,
)
from wato_lidar_preprocessing.voxel import voxel_indices

from .global_map_prior import GlobalMapPrior
from .io_helpers import load_mf_mos_world_mask, load_world_full, sigmoid

log = logging.getLogger(__name__)


# Classification codes written to voxel_diag.npz.
# viz.py's _CLASS_COLORS keys must stay aligned with these values.
CLASS_STATIC = 0
CLASS_AMBIGUOUS = 1          # evidenced + has_hits + p_dynamic ≤ p_occ < p_static
CLASS_UNDER_EVIDENCED = 2    # has_hits but n_obs < min_observations
CLASS_FREE_ONLY = 3          # n_hits < min_occupied_hits
CLASS_DYNAMIC = 4            # evidenced + has_hits + p_occ < p_dynamic_threshold


def build_log_odds_grid(
    meta_rows: list[dict],
    cfg: ComponentConfig,
    origin: np.ndarray,
    chunk_id: str,
    *,
    cache_xyz: bool,
    cache_intensity: bool = True,
    bag_id: str | None = None,
    global_map_prior: GlobalMapPrior | None = None,
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

    Args:
        global_map_prior: optional GlobalMapPrior (two-pass mode). Endpoints
            within match_radius_m of the global map get a credibility-weighted
            log-odds boost (UniLiPs IWU Eq. 2-3). Touches log_odds only —
            n_hits stays backed by real sweep returns so has_hits is not
            bypassed.

    Returns:
        xyz_cache, intensity_cache, ground_mask_cache: per-sweep caches
            (entries are None when the sweep was invalid or caching is off).
        sweep_keys: per-sweep int64 voxel-key arrays, always full length so
            dynamic_mask.npy stays length-aligned with the world NPZ.
        frame_keys: frame_id → list of key arrays for per-frame voxel
            occupancy export.
        log_odds_arrays: (unique_keys, lo_vals, n_obs_vals, n_hits_vals,
            mf_mos_votes_arr, n_sweep_hits_arr) sorted by key.
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
            ground_mask_cache.append(ground_mask)
            sweep_keys.append(np.empty(0, dtype=np.int64))
            continue

        # Fail fast before mutating any accumulators or caches for this sweep.
        if sweep_origin is None:
            raise ValueError(
                f"chunk {chunk_id} sweep {row.get('sweep_id')}: world NPZ "
                f"missing 'origin' field — re-run deskew to regenerate."
            )

        keys = voxel_indices(xyz, origin, cfg.voxel_size_m, chunk_id=chunk_id)
        sweep_keys.append(keys)
        xyz_cache.append(xyz if cache_xyz else None)
        intensity_cache.append(intensity if cache_xyz and cache_intensity else None)
        # ground_mask is always cached — masking.py needs it regardless of
        # cache_xyz, and it's 1 bit per point (cheap even on big chunks).
        ground_mask_cache.append(ground_mask)

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

        if cfg.ground_endpoint_strategy == "skip_endpoint":
            # Traverse all rays; the kernel skips +l_occ at ground endpoints
            # so road-surface voxels keep n_hits==0 while air voxels above
            # the road still accumulate free-space evidence.
            endpoints_arr = xyz
            is_ground_arr = ground_mask
        else:  # "skip_ray" — drop ground rays entirely.
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
                r_max=cfg.r_max_credibility_m,
                use_range_weight=cfg.use_range_weighted_log_odds,
            )

        # UniLiPs IWU boost. Applied AFTER update_sweep_log_odds so it adds
        # to whatever ray traversal already wrote. Ground points are excluded:
        # the global static map encodes above-ground structure, and boosting
        # untraversed ground voxels creates phantom entries with n_obs=0.
        if global_map_prior is not None and xyz.shape[0] > 0:
            if ground_mask is not None:
                boost_select = ~ground_mask
                boost_xyz = xyz[boost_select]
                boost_keys = keys[boost_select]
            else:
                boost_xyz = xyz
                boost_keys = keys
            if boost_xyz.shape[0] > 0:
                map_hit, r_star = global_map_prior.query_sweep(boost_xyz, sweep_origin)
                if map_hit.any():
                    apply_global_map_boost(
                        boost_keys[map_hit],
                        r_star[map_hit],
                        cfg.l_occ_global_map,
                        cfg.log_odds_clamp,
                        log_odds_dict,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, dict[str, int]]:
    """Return (static_arr, not_dynamic_arr, mf_mos_dynamic_arr, classification, diag).

    static_arr:         voxels confident enough to label static.
    not_dynamic_arr:    sorted union of (static + free_only + under-with-hits
                        + ambiguous). Points in any of these voxels get
                        mask=False in Pass 2.
    mf_mos_dynamic_arr: sorted voxel keys where MF-MOS votes exceed thresholds,
                        or None when vote aggregation was not active.
    classification:     int8 array per unique_keys with CLASS_* codes —
                        produced here so write_chunk_voxel_diagnostics doesn't
                        re-derive the bucketing and silently diverge.
    """
    if unique_keys.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, None, np.empty(0, dtype=np.int8), {
            "n_evidenced": 0,
            "n_under_evidenced_with_hits": 0,
            "n_ambiguous": 0,
            "n_free_only": 0,
        }

    p_occ = sigmoid(lo_vals)
    evidenced = n_obs_vals >= cfg.min_observations
    has_hits = n_hits_vals >= cfg.min_occupied_hits

    static_mask = evidenced & has_hits & (p_occ >= cfg.p_static_threshold)
    static_arr = unique_keys[static_mask]

    free_only_mask = n_hits_vals < cfg.min_occupied_hits
    free_only_arr = unique_keys[free_only_mask]

    under_evidenced_with_hits_mask = (~evidenced) & has_hits
    under_arr = unique_keys[under_evidenced_with_hits_mask]

    ambiguous_mask = (
        evidenced
        & has_hits
        & (p_occ < cfg.p_static_threshold)
        & (p_occ >= cfg.p_dynamic_threshold)
    )
    ambiguous_arr = unique_keys[ambiguous_mask]

    parts = [a for a in (static_arr, free_only_arr, under_arr, ambiguous_arr) if a.size > 0]
    if not parts:
        not_dynamic_arr = np.empty(0, dtype=np.int64)
    elif len(parts) == 1:
        not_dynamic_arr = parts[0]
    else:
        not_dynamic_arr = np.unique(np.concatenate(parts))

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
        log.info(
            "mf_mos vote aggregation: %d voxels with votes, %d pass thresholds "
            "(min_votes=%d, fraction>=%.2f) → %d mf_mos-dynamic voxels",
            int((mf_mos_votes_arr > 0).sum()),
            int(mf_mos_mask.sum()),
            cfg.min_mf_mos_votes,
            cfg.mf_mos_vote_fraction_threshold,
            mf_mos_dynamic_arr.size,
        )

    # CLASS_FREE_ONLY is the default; predicates below are mutually
    # exclusive partitions of (evidenced, has_hits, p_occ) space.
    dynamic_mask = evidenced & has_hits & (p_occ < cfg.p_dynamic_threshold)
    classification = np.full(unique_keys.shape[0], CLASS_FREE_ONLY, dtype=np.int8)
    classification[under_evidenced_with_hits_mask] = CLASS_UNDER_EVIDENCED
    classification[dynamic_mask] = CLASS_DYNAMIC
    classification[ambiguous_mask] = CLASS_AMBIGUOUS
    classification[static_mask] = CLASS_STATIC

    diag = {
        "n_evidenced": int(evidenced.sum()),
        "n_under_evidenced_with_hits": int(under_evidenced_with_hits_mask.sum()),
        "n_ambiguous": int(ambiguous_mask.sum()),
        "n_free_only": int(free_only_mask.sum()),
    }
    return static_arr, not_dynamic_arr, mf_mos_dynamic_arr, classification, diag
