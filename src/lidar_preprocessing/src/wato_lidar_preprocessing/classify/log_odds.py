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
    apply_global_map_boost,
    extract_log_odds_arrays,
    make_log_odds_dicts,
    update_sweep_log_odds,
)
from wato_lidar_preprocessing.voxel import voxel_indices

from .global_map_prior import GlobalMapPrior
from .io_helpers import load_mf_mos_world_mask, load_world_full, sigmoid

log = logging.getLogger(__name__)


# Classification label codes used in voxel_diag.npz.  Defined here so the
# bucketing logic and the codes that name those buckets live together —
# previously the writer recomputed the buckets from log_odds/n_obs/n_hits
# using copy-pasted thresholds, which silently went out of sync whenever the
# classifier changed.  viz.py's _CLASS_COLORS keys must stay aligned with
# these values.
CLASS_STATIC = 0
CLASS_AMBIGUOUS = 1          # evidenced + has_hits + p_dynamic ≤ p_occ < p_static
CLASS_UNDER_EVIDENCED = 2    # has_hits but n_obs < min_observations
CLASS_FREE_ONLY = 3          # n_hits < min_occupied_hits (only ever traversed)
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
        global_map_prior: optional GlobalMapPrior built from a bag-level
            global_static_map.npz.  When provided (two-pass mode), each sweep's
            endpoints get a credibility-weighted log-odds boost where they fall
            within match_radius_m of the global map (UniLiPs IWU Eq. 2-3).
            The boost touches only log_odds — n_hits stays backed by real
            sweep returns so the has_hits gate is not bypassed.

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
            ground_mask_cache.append(ground_mask)
            sweep_keys.append(np.empty(0, dtype=np.int64))
            continue

        # sweep_keys stays full length (matches xyz from world NPZ).  The
        # ground-aware filtering for ray traversal happens INSIDE the call
        # to update_sweep_log_odds via is_ground / endpoints filtering.
        keys = voxel_indices(xyz, origin, cfg.voxel_size_m, chunk_id=chunk_id)
        sweep_keys.append(keys)
        xyz_cache.append(xyz if cache_xyz else None)
        intensity_cache.append(intensity if cache_xyz and cache_intensity else None)
        # Always cache ground_mask — it's 1 bit per point (a few MB even on
        # the biggest chunks).  Dropping it when cache_xyz=False made the
        # ground filter at masking.py silently no-op on large chunks where
        # the cache auto-disabled, leaking ground points into both static
        # and dynamic clouds.
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

        if sweep_origin is None:
            # Artifact corruption — deskew should always write `origin`.
            # Silently skipping leaves the sweep's voxels with no log-odds
            # entries, so the sweep's points fall through to dynamic-by-
            # default in masking.py.  Hard-fail instead.
            raise ValueError(
                f"chunk {chunk_id} sweep {row.get('sweep_id')}: world NPZ "
                f"missing 'origin' field. This shouldn't happen in normal "
                f"flow — re-run deskew on this chunk to regenerate the "
                f"artifact. (Continuing silently was the cause of the "
                f"sweep-pose-fallback bug; we no longer accept this case.)"
            )

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
                r_max=cfg.r_max_credibility_m,
                use_range_weight=cfg.use_range_weighted_log_odds,
            )

        # UniLiPs IWU boost (two-pass mode).  Applied AFTER update_sweep_log_odds
        # so it adds to whatever the normal ray-traversal already wrote.
        #
        # Ground points are excluded from the boost query: in skip_ray mode
        # ground rays are never traversed (no n_obs entry for the endpoint
        # voxel), so a boost on those voxels creates phantom log_odds entries
        # with n_obs=0 / n_hits=0 — they go to free_only (correct) but inflate
        # unique_keys and waste memory in extraction.  In skip_endpoint mode
        # the same can happen for ground endpoint voxels that no other ray
        # traverses.  Filtering also matches the prior's intent: the global
        # static map encodes above-ground structure, not the road surface.
        # The sweep_origin-is-None branch above already hard-raises, so this
        # block is unreachable with a None origin.
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

    static_arr           : voxels confident enough to label static (in the static cloud).
    not_dynamic_arr      : sorted union of (static + free-only + under-evidenced-with-hits);
                           Pass 2 uses this for the dynamic mask. Points in any of these
                           voxels get mask=False.
    mf_mos_dynamic_arr   : sorted voxel keys where MF-MOS votes exceed thresholds,
                           or None when vote aggregation was not active.
    classification       : int8 array per unique_keys with CLASS_* codes — produced
                           here so write_chunk_voxel_diagnostics doesn't re-derive
                           the bucketing from log_odds/n_obs/n_hits (and silently
                           diverge if the thresholds ever change).
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

    # Confident static: passes all three gates.
    static_mask = evidenced & has_hits & (p_occ >= cfg.p_static_threshold)
    static_arr = unique_keys[static_mask]

    # Free-space-only: voxels whose endpoint-hit count is below
    # min_occupied_hits — too few hits to confidently call them occupied,
    # regardless of how negative log_odds got.  At min_occupied_hits=1 this
    # collapses to the historical `n_hits == 0` predicate (pure free-space).
    # At >1 it also catches "sparse-surface" voxels (one stray return on a
    # tree branch / building edge surrounded by many through-rays); without
    # this gate, those voxels fall into a classification hole — has_hits is
    # False so they miss static/under/ambiguous, but free_only_mask used to
    # require n_hits==0 so they missed that too, falling through to dynamic.
    # That made min_occupied_hits actively counterproductive for filtering
    # NuScenes-style false positives on textured statics.
    free_only_mask = n_hits_vals < cfg.min_occupied_hits
    free_only_arr = unique_keys[free_only_mask]

    # BUG FIX #2: voxels with hits but too few observations get the benefit
    # of the doubt — into not_dynamic_arr (mask=False) but NOT into
    # static_arr (don't pollute the static cloud with low-confidence points).
    under_evidenced_with_hits_mask = (~evidenced) & has_hits
    under_arr = unique_keys[under_evidenced_with_hits_mask]

    # Ambiguous: evidenced + has_hits but p_occ is in the band
    # [p_dynamic_threshold, p_static_threshold) — carving evidence and hit
    # evidence are roughly balanced.  Pre-fix this bucket fell through to
    # the implicit dynamic class (p_dynamic_threshold was configured but
    # never consulted), which leaked textured statics — brick walls, tree
    # canopies, sparse foliage — into dynamic_map.npz whenever through-ray
    # beam noise carved their voxels.  Routing to not_dynamic_arr (same
    # treatment as under-evidenced-with-hits) means only voxels with
    # confident carving (p_occ < p_dynamic_threshold AND has hits) end up
    # dynamic.
    ambiguous_mask = (
        evidenced
        & has_hits
        & (p_occ < cfg.p_static_threshold)
        & (p_occ >= cfg.p_dynamic_threshold)
    )
    ambiguous_arr = unique_keys[ambiguous_mask]

    # not_dynamic_arr is the union of (static + free_only + under-with-hits + ambiguous).
    parts = [a for a in (static_arr, free_only_arr, under_arr, ambiguous_arr) if a.size > 0]
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
        log.info(
            "mf_mos vote aggregation: %d voxels with votes, %d pass thresholds "
            "(min_votes=%d, fraction>=%.2f) → %d mf_mos-dynamic voxels",
            int((mf_mos_votes_arr > 0).sum()),
            int(mf_mos_mask.sum()),
            cfg.min_mf_mos_votes,
            cfg.mf_mos_vote_fraction_threshold,
            mf_mos_dynamic_arr.size,
        )

    # Build the int8 classification array once.  Order matters — each later
    # assignment overrides earlier ones where masks overlap, so we go least-
    # to most-specific.  CLASS_FREE_ONLY is the default (catches ~has_hits).
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
