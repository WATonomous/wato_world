"""Log-odds ray-casting classifier.

All log-odds increments, thresholds, range weighting and carve margin come
from the datasheet SensorModel (see sensor_model.py) — no hand-tuned knobs.
"""

from __future__ import annotations

import logging

import numpy as np
from tqdm import tqdm

from wato_lidar_preprocessing.config import ComponentConfig
from wato_lidar_preprocessing.ray_traversal import (
    accumulate_cov,
    apply_global_map_boost,
    compute_normals,
    extract_log_odds_arrays,
    make_cov_dicts,
    make_log_odds_dicts,
    update_sweep_log_odds,
)
from wato_lidar_preprocessing.sensor_model import SensorModel
from wato_lidar_preprocessing.voxel import voxel_indices

from .global_map_prior import GlobalMapPrior
from .io_helpers import load_world_full, sigmoid

log = logging.getLogger(__name__)


# Classification codes written to voxel_diag.npz.
# viz.py's _CLASS_COLORS keys must stay aligned with these values.
CLASS_STATIC = 0
CLASS_AMBIGUOUS = 1  # evidenced + has_hits + p_dynamic ≤ p_occ < p_static
CLASS_UNDER_EVIDENCED = 2  # has_hits but n_obs < min_observations
CLASS_FREE_ONLY = 3  # n_hits < min_occupied_hits
CLASS_DYNAMIC = 4  # evidenced + has_hits + p_occ < p_dynamic_threshold


def build_log_odds_grid(
    meta_rows: list[dict],
    cfg: ComponentConfig,
    origin: np.ndarray,
    chunk_id: str,
    *,
    cache_xyz: bool,
    cache_intensity: bool = True,
    pose_sigma_m: float,
    sensor_model: SensorModel,
    global_map_prior: GlobalMapPrior | None = None,
) -> tuple[
    list[np.ndarray | None],
    list[np.ndarray | None],
    list[np.ndarray | None],
    list[np.ndarray],
    dict[int, list[np.ndarray]],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """Pass 1: sub-pass 1a estimates per-voxel surface normals, sub-pass 1b
    ray-casts the log-odds grid with the incidence gate. Caches xyz/intensity/
    ground/origin/keys per sweep for Pass 2.

    pose_sigma_m: per-bag SLAM pose noise (poses.parquet); sets the carve margin.
    global_map_prior: two-pass mode — map-matched endpoints get a one-time
        credibility-weighted prior shift; touches log_odds only.

    Returns the caches + log_odds_arrays = (unique_keys, lo, n_obs, n_hits).
    """
    log_odds_dict, n_obs_dict, n_hits_dict = make_log_odds_dicts()
    cov_dicts = make_cov_dicts()

    voxel_size = cfg.voxel_size_m
    l_occ = sensor_model.l_occ
    l_free = sensor_model.l_free
    clamp = sensor_model.log_odds_clamp
    d_star = sensor_model.credibility_crossover_m(voxel_size)
    margin_m = sensor_model.carve_margin_m(pose_sigma_m)
    grazing_cos = sensor_model.grazing_cos_threshold(voxel_size)
    max_len = cfg.effective_max_ray_length_m()
    log.info(
        "chunk %s log-odds model (%s): l_occ=%.3f l_free=%.3f clamp=%.3f "
        "d*=%.1fm carve_margin=%.3fm grazing_cos=%.2f (σ_range=%.3f "
        "σ_pose=%.3f) max_ray=%.1fm",
        chunk_id,
        sensor_model.name,
        l_occ,
        l_free,
        clamp,
        d_star,
        margin_m,
        grazing_cos,
        sensor_model.range_sigma_m,
        pose_sigma_m,
        max_len,
    )

    # Two-pass global prior: accumulate per-voxel max credibility across all
    # sweeps, applied ONCE after the carve pass (a prior shift, not gain).
    prior_keys_parts: list[np.ndarray] = []
    prior_cred_parts: list[np.ndarray] = []

    xyz_cache: list[np.ndarray | None] = []
    intensity_cache: list[np.ndarray | None] = []
    ground_mask_cache: list[np.ndarray | None] = []
    origin_cache: list[np.ndarray | None] = []
    sweep_keys: list[np.ndarray] = []
    frame_keys: dict[int, list[np.ndarray]] = {}

    # --- Sub-pass 1a: load sweeps, build caches/keys, accumulate per-voxel
    # point moments (non-ground returns) for surface-normal estimation. ------
    for row in tqdm(
        meta_rows,
        desc=f"classify chunk {chunk_id} pass 1a",
        unit="sweep",
    ):
        if row.get("valid") is False:
            xyz_cache.append(None)
            intensity_cache.append(None)
            ground_mask_cache.append(None)
            origin_cache.append(None)
            sweep_keys.append(np.empty(0, dtype=np.int64))
            continue

        xyz, intensity, sweep_origin, ground_mask = load_world_full(row["world_path"])

        if xyz.shape[0] == 0:
            xyz_cache.append(xyz if cache_xyz else None)
            intensity_cache.append(intensity if cache_xyz and cache_intensity else None)
            ground_mask_cache.append(ground_mask)
            origin_cache.append(sweep_origin)
            sweep_keys.append(np.empty(0, dtype=np.int64))
            continue

        # Fail fast before mutating any accumulators or caches for this sweep.
        if sweep_origin is None:
            raise ValueError(
                f"chunk {chunk_id} sweep {row.get('sweep_id')}: world NPZ "
                f"missing 'origin' field — re-run deskew to regenerate."
            )

        keys = voxel_indices(xyz, origin, voxel_size, chunk_id=chunk_id)
        sweep_keys.append(keys)
        xyz_cache.append(xyz if cache_xyz else None)
        intensity_cache.append(intensity if cache_xyz and cache_intensity else None)
        # ground_mask + origin always cached — cheap, and Pass 2 / sub-pass 1b
        # need them regardless of cache_xyz.
        ground_mask_cache.append(ground_mask)
        origin_cache.append(sweep_origin)

        fid = row.get("frame_id")
        if fid is not None:
            frame_keys.setdefault(int(fid), []).append(keys)

        # Normals from non-ground returns (kernel skips ground internally).
        accumulate_cov(xyz, ground_mask, origin, voxel_size, cov_dicts)

    # Estimate one surface normal per planar, sufficiently-observed voxel.
    normal_dicts = compute_normals(cov_dicts, min_pts=cfg.min_observations)

    # --- Sub-pass 1b: ray traversal with the incidence gate + global prior. --
    for i, row in enumerate(
        tqdm(meta_rows, desc=f"classify chunk {chunk_id} pass 1b", unit="sweep")
    ):
        if row.get("valid") is False:
            continue
        keys = sweep_keys[i]
        if keys.shape[0] == 0:
            continue
        sweep_origin = origin_cache[i]
        ground_mask = ground_mask_cache[i]
        if xyz_cache[i] is not None:
            xyz = xyz_cache[i]
        else:
            xyz, _, _, _ = load_world_full(row["world_path"])

        if cfg.ground_endpoint_strategy == "skip_endpoint":
            # Kernel skips +l_occ at ground endpoints but still carves the ray.
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
                voxel_size,
                margin_m,
                max_len,
                log_odds_dict,
                n_obs_dict,
                n_hits_dict,
                l_occ,
                l_free,
                clamp,
                d_star,
                normal_dicts=normal_dicts,
                grazing_cos=grazing_cos,
            )

        # Collect global-map matches for the one-time prior (non-ground only).
        if global_map_prior is not None:
            if ground_mask is not None:
                boost_select = ~ground_mask
                boost_xyz = xyz[boost_select]
                boost_keys = keys[boost_select]
            else:
                boost_xyz = xyz
                boost_keys = keys
            if boost_xyz.shape[0] > 0:
                map_hit, ranges = global_map_prior.query_sweep(boost_xyz, sweep_origin)
                if map_hit.any():
                    cred = sensor_model.range_weight(ranges[map_hit], voxel_size)
                    prior_keys_parts.append(boost_keys[map_hit])
                    prior_cred_parts.append(cred.astype(np.float32))

    # One-time global-map prior shift over the chunk-wide matched voxel set.
    if prior_keys_parts:
        apply_global_map_boost(
            np.concatenate(prior_keys_parts),
            np.concatenate(prior_cred_parts),
            sensor_model.l_map_prior,
            clamp,
            log_odds_dict,
        )

    unique_keys, lo_vals, n_obs_vals, n_hits_vals = extract_log_odds_arrays(
        log_odds_dict, n_obs_dict, n_hits_dict
    )
    return (
        xyz_cache,
        intensity_cache,
        ground_mask_cache,
        sweep_keys,
        frame_keys,
        (unique_keys, lo_vals, n_obs_vals, n_hits_vals),
    )


def classify_from_log_odds(
    unique_keys: np.ndarray,
    lo_vals: np.ndarray,
    n_obs_vals: np.ndarray,
    n_hits_vals: np.ndarray,
    cfg: ComponentConfig,
    sensor_model: SensorModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Return (static_arr, not_dynamic_arr, classification, diag).

    static if p_occ ≥ p_static (= p_hit), dynamic if p_occ < p_dynamic
    (= 1-p_hit); the band between is AMBIGUOUS → not-dynamic (conservative).
    not_dynamic_arr is the union of (static + free_only + under-with-hits +
    ambiguous); points in any of those get mask=False.
    """
    if unique_keys.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return (
            empty,
            empty,
            np.empty(0, dtype=np.int8),
            {
                "n_evidenced": 0,
                "n_under_evidenced_with_hits": 0,
                "n_ambiguous": 0,
                "n_free_only": 0,
            },
        )

    p_static_threshold = sensor_model.p_static_threshold
    p_dynamic_threshold = sensor_model.p_dynamic_threshold

    p_occ = sigmoid(lo_vals)
    evidenced = n_obs_vals >= cfg.min_observations
    has_hits = n_hits_vals >= cfg.min_occupied_hits

    static_mask = evidenced & has_hits & (p_occ >= p_static_threshold)
    static_arr = unique_keys[static_mask]

    free_only_mask = n_hits_vals < cfg.min_occupied_hits
    free_only_arr = unique_keys[free_only_mask]

    under_evidenced_with_hits_mask = (~evidenced) & has_hits
    under_arr = unique_keys[under_evidenced_with_hits_mask]

    ambiguous_mask = (
        evidenced
        & has_hits
        & (p_occ < p_static_threshold)
        & (p_occ >= p_dynamic_threshold)
    )
    ambiguous_arr = unique_keys[ambiguous_mask]

    parts = [
        a for a in (static_arr, free_only_arr, under_arr, ambiguous_arr) if a.size > 0
    ]
    if not parts:
        not_dynamic_arr = np.empty(0, dtype=np.int64)
    elif len(parts) == 1:
        not_dynamic_arr = parts[0]
    else:
        not_dynamic_arr = np.unique(np.concatenate(parts))

    # CLASS_FREE_ONLY is the default; predicates below are mutually
    # exclusive partitions of (evidenced, has_hits, p_occ) space.
    dynamic_mask = evidenced & has_hits & (p_occ < p_dynamic_threshold)
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
    return static_arr, not_dynamic_arr, classification, diag
