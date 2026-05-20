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

from wato_common.artifact_store import mapmos_logit_path
from wato_lidar_preprocessing.config import ComponentConfig
from wato_lidar_preprocessing.mapmos.io import read_logits
from wato_lidar_preprocessing.ray_traversal import (
    extract_log_odds_arrays,
    extract_max_logit_array,
    make_log_odds_dicts,
    make_max_logit_dict,
    update_sweep_log_odds,
)
from wato_lidar_preprocessing.voxel import voxel_indices

from .io_helpers import load_world_full, sigmoid

log = logging.getLogger(__name__)


def _load_and_validate_logits(
    bag_id: str,
    chunk_id: str,
    sweep_id: int,
    n_expected: int,
    cfg: ComponentConfig,
) -> np.ndarray | None:
    """Read a per-sweep MapMOS logit sidecar, validate, clamp.

    Returns None when:
      - cfg.mapmos.enabled is False (no MapMOS contribution),
      - the file is missing (classify falls through to geometry-only),
      - the file length disagrees with the xyz NPZ length (corrupt; warn and
        fall through — plan non-negotiables #2 and #14).

    Plan non-negotiable #20: branch on `read_logits is None`, NOT on
    `len(...) == 0`. A zero-length sidecar for a zero-point sweep is
    legitimate and aligned.

    Plan non-negotiable #5: clamp at the boundary (warn on overshoot)
    BEFORE the values reach the kernel — keeps a model-blowup logit of
    1e6 from injecting massive bias into the log-odds grid.
    """
    if not cfg.mapmos.enabled:
        return None
    uri = mapmos_logit_path(bag_id, chunk_id, sweep_id)
    logits = read_logits(uri)
    if logits is None:
        log.debug(
            "chunk %s sweep %s: no MapMOS sidecar; geometry-only fusion",
            chunk_id,
            sweep_id,
        )
        return None
    if logits.shape[0] != n_expected:
        log.warning(
            "chunk %s sweep %s: MapMOS logit length %d != xyz length %d — "
            "ignoring sidecar for this sweep",
            chunk_id,
            sweep_id,
            logits.shape[0],
            n_expected,
        )
        return None
    clamp = float(cfg.mapmos.fusion.logit_clamp)
    if logits.size:
        raw_max = float(np.abs(logits).max())
        if raw_max > clamp:
            log.warning(
                "chunk %s sweep %s: max |logit|=%.2f exceeded clamp %.2f",
                chunk_id,
                sweep_id,
                raw_max,
                clamp,
            )
        logits = np.clip(logits, -clamp, clamp).astype(np.float32, copy=False)
    else:
        logits = logits.astype(np.float32, copy=False)
    return logits


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
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """Pass 1 for the log-odds path.

    Loads each sweep once, populates the typed-dict log_odds accumulators
    via ray_traversal, builds sweep_keys for Pass 2, and (optionally) caches
    xyz/intensity/ground_mask in memory so Pass 2 doesn't re-read the NPZs.

    When cfg.mapmos.enabled is True AND a per-sweep sidecar logit file
    exists at mapmos_logit_path(bag_id, chunk_id, sweep_id), the logits
    are length-validated against the world NPZ, clamped to ±logit_clamp,
    and folded into the kernel's endpoint update as a per-point prior.
    The kernel also tracks raw per-voxel max logits for the under-evidenced
    rescue path (plan step 4).

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
                              max_logit_vals) sorted by key. max_logit_vals
                              is aligned with unique_keys (plan #19); each
                              entry is the RAW (not alpha-scaled) max
                              MapMOS prior seen at that voxel, or -inf if
                              no MapMOS prior touched it.
    """
    log_odds_dict, n_obs_dict, n_hits_dict = make_log_odds_dicts()
    # Always allocate the max_logit dict — the kernel only writes when
    # priors are present. Cost is one empty Numba typed dict per chunk.
    max_logit_dict = make_max_logit_dict() if cfg.mapmos.enabled else None

    use_mapmos = cfg.mapmos.enabled and (bag_id is not None)
    if cfg.mapmos.enabled and bag_id is None:
        log.warning(
            "chunk %s: cfg.mapmos.enabled=True but build_log_odds_grid was "
            "called without bag_id; MapMOS priors will be ignored",
            chunk_id,
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

        if sweep_origin is None:
            # Hard-fail rather than skip: the orphan sweep's keys are already
            # in `sweep_keys` (line above) but its evidence never reaches the
            # log-odds grid. Pass 2's searchsorted then misses every one of
            # those keys -> mask=True for every point in the sweep -> the
            # entire sweep silently becomes phantom dynamic in
            # dynamic_map.npz. Producing a corrupt chunk is worse than
            # producing none for an offline annotation pipeline; the deskew
            # that wrote this NPZ owes us an origin.
            raise ValueError(
                f"chunk {chunk_id!r} sweep {row.get('sweep_id')!r}: world NPZ at "
                f"{row.get('world_path')!r} is missing the 'origin' field. "
                "Ray traversal cannot run without it, and skipping the sweep "
                "would silently mislabel all its points as dynamic. Re-run "
                "deskew on this chunk (lidar_preprocessing --force) so the "
                "origin gets written, or set classification_method=persistence "
                "to bypass the log-odds path entirely."
            )

        # Read MapMOS sidecar (None when disabled / missing / corrupt).
        # Done BEFORE the endpoint-strategy branch because the priors
        # array is indexed by point and needs to follow the same
        # filtering the endpoint array gets.
        sweep_logits = None
        if use_mapmos:
            sweep_logits = _load_and_validate_logits(
                bag_id, chunk_id, int(row["sweep_id"]), xyz.shape[0], cfg
            )

        if cfg.ground_endpoint_strategy == "skip_endpoint":
            # Traverse all rays; skip +l_occ at ground endpoints so air
            # voxels above the road accumulate free-space evidence while
            # road-surface voxels keep n_hits==0.
            endpoints_arr = xyz
            is_ground_arr = ground_mask  # None → all-False inside traversal
            endpoint_priors = sweep_logits  # same length as endpoints_arr
        else:  # "skip_ray"
            # Legacy: skip ground rays entirely. Bug fix #4 handles the
            # downstream consequence (ground voxels never enter log_odds)
            # via a post-filter in Pass 2 — sweep_keys stays full-length.
            if ground_mask is not None:
                endpoints_arr = xyz[~ground_mask]
                endpoint_priors = (
                    sweep_logits[~ground_mask] if sweep_logits is not None else None
                )
            else:
                endpoints_arr = xyz
                endpoint_priors = sweep_logits
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
                endpoint_priors=endpoint_priors,
                alpha=cfg.mapmos.fusion.alpha,
                max_logit=max_logit_dict,
            )

    unique_keys, lo_vals, n_obs_vals, n_hits_vals = extract_log_odds_arrays(
        log_odds_dict, n_obs_dict, n_hits_dict
    )
    if max_logit_dict is not None:
        max_logit_vals = extract_max_logit_array(unique_keys, max_logit_dict)
    else:
        max_logit_vals = np.full(unique_keys.shape[0], np.float32(-1e18), dtype=np.float32)

    return (
        xyz_cache,
        intensity_cache,
        ground_mask_cache,
        sweep_keys,
        frame_keys,
        (unique_keys, lo_vals, n_obs_vals, n_hits_vals, max_logit_vals),
    )


def classify_from_log_odds(
    unique_keys: np.ndarray,
    lo_vals: np.ndarray,
    n_obs_vals: np.ndarray,
    n_hits_vals: np.ndarray,
    cfg: ComponentConfig,
    max_logit_vals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Return (static_arr, not_dynamic_arr, diag).

    static_arr           : voxels confident enough to label static (in the static cloud).
    not_dynamic_arr      : sorted union of (static + free-only + under-evidenced-with-hits
                           + ambiguous); Pass 2 uses this for the dynamic mask. Points in
                           any of these voxels get mask=False.

    Three-band semantics on evidenced + hit voxels:
      p_occ >= p_static_threshold              → confident static
      p_dynamic_threshold < p_occ < p_static_threshold → ambiguous (NOT dynamic)
      p_occ <= p_dynamic_threshold             → confident dynamic

    For an offline auto-label pipeline a false-positive dynamic = phantom
    moving-object proposal that has to be tracked, refined, then thrown out
    downstream. The ambiguous bucket exists so we only emit dynamic labels
    when the geometric evidence (or MapMOS rescue, below) is decisive.

    Plan step 4 — under-evidenced rescue: when cfg.mapmos.fusion.apply_to_under_evidenced
    AND a voxel's RAW max logit crosses cfg.mapmos.fusion.under_evidenced_logit_threshold,
    the voxel drops OUT of not_dynamic_arr (its points get labeled dynamic).
    The comparison is against RAW logits — independent of `alpha`, so
    re-tuning fusion never moves the rescue boundary. Rescue only applies
    to the under-evidenced bucket; the ambiguous bucket is the geometric
    grid's own uncertainty signal and is left untouched.
    """
    if unique_keys.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, {
            "n_evidenced": 0,
            "n_under_evidenced_with_hits": 0,
            "n_free_only": 0,
            "n_under_evidenced_rescued": 0,
            "n_ambiguous": 0,
            "n_confident_dynamic": 0,
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

    # Ambiguous middle band — evidenced + hit but p_occ sits between the
    # dynamic and static thresholds. Routes into not_dynamic_arr (see
    # docstring); confident-dynamic stays as `~not_dynamic_arr` in Pass 2.
    ambiguous_mask = (
        evidenced
        & has_hits
        & (p_occ > cfg.p_dynamic_threshold)
        & (p_occ < cfg.p_static_threshold)
    )
    ambiguous_arr = unique_keys[ambiguous_mask]

    # MapMOS under-evidenced rescue (plan step 4 / non-negotiables #4, #18).
    # max_logit_vals is aligned with unique_keys; rescue fires when the
    # voxel's RAW max prior crosses the threshold. Apply BEFORE deriving
    # under_arr so rescued voxels are not added to not_dynamic_arr.
    rescue_mask = np.zeros_like(under_evidenced_with_hits_mask)
    if (
        cfg.mapmos.enabled
        and cfg.mapmos.fusion.apply_to_under_evidenced
        and max_logit_vals is not None
        and max_logit_vals.shape[0] == unique_keys.shape[0]
    ):
        threshold = np.float32(cfg.mapmos.fusion.under_evidenced_logit_threshold)
        rescue_mask = under_evidenced_with_hits_mask & (max_logit_vals >= threshold)
    under_arr = unique_keys[under_evidenced_with_hits_mask & ~rescue_mask]

    # not_dynamic_arr = static + free_only + under-with-hits-not-rescued + ambiguous.
    parts = [
        a for a in (static_arr, free_only_arr, under_arr, ambiguous_arr) if a.size > 0
    ]
    if not parts:
        not_dynamic_arr = np.empty(0, dtype=np.int64)
    elif len(parts) == 1:
        not_dynamic_arr = parts[0]
    else:
        # np.unique sorts and dedupes — exactly what searchsorted needs.
        not_dynamic_arr = np.unique(np.concatenate(parts))

    # Diagnostic-only: voxels that actually came out the bottom as dynamic.
    confident_dynamic_mask = (
        evidenced & has_hits & (p_occ <= cfg.p_dynamic_threshold)
    )

    diag = {
        "n_evidenced": int(evidenced.sum()),
        "n_under_evidenced_with_hits": int(under_evidenced_with_hits_mask.sum()),
        "n_free_only": int(free_only_mask.sum()),
        "n_under_evidenced_rescued": int(rescue_mask.sum()),
        "n_ambiguous": int(ambiguous_mask.sum()),
        "n_confident_dynamic": int(confident_dynamic_mask.sum()),
    }
    return static_arr, not_dynamic_arr, diag
