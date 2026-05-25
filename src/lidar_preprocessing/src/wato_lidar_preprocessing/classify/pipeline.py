"""Step B — Voxel-based static/dynamic classification (orchestration).

Two-pass algorithm:
  Pass 1: For every sweep, discretize world-frame points to voxel keys and
          (log_odds mode only) accumulate occupancy log-odds via ray traversal.
  Pass 2: For every sweep, apply the static/dynamic label as a boolean mask
          and write {sweep_id:06d}_dynamic_mask.npy.

The static cloud is written to static_map.npz and the dynamic cloud to
dynamic_map.npz; downstream consumers depend on the dynamic_mask.npy length
matching the world NPZ point count, so Pass 2 never filters keys.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from wato_common.artifact_store import (
    dynamic_map_path,
    ensure_local_dir,
    lidar_proc_dir,
    lidar_proc_index_path,
    local_path,
    static_map_path,
)
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import PROCESSED_SWEEPS_SCHEMA, ProcessedSweepMeta
from wato_lidar_preprocessing.config import ComponentConfig
from wato_lidar_preprocessing.voxel import voxel_indices

from .io_helpers import (
    cache_byte_budget,
    estimate_cache_bytes,
    load_mf_mos_world_mask,
    load_world_full,
    load_world_xyz_intensity,
    origin_from_index,
)
from .log_odds import build_log_odds_grid, classify_from_log_odds
from .masking import apply_classification_to_sweep
from .occupancy_export import (
    write_chunk_voxel_diagnostics,
    write_chunk_voxel_occupancy,
    write_per_frame_voxel_occupancy,
)
from .persistence import classify_persistence

log = logging.getLogger(__name__)


@dataclass
class ClassifyResult:
    n_static: int
    n_dynamic: int
    static_map_path: str
    cache_auto_disabled: bool = False
    estimated_cache_bytes: int = 0


def _write_empty_outputs(
    bag_id: str,
    chunk_id: str,
    voxel_size: float,
) -> ClassifyResult:
    """Write sentinel static_map.npz + dynamic_map.npz when nothing to classify."""
    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    out_uri = static_map_path(bag_id, chunk_id)
    np.savez_compressed(
        local_path(out_uri),
        xyz=np.empty((0, 3), dtype=np.float64),
        voxel_size=np.float32(voxel_size),
        origin=np.zeros(3, dtype=np.float64),
        static_voxel_keys=np.empty(0, dtype=np.int64),
    )
    np.savez_compressed(
        local_path(dynamic_map_path(bag_id, chunk_id)),
        xyz=np.empty((0, 3), dtype=np.float64),
        sweep_id=np.empty(0, dtype=np.int32),
    )
    return ClassifyResult(0, 0, out_uri)



def _run_pass_1_persistence(
    cfg: ComponentConfig,
    meta_rows: list[dict],
    origin: np.ndarray,
    chunk_id: str,
    *,
    cache_xyz: bool,
) -> tuple[
    list[np.ndarray | None],
    list[np.ndarray | None],
    list[np.ndarray | None],
    list[np.ndarray],
    dict[int, list[np.ndarray]],
]:
    """Pass 1 for the persistence path (no log-odds accumulators).

    Returns the same 5-tuple shape as build_log_odds_grid (minus the log-odds
    arrays) so pipeline.process_chunk treats the two paths uniformly.
    ground_mask_cache is populated even when cache_xyz is False — masking.py
    needs it to keep ground points out of dynamic_map.npz, and it's only
    1 bit per point so size is a non-issue.
    """
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

        # load_world_full also returns sensor_origin (unused in persistence)
        # so we discard it.  Switching from load_world_xyz_intensity is the
        # only way to get ground_mask without a second NPZ read.
        xyz, intensity, _origin, ground_mask = load_world_full(row["world_path"])
        if xyz.shape[0] == 0:
            xyz_cache.append(xyz if cache_xyz else None)
            intensity_cache.append(intensity if cache_xyz else None)
            ground_mask_cache.append(ground_mask)
            sweep_keys.append(np.empty(0, dtype=np.int64))
            continue

        keys = voxel_indices(xyz, origin, cfg.voxel_size_m, chunk_id=chunk_id)
        sweep_keys.append(keys)
        xyz_cache.append(xyz if cache_xyz else None)
        intensity_cache.append(intensity if cache_xyz else None)
        ground_mask_cache.append(ground_mask)
        fid = row.get("frame_id")
        if fid is not None:
            frame_keys.setdefault(int(fid), []).append(keys)

    return xyz_cache, intensity_cache, ground_mask_cache, sweep_keys, frame_keys


def process_chunk(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
) -> ClassifyResult:
    """Classify all world-frame sweeps in a chunk as static or dynamic."""
    meta_rows = read_rows(lidar_proc_index_path(bag_id, chunk_id))
    if not meta_rows:
        log.warning(
            "chunk %s: no processed sweeps in lidar_proc_index — writing empty static map",
            chunk_id,
        )
        return _write_empty_outputs(bag_id, chunk_id, cfg.voxel_size_m)

    any_intensity = any(bool(r.get("has_intensity", False)) for r in meta_rows)

    # Auto-disable the in-memory cache when the chunk is too big.  The config
    # flag stays as a force-on override; users who set it explicitly opt out
    # of the safety net.
    cache_budget = cache_byte_budget()
    estimated_bytes = estimate_cache_bytes(meta_rows)
    cache_auto_disabled = False
    if cfg.cache_world_xyz_in_memory and estimated_bytes > cache_budget:
        log.warning(
            "chunk %s: estimated cache size %.2f GB exceeds %.2f GB cap — disabling in-memory cache",
            chunk_id,
            estimated_bytes / 1e9,
            cache_budget / 1e9,
        )
        cache_xyz = False
        cache_auto_disabled = True
    else:
        cache_xyz = bool(cfg.cache_world_xyz_in_memory)

    origin = origin_from_index(meta_rows)
    if origin is None:
        return _write_empty_outputs(bag_id, chunk_id, cfg.voxel_size_m)

    use_log_odds = cfg.classification_method == "log_odds"
    threshold = None
    diag: dict[str, int] = {}

    if use_log_odds:
        (
            xyz_cache,
            intensity_cache,
            ground_mask_cache,
            sweep_keys,
            frame_keys,
            (unique_keys, lo_vals, n_obs_vals, n_hits_vals, mf_mos_votes_arr, n_sweep_hits_arr),
        ) = build_log_odds_grid(
            meta_rows, cfg, origin, chunk_id, cache_xyz=cache_xyz, bag_id=bag_id
        )
        static_arr, not_dynamic_arr, mf_mos_dynamic_arr, diag = classify_from_log_odds(
            unique_keys, lo_vals, n_obs_vals, n_hits_vals, cfg,
            mf_mos_votes_arr=mf_mos_votes_arr,
            n_sweep_hits_arr=n_sweep_hits_arr,
        )
    else:
        (
            xyz_cache,
            intensity_cache,
            ground_mask_cache,
            sweep_keys,
            frame_keys,
        ) = _run_pass_1_persistence(
            cfg, meta_rows, origin, chunk_id, cache_xyz=cache_xyz
        )
        static_arr, not_dynamic_arr, threshold = classify_persistence(
            sweep_keys, len(meta_rows), cfg
        )
        mf_mos_dynamic_arr = None
        unique_keys = np.empty(0, dtype=np.int64)
        lo_vals = np.empty(0, dtype=np.float32)
        n_obs_vals = np.empty(0, dtype=np.int32)
        n_hits_vals = np.empty(0, dtype=np.int32)

    # Pass 2: per-sweep masks + accumulate static/dynamic clouds.
    static_xyz_chunks: list[np.ndarray] = []
    static_intensity_chunks: list[np.ndarray] = []
    dyn_xyz_chunks: list[np.ndarray] = []
    dyn_intensity_chunks: list[np.ndarray] = []
    dyn_sweep_id_chunks: list[np.ndarray] = []
    total_static = 0
    total_dynamic = 0
    updated_meta: list[dict] = []

    for i, (row, keys) in enumerate(
        tqdm(
            zip(meta_rows, sweep_keys),
            total=len(meta_rows),
            desc=f"classify chunk {chunk_id} pass 2",
            unit="sweep",
        )
    ):
        sweep_id = int(row["sweep_id"])
        if row.get("valid") is False:
            updated_meta.append(_invalid_meta_row(row, sweep_id))
            continue

        # log_odds: mf_mos_dynamic_arr is pre-computed chunk-wide from voxel votes.
        # persistence: load per-sweep mask and build a sweep-local dynamic array —
        # same semantics as the old per-point OR, but through the shared searchsorted path.
        sweep_mf_mos_dynamic_arr = mf_mos_dynamic_arr
        if (
            not use_log_odds
            and cfg.mf_mos.enabled
            and cfg.mf_mos.fusion_mode != "independent"
            and keys.shape[0] > 0
        ):
            mf_mask = load_mf_mos_world_mask(
                bag_id, chunk_id, row, keys.shape[0], cfg.filter_nonfinite_points
            )
            if mf_mask is not None:
                sweep_mf_mos_dynamic_arr = np.sort(np.unique(keys[mf_mask]))

        result = apply_classification_to_sweep(
            row,
            sweep_id,
            keys,
            static_arr,
            not_dynamic_arr,
            xyz_cache[i],
            intensity_cache[i],
            ground_mask_cache[i],
            cfg,
            bag_id,
            chunk_id,
            any_intensity,
            mf_mos_dynamic_arr=sweep_mf_mos_dynamic_arr,
        )
        total_static += result.n_static
        total_dynamic += result.n_dynamic
        if result.static_xyz is not None:
            static_xyz_chunks.append(result.static_xyz)
        if result.static_intensity is not None:
            static_intensity_chunks.append(result.static_intensity)
        if result.dyn_xyz is not None:
            dyn_xyz_chunks.append(result.dyn_xyz)
            dyn_sweep_id_chunks.append(result.dyn_sweep_id)
        if result.dyn_intensity is not None:
            dyn_intensity_chunks.append(result.dyn_intensity)

        updated_meta.append(
            ProcessedSweepMeta(
                bag_id=row["bag_id"],
                chunk_id=row["chunk_id"],
                sweep_id=sweep_id,
                lidar_id=row["lidar_id"],
                reference_timestamp_ns=int(row["reference_timestamp_ns"]),
                n_points_total=keys.shape[0],
                n_points_static=result.n_static,
                n_points_dynamic=result.n_dynamic,
                n_points_ground=int(row.get("n_points_ground") or 0),
                world_path=row["world_path"],
                dynamic_mask_path=result.mask_uri,
                has_intensity=bool(row.get("has_intensity", False)),
                deskewed=bool(row.get("deskewed", False)),
                valid=True,
                drop_reason=None,
                world_xmin=row.get("world_xmin"),
                world_xmax=row.get("world_xmax"),
                world_ymin=row.get("world_ymin"),
                world_ymax=row.get("world_ymax"),
                world_zmin=row.get("world_zmin"),
                world_zmax=row.get("world_zmax"),
                frame_id=row.get("frame_id"),
                mf_mos_mask_path=row.get("mf_mos_mask_path"),
            ).model_dump()
        )

    write_table(
        updated_meta, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id)
    )

    out_uri = static_map_path(bag_id, chunk_id)
    save_kwargs: dict[str, np.ndarray] = {}
    if static_xyz_chunks:
        save_kwargs["xyz"] = np.concatenate(static_xyz_chunks, axis=0)
        if any_intensity and static_intensity_chunks:
            save_kwargs["intensity"] = np.concatenate(static_intensity_chunks)
    else:
        save_kwargs["xyz"] = np.empty((0, 3), dtype=np.float64)
    save_kwargs["voxel_size"] = np.float32(cfg.voxel_size_m)
    save_kwargs["origin"] = origin
    save_kwargs["static_voxel_keys"] = static_arr
    np.savez_compressed(local_path(out_uri), **save_kwargs)

    dyn_save_kwargs: dict[str, np.ndarray] = {}
    if dyn_xyz_chunks:
        dyn_save_kwargs["xyz"] = np.concatenate(dyn_xyz_chunks, axis=0)
        dyn_save_kwargs["sweep_id"] = np.concatenate(dyn_sweep_id_chunks)
        if any_intensity and dyn_intensity_chunks:
            dyn_save_kwargs["intensity"] = np.concatenate(dyn_intensity_chunks)
    else:
        dyn_save_kwargs["xyz"] = np.empty((0, 3), dtype=np.float64)
        dyn_save_kwargs["sweep_id"] = np.empty(0, dtype=np.int32)
    np.savez_compressed(
        local_path(dynamic_map_path(bag_id, chunk_id)), **dyn_save_kwargs
    )

    if use_log_odds:
        log.info(
            "chunk %s: static=%d dynamic=%d "
            "(log_odds: %d touched, %d evidenced, "
            "%d under-evidenced-with-hits, %d ambiguous, %d free-only)",
            chunk_id,
            total_static,
            total_dynamic,
            unique_keys.size,
            diag.get("n_evidenced", 0),
            diag.get("n_under_evidenced_with_hits", 0),
            diag.get("n_ambiguous", 0),
            diag.get("n_free_only", 0),
        )
    else:
        log.info(
            "chunk %s: static=%d dynamic=%d voxel_threshold=%d/%d",
            chunk_id,
            total_static,
            total_dynamic,
            threshold,
            len(meta_rows),
        )

    if cfg.save_voxel_occupancy and (use_log_odds and unique_keys.size > 0):
        write_chunk_voxel_occupancy(
            bag_id,
            chunk_id,
            unique_keys,
            cfg.voxel_size_m,
            origin,
            log_odds=lo_vals,
            n_obs=n_obs_vals,
            n_hits=n_hits_vals,
        )
    elif cfg.save_voxel_occupancy:
        # Persistence path: emit a coords-only occupancy file built from the
        # union of all sweep_keys.
        all_keys_parts = [k for k in sweep_keys if k.size > 0]
        if all_keys_parts:
            unique_persistence_keys = np.unique(np.concatenate(all_keys_parts))
            write_chunk_voxel_occupancy(
                bag_id,
                chunk_id,
                unique_persistence_keys,
                cfg.voxel_size_m,
                origin,
            )

    if cfg.save_per_frame_voxel_occupancy and frame_keys:
        write_per_frame_voxel_occupancy(
            bag_id, chunk_id, frame_keys, cfg.voxel_size_m, origin
        )

    # Per-voxel diagnostics: full log_odds/n_obs/n_hits/classification for
    # every voxel, including carved (log_odds < 0) ones that
    # voxel_occupancy.npz filters out.  Powers the viz's p_occ color mode
    # for the dynamic cloud.  log_odds path only — persistence doesn't
    # produce per-voxel log_odds.
    if cfg.save_voxel_diagnostics and use_log_odds and unique_keys.size > 0:
        write_chunk_voxel_diagnostics(
            bag_id,
            chunk_id,
            unique_keys,
            lo_vals,
            n_obs_vals,
            n_hits_vals,
            cfg.voxel_size_m,
            origin,
            p_static_threshold=cfg.p_static_threshold,
            p_dynamic_threshold=cfg.p_dynamic_threshold,
            min_observations=cfg.min_observations,
            min_occupied_hits=cfg.min_occupied_hits,
        )

    return ClassifyResult(
        n_static=total_static,
        n_dynamic=total_dynamic,
        static_map_path=out_uri,
        cache_auto_disabled=cache_auto_disabled,
        estimated_cache_bytes=estimated_bytes,
    )


def _invalid_meta_row(row: dict, sweep_id: int) -> dict:
    """Pass-through meta row for valid=False sweeps (deskew failed upstream)."""
    return ProcessedSweepMeta(
        bag_id=row["bag_id"],
        chunk_id=row["chunk_id"],
        sweep_id=sweep_id,
        lidar_id=row["lidar_id"],
        reference_timestamp_ns=int(row["reference_timestamp_ns"]),
        n_points_total=int(row.get("n_points_total") or 0),
        n_points_static=0,
        n_points_dynamic=0,
        n_points_ground=int(row.get("n_points_ground") or 0),
        world_path=row.get("world_path", ""),
        dynamic_mask_path="",
        has_intensity=bool(row.get("has_intensity", False)),
        deskewed=bool(row.get("deskewed", False)),
        valid=False,
        drop_reason=row.get("drop_reason"),
        world_xmin=row.get("world_xmin"),
        world_xmax=row.get("world_xmax"),
        world_ymin=row.get("world_ymin"),
        world_ymax=row.get("world_ymax"),
        world_zmin=row.get("world_zmin"),
        world_zmax=row.get("world_zmax"),
        frame_id=row.get("frame_id"),
        mf_mos_mask_path=row.get("mf_mos_mask_path"),
    ).model_dump()
