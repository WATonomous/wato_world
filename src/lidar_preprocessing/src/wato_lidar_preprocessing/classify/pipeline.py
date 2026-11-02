"""Step B — voxel static/dynamic classification.

  Pass 1: ray-trace every sweep into a log-odds voxel grid (constants from
          the SensorModel; the carve margin also uses the bag's pose σ),
          frame by frame: a frame's hits shield their one-voxel
          neighbourhood from its carving, and evidence sums unclamped.
  Pass 2: per sweep, the dynamic candidates — voxel in the carved-dynamic set,
          not ground, beyond cfg.dynamic_min_range_m. dynamic_filter then
          cleans them across the chunk, and each sweep's
          {sweep_id:06d}_dynamic_mask.npy is written once. Never filters keys:
          the mask must match the world NPZ's length.

static_map.npz also carries the static / dynamic / ambiguous voxel-key sets
that Step C and Step F read: the raw voxel verdict, before dynamic_filter.
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
from wato_lidar_preprocessing._inputs import load_pose_samples
from wato_lidar_preprocessing.config import ComponentConfig
from wato_lidar_preprocessing.sensor_model import estimate_pose_sigma_m

from .dynamic_filter import filter_candidates, sweep_rate_hz
from .global_map_prior import GlobalMapPrior
from .io_helpers import (
    cache_byte_budget,
    estimate_cache_bytes,
    origin_from_index,
)
from .log_odds import (
    CLASS_AMBIGUOUS,
    build_log_odds_grid,
    classify_from_log_odds,
    frame_groups,
)
from .masking import apply_classification_to_sweep, dynamic_candidates
from .occupancy_export import (
    write_chunk_voxel_diagnostics,
    write_chunk_voxel_occupancy,
    write_per_frame_voxel_occupancy,
)

log = logging.getLogger(__name__)


@dataclass
class ClassifyResult:
    n_static: int
    n_dynamic: int
    static_map_path: str
    n_cluster_dropped: int = 0
    n_persistence_dropped: int = 0
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
        dynamic_voxel_keys=np.empty(0, dtype=np.int64),
        ambiguous_voxel_keys=np.empty(0, dtype=np.int64),
    )
    np.savez_compressed(
        local_path(dynamic_map_path(bag_id, chunk_id)),
        xyz=np.empty((0, 3), dtype=np.float64),
        sweep_id=np.empty(0, dtype=np.int32),
    )
    return ClassifyResult(0, 0, out_uri)


def _estimate_pose_sigma_m(bag_id: str, chunk_id: str, floor_m: float) -> float:
    """Per-bag SLAM pose noise [m] from this chunk's poses.parquet jitter."""
    try:
        samples = load_pose_samples(bag_id, chunk_id)
    except Exception as exc:  # noqa: BLE001 — missing/short poses → fall back
        log.warning(
            "chunk %s: could not load poses for σ_pose (%s); using floor", chunk_id, exc
        )
        return floor_m
    if len(samples) < 3:
        return floor_m
    translations = np.array([s.translation for s in samples], dtype=np.float64)
    return estimate_pose_sigma_m(translations, floor_m=floor_m)


def process_chunk(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
    *,
    global_map_prior: GlobalMapPrior | None = None,
) -> ClassifyResult:
    """Classify all world-frame sweeps in a chunk as static or dynamic.

    Args:
        global_map_prior: optional bag-level static map prior (two-pass mode).
            When set, every map-matched endpoint gets a one-time credibility-
            weighted prior shift derived from the sensor model.
    """
    meta_rows = read_rows(lidar_proc_index_path(bag_id, chunk_id))
    if not meta_rows:
        log.warning(
            "chunk %s: no processed sweeps in lidar_proc_index — writing empty static map",
            chunk_id,
        )
        return _write_empty_outputs(bag_id, chunk_id, cfg.voxel_size_m)

    any_intensity = any(bool(r.get("has_intensity", False)) for r in meta_rows)

    # Auto-disable the in-memory cache when the chunk is too big.
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

    sensor_model = cfg.build_sensor_model()
    pose_sigma_m = _estimate_pose_sigma_m(
        bag_id, chunk_id, floor_m=sensor_model.range_sigma_m
    )

    (
        xyz_cache,
        intensity_cache,
        ground_mask_cache,
        origin_cache,
        sweep_keys,
        frame_keys,
        (unique_keys, lo_vals, n_obs_vals, n_hits_vals),
    ) = build_log_odds_grid(
        meta_rows,
        cfg,
        origin,
        chunk_id,
        cache_xyz=cache_xyz,
        pose_sigma_m=pose_sigma_m,
        sensor_model=sensor_model,
        global_map_prior=global_map_prior,
    )
    (
        static_arr,
        dynamic_arr,
        classification,
        diag,
    ) = classify_from_log_odds(
        unique_keys, lo_vals, n_obs_vals, n_hits_vals, cfg, sensor_model
    )

    # Pass 2a: the voxel verdict per point, every sweep of the chunk.
    masks: list[np.ndarray | None] = [None] * len(meta_rows)
    cand_xyz: list[np.ndarray] = []
    cand_sweep: list[np.ndarray] = []
    cand_frame: list[np.ndarray] = []
    groups = frame_groups(meta_rows, cfg.frame_sync.canonical_lidar)
    for frame, group in enumerate(groups):
        for i in group:
            row = meta_rows[i]
            mask, xyz = dynamic_candidates(
                row,
                sweep_keys[i],
                dynamic_arr,
                xyz_cache[i],
                ground_mask_cache[i],
                origin_cache[i],
                dynamic_min_range_m=cfg.dynamic_min_range_m,
            )
            masks[i] = mask
            n = int(mask.sum())
            if n:
                cand_xyz.append(xyz[mask])
                cand_sweep.append(np.full(n, i, dtype=np.int64))
                cand_frame.append(np.full(n, frame, dtype=np.int64))

    # Pass 2b: chunk-wide cleanup of the candidates, before anything is written.
    n_cluster_dropped = n_persistence_dropped = 0
    if cand_xyz:
        filtered = filter_candidates(
            cfg.dynamic_filter,
            np.concatenate(cand_xyz),
            np.concatenate(cand_sweep),
            np.concatenate(cand_frame),
            sweep_rate_hz(
                [
                    int(r["reference_timestamp_ns"])
                    for r in meta_rows
                    if r.get("valid") is not False
                ]
            ),
        )
        n_cluster_dropped = filtered.n_cluster_dropped
        n_persistence_dropped = filtered.n_persistence_dropped
        # Candidates are laid out sweep by sweep, in each mask's point order.
        offset = 0
        for part in cand_sweep:
            i, n = int(part[0]), part.shape[0]
            mask = masks[i]
            mask[np.flatnonzero(mask)[~filtered.keep[offset : offset + n]]] = False
            offset += n
    del cand_xyz, cand_sweep, cand_frame

    # Pass 2c: write the masks and accumulate the static/dynamic clouds.
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

        result = apply_classification_to_sweep(
            row,
            sweep_id,
            keys,
            static_arr,
            masks[i],
            xyz_cache[i],
            intensity_cache[i],
            ground_mask_cache[i],
            bag_id,
            chunk_id,
            any_intensity,
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
    # Voxels classed as movers (the carved-dynamic set), sorted. Step C drops
    # ground points landing in them; Step F seeds proposals from them.
    save_kwargs["dynamic_voxel_keys"] = dynamic_arr
    # Evidenced voxels with hits whose p_occ sits between the dynamic and
    # static thresholds, sorted. Neither cloud uses them (conservative), but
    # Step F's recall-oriented motion proposals flag them (AW_AMBIGUOUS) and
    # attach them to clusters seeded by stronger sources.
    save_kwargs["ambiguous_voxel_keys"] = unique_keys[classification == CLASS_AMBIGUOUS]
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

    log.info(
        "chunk %s: static=%d dynamic=%d (filter dropped %d cluster + %d "
        "persistence; log_odds: %d touched, %d evidenced, %d dynamic-voxels, "
        "%d under-evidenced-with-hits, %d ambiguous, %d free-only)",
        chunk_id,
        total_static,
        total_dynamic,
        n_cluster_dropped,
        n_persistence_dropped,
        unique_keys.size,
        diag.get("n_evidenced", 0),
        diag.get("n_dynamic_voxels", 0),
        diag.get("n_under_evidenced_with_hits", 0),
        diag.get("n_ambiguous", 0),
        diag.get("n_free_only", 0),
    )

    if cfg.save_voxel_occupancy and unique_keys.size > 0:
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

    if cfg.save_per_frame_voxel_occupancy and frame_keys:
        write_per_frame_voxel_occupancy(
            bag_id, chunk_id, frame_keys, cfg.voxel_size_m, origin
        )

    # voxel_diag.npz includes carved (log_odds < 0) voxels that
    # voxel_occupancy.npz filters out.
    if cfg.save_voxel_diagnostics and unique_keys.size > 0:
        write_chunk_voxel_diagnostics(
            bag_id,
            chunk_id,
            unique_keys,
            lo_vals,
            n_obs_vals,
            n_hits_vals,
            classification,
            cfg.voxel_size_m,
            origin,
        )

    return ClassifyResult(
        n_static=total_static,
        n_dynamic=total_dynamic,
        static_map_path=out_uri,
        n_cluster_dropped=n_cluster_dropped,
        n_persistence_dropped=n_persistence_dropped,
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
    ).model_dump()
