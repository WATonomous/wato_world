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
    lidar_sweep_path,
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
    load_world_xyz_intensity,
    origin_from_index,
)
from .log_odds import build_log_odds_grid, classify_from_log_odds
from .masking import apply_classification_to_sweep
from .occupancy_export import (
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


def _load_mf_mos_world_mask(
    bag_id: str,
    chunk_id: str,
    row: dict,
    n_world: int,
    filter_nonfinite: bool,
) -> np.ndarray | None:
    """Load an MF-MOS raw-length mask and align it to world-frame length.

    MF-MOS masks are (n_raw,) bool aligned to the raw lidar NPZ.  Deskew
    applies a nonfinite filter, so n_world <= n_raw.  We re-apply the same
    filter here so the mask can be fused with the (n_world,) voxel mask.

    Returns None if the mask is unavailable or can't be loaded.
    """
    mf_path = row.get("mf_mos_mask_path")
    if not mf_path:
        return None
    try:
        mf_raw = np.load(local_path(mf_path))
    except Exception as exc:  # noqa: BLE001
        log.warning("could not load mf_mos mask %s: %s", mf_path, exc)
        return None

    n_raw = mf_raw.shape[0]
    if n_raw == n_world:
        return mf_raw  # common case: no NaN/inf points were dropped

    if n_raw < n_world:
        log.warning(
            "sweep %s: mf_mos mask len %d < world len %d — fusion skipped for this sweep",
            row.get("sweep_id"),
            n_raw,
            n_world,
        )
        return None

    # n_raw > n_world: re-apply the nonfinite filter to get world-aligned mask.
    raw_uri = lidar_sweep_path(
        bag_id,
        chunk_id,
        str(row.get("lidar_id", "")),
        int(row.get("sweep_id", 0)),
    )
    try:
        raw = np.load(local_path(raw_uri))
    except Exception as exc:  # noqa: BLE001
        log.warning("could not load raw sweep for mf_mos fusion alignment: %s", exc)
        return None

    x_r = raw["x"].astype(np.float64)
    y_r = raw["y"].astype(np.float64)
    z_r = raw["z"].astype(np.float64)
    if filter_nonfinite:
        finite = np.isfinite(x_r) & np.isfinite(y_r) & np.isfinite(z_r)
    else:
        finite = np.ones(n_raw, dtype=bool)

    mf_world = mf_raw[finite]
    if mf_world.shape[0] != n_world:
        log.warning(
            "sweep %s: mf_mos world-aligned len %d != world len %d — fusion skipped",
            row.get("sweep_id"),
            mf_world.shape[0],
            n_world,
        )
        return None
    return mf_world


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
    ground_mask_cache is always all-None for persistence — bug fix #4 only
    applies to the log-odds path.
    """
    xyz_cache: list[np.ndarray | None] = []
    intensity_cache: list[np.ndarray | None] = []
    ground_mask_cache: list[np.ndarray | None] = [None] * len(meta_rows)
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
            sweep_keys.append(np.empty(0, dtype=np.int64))
            continue

        xyz, intensity = load_world_xyz_intensity(row["world_path"])
        if xyz.shape[0] == 0:
            xyz_cache.append(xyz if cache_xyz else None)
            intensity_cache.append(intensity if cache_xyz else None)
            sweep_keys.append(np.empty(0, dtype=np.int64))
            continue

        keys = voxel_indices(xyz, origin, cfg.voxel_size_m, chunk_id=chunk_id)
        sweep_keys.append(keys)
        xyz_cache.append(xyz if cache_xyz else None)
        intensity_cache.append(intensity if cache_xyz else None)
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
            (unique_keys, lo_vals, n_obs_vals, n_hits_vals),
        ) = build_log_odds_grid(
            meta_rows, cfg, origin, chunk_id, cache_xyz=cache_xyz
        )
        static_arr, not_dynamic_arr, diag = classify_from_log_odds(
            unique_keys, lo_vals, n_obs_vals, n_hits_vals, cfg
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

    use_mf_mos_fusion = (
        cfg.mf_mos.enabled and cfg.mf_mos.fusion_mode != "independent"
    )

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

        # Load MF-MOS mask when fusion is active and points exist.
        mf_mos_mask = None
        if use_mf_mos_fusion and keys.shape[0] > 0:
            mf_mos_mask = _load_mf_mos_world_mask(
                bag_id, chunk_id, row, keys.shape[0], cfg.filter_nonfinite_points
            )

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
            mf_mos_mask=mf_mos_mask,
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
            "%d under-evidenced-with-hits, %d free-only)",
            chunk_id,
            total_static,
            total_dynamic,
            unique_keys.size,
            diag.get("n_evidenced", 0),
            diag.get("n_under_evidenced_with_hits", 0),
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
