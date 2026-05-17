"""Step B — Voxel-based static/dynamic classification.

Two-pass algorithm:
  Pass 1: For every sweep, discretize world-frame points to voxel keys (packed
          int64).  Concatenate (key, sweep_idx) pairs across sweeps and use
          np.unique to count how many distinct sweeps populated each voxel.
  Pass 2: For every sweep, apply the static/dynamic label as a boolean mask
          (np.searchsorted into a sorted static-key array) and write
          {sweep_id:06d}_dynamic_mask.npy.

The static cloud (all points from static voxels) is written to static_map.npz
and later consumed by ground extraction and downstream components.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

from wato_common.artifact_store import (
    dynamic_map_path,
    dynamic_mask_path,
    ensure_local_dir,
    lidar_proc_dir,
    lidar_proc_index_path,
    local_path,
    static_map_path,
    voxel_occupancy_frame_path,
    voxel_occupancy_path,
)
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import PROCESSED_SWEEPS_SCHEMA, ProcessedSweepMeta
from wato_lidar_preprocessing.config import ComponentConfig
from wato_lidar_preprocessing.voxel import (
    AXIS_BITS,
    VoxelOverflowError,
    pack_voxel_key,
    voxel_indices,
)

log = logging.getLogger(__name__)

# Re-exports so existing call sites (and tests) continue to import these
# names from classify; the implementations live in voxel.py.
_pack_voxel_key = pack_voxel_key
_voxel_indices = voxel_indices
__all__ = ["VoxelOverflowError", "process_chunk", "ClassifyResult"]

# Auto-disable the in-memory xyz/intensity cache when the estimated total size
# exceeds this many bytes.  At 4 GB we cap roughly 250 M float64-xyz points
# (~25 sweeps of 10 M each); the explicit cfg flag still acts as a force-on
# override.  Set the env var WATO_LIDAR_CACHE_BYTES to override per-host.
_DEFAULT_CACHE_BYTES = 4 * 1024**3


def _cache_byte_budget() -> int:
    """Resolve the cache size cap, honouring WATO_LIDAR_CACHE_BYTES if set.

    Invalid values (non-int, <=0) are ignored with a warning so a typo in
    the env doesn't silently disable the safety net.
    """
    raw = os.environ.get("WATO_LIDAR_CACHE_BYTES")
    if raw is None:
        return _DEFAULT_CACHE_BYTES
    try:
        v = int(raw)
        if v <= 0:
            raise ValueError("must be > 0")
        return v
    except (TypeError, ValueError) as exc:
        log.warning(
            "WATO_LIDAR_CACHE_BYTES=%r ignored (%s); falling back to default %d",
            raw,
            exc,
            _DEFAULT_CACHE_BYTES,
        )
        return _DEFAULT_CACHE_BYTES


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
    """Write sentinel static_map.npz + dynamic_map.npz when there's nothing to classify."""
    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    out_uri = static_map_path(bag_id, chunk_id)
    np.savez_compressed(
        local_path(out_uri),
        xyz=np.empty((0, 3), dtype=np.float64),
        # NOTE: voxel_size and origin describe THIS chunk's local grid only —
        # different chunks of the same bag use different origins.  reduce.py
        # ignores them and works directly off xyz.
        voxel_size=np.float32(voxel_size),
        origin=np.zeros(3, dtype=np.float64),
        static_voxel_keys=np.empty(0, dtype=np.int64),
    )
    # Empty dynamic_map.npz so downstream consumers can load unconditionally
    # — distinguishing "chunk has no dynamic points" from "chunk not yet
    # processed" still works via lidar_proc_summary.parquet.
    np.savez_compressed(
        local_path(dynamic_map_path(bag_id, chunk_id)),
        xyz=np.empty((0, 3), dtype=np.float64),
        sweep_id=np.empty(0, dtype=np.int32),
    )
    return ClassifyResult(0, 0, out_uri)


def _count_unique_sweeps_per_voxel(
    sweep_keys: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Count distinct sweeps each voxel appears in.

    Builds a single (key, sweep_idx) array, dedupes per (voxel, sweep) so a
    voxel hit twice within one sweep counts once, then groups by voxel key
    and counts sweeps.  Avoids the per-voxel Python `set` allocation that
    used to dominate runtime + memory.

    Returns:
        unique_keys: sorted int64 array of voxel keys
        sweep_counts: int64 array, parallel to unique_keys
    """
    parts_keys: list[np.ndarray] = []
    parts_idx: list[np.ndarray] = []
    for i, keys in enumerate(sweep_keys):
        if keys.size == 0:
            continue
        # Dedupe within this sweep so duplicate hits don't inflate the count.
        unique_in_sweep = np.unique(keys)
        parts_keys.append(unique_in_sweep)
        parts_idx.append(np.full(unique_in_sweep.shape[0], i, dtype=np.int64))

    if not parts_keys:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    all_keys = np.concatenate(parts_keys)
    # Each (key, sweep_idx) pair is already unique by construction (we deduped
    # within each sweep above), so np.unique on keys with return_counts gives
    # us the number of distinct sweeps per voxel directly.
    unique_keys, sweep_counts = np.unique(all_keys, return_counts=True)
    return unique_keys, sweep_counts.astype(np.int64)


def _load_world_xyz_intensity(
    world_path_uri: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Single load of a world NPZ → (xyz, intensity-or-None)."""
    data = np.load(local_path(world_path_uri))
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1)
    intensity = data["intensity"].astype(np.float32) if "intensity" in data else None
    return xyz, intensity


def _origin_from_index(meta_rows: list[dict]) -> np.ndarray | None:
    """Compute the chunk-level world-frame origin from the parquet bbox.

    Reads the per-sweep `world_xmin/ymin/zmin` columns populated by deskew.
    Returns None if every valid row is empty (e.g. no points survived the
    nonfinite filter), in which case the caller writes an empty sentinel.
    Raises ValueError if a row is non-empty but missing bbox columns —
    that means the parquet was written by a stale deskew and the caller
    needs to re-run it.
    """
    xmins, ymins, zmins = [], [], []
    for row in meta_rows:
        if row.get("valid") is False:
            continue
        xm = row.get("world_xmin")
        ym = row.get("world_ymin")
        zm = row.get("world_zmin")
        if xm is None or ym is None or zm is None:
            n_pts = int(row.get("n_points_total") or 0)
            if n_pts > 0:
                raise ValueError(
                    f"sweep {row.get('sweep_id')!r} has n_points_total={n_pts} "
                    "but is missing world_x/y/zmin columns — parquet was "
                    "written by a stale deskew; re-run lidar_preprocessing "
                    "with --force on this chunk."
                )
            continue
        xmins.append(xm)
        ymins.append(ym)
        zmins.append(zm)
    if not xmins:
        return None
    return np.array([min(xmins), min(ymins), min(zmins)], dtype=np.float64)


def _estimate_cache_bytes(meta_rows: list[dict]) -> int:
    """Estimate the in-memory size of (xyz float64 + intensity float32) caches."""
    total_pts = 0
    for r in meta_rows:
        if r.get("valid") is False:
            continue
        total_pts += int(r.get("n_points_total") or 0)
    # 3 × float64 (xyz) + float32 (intensity) per point.
    return total_pts * (3 * 8 + 4)


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

    # Pre-scan: do any sweeps carry intensity?  If so, every contributing sweep
    # gets a real-or-zero intensity contribution so the output column aligns
    # with xyz (no length mismatch).
    any_intensity = any(bool(r.get("has_intensity", False)) for r in meta_rows)

    # Auto-disable the in-memory cache when the chunk is too big.  The config
    # flag stays as a force-on override; users who set it explicitly opt out
    # of the safety net.  Done before any NPZ load so we never page in data
    # we'd immediately throw away.
    cache_budget = _cache_byte_budget()
    estimated_bytes = _estimate_cache_bytes(meta_rows)
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

    # Build the chunk-level origin from the parquet bbox columns deskew wrote.
    # _origin_from_index raises if a non-empty sweep lacks those columns,
    # which means the parquet is stale and the caller needs to re-run deskew.
    origin = _origin_from_index(meta_rows)
    if origin is None:
        # Every valid sweep had n_points_total == 0 → nothing to classify.
        return _write_empty_outputs(bag_id, chunk_id, cfg.voxel_size_m)

    # Pass 1: load each sweep once, build voxel keys, and (optionally) cache
    # xyz / intensity for pass 2.  This is the single source of NPZ I/O when
    # the cache is on.
    xyz_cache: list[np.ndarray | None] = []
    intensity_cache: list[np.ndarray | None] = []
    sweep_keys: list[np.ndarray] = []  # one int64 array per sweep
    # frame_id → list of key arrays (for per-frame voxel occupancy export).
    frame_keys: dict[int, list[np.ndarray]] = {}

    for row in meta_rows:
        if row.get("valid") is False:
            xyz_cache.append(None)
            intensity_cache.append(None)
            sweep_keys.append(np.empty(0, dtype=np.int64))
            continue
        xyz, intensity = _load_world_xyz_intensity(row["world_path"])
        if xyz.shape[0] == 0:
            xyz_cache.append(xyz if cache_xyz else None)
            intensity_cache.append(intensity if cache_xyz else None)
            sweep_keys.append(np.empty(0, dtype=np.int64))
            continue
        keys = _voxel_indices(xyz, origin, cfg.voxel_size_m, chunk_id=chunk_id)
        sweep_keys.append(keys)
        xyz_cache.append(xyz if cache_xyz else None)
        intensity_cache.append(intensity if cache_xyz else None)
        fid = row.get("frame_id")
        if fid is not None:
            frame_keys.setdefault(int(fid), []).append(keys)

    unique_keys, sweep_counts = _count_unique_sweeps_per_voxel(sweep_keys)

    n_sweeps = len(meta_rows)
    threshold = max(cfg.static_sweep_min, int(cfg.static_sweep_fraction * n_sweeps))

    # Sorted int64 array of static voxel keys for searchsorted lookup.
    if unique_keys.size > 0:
        static_arr = unique_keys[sweep_counts >= threshold]
    else:
        static_arr = np.empty(0, dtype=np.int64)

    # Pass 2: per-sweep masks + accumulate static cloud + dynamic cloud.
    # The dynamic accumulators mirror the static ones so downstream consumers
    # (proposal_generation's LiDAR detector, SLF candidate clustering) can
    # load one artifact per chunk instead of iterating every sweep.
    static_xyz_chunks: list[np.ndarray] = []
    static_intensity_chunks: list[np.ndarray] = []
    dyn_xyz_chunks: list[np.ndarray] = []
    dyn_intensity_chunks: list[np.ndarray] = []
    dyn_sweep_id_chunks: list[np.ndarray] = []
    total_static = 0
    total_dynamic = 0

    updated_meta: list[dict] = []

    for i, (row, keys) in enumerate(zip(meta_rows, sweep_keys)):
        sweep_id = int(row["sweep_id"])
        # valid=False rows from upstream (deskew failure) pass straight through
        # so the failure context survives to downstream stages.  parquet stores
        # missing columns as None, which we treat as valid=True.
        if row.get("valid") is False:
            updated_meta.append(
                ProcessedSweepMeta(
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
            )
            continue
        n = keys.shape[0]
        has_intensity = bool(row.get("has_intensity", False))

        if n == 0:
            mask = np.zeros(0, dtype=bool)
            n_static_s = n_dyn_s = 0
        else:
            # static_arr is sorted; searchsorted + equality check is the fast
            # path (np.isin defaults to a hash-table when arr is not unique).
            if static_arr.size > 0:
                pos = np.searchsorted(static_arr, keys)
                pos = np.clip(pos, 0, static_arr.size - 1)
                is_static = static_arr[pos] == keys
            else:
                is_static = np.zeros(n, dtype=bool)
            mask = ~is_static  # True == dynamic
            n_dyn_s = int(mask.sum())
            n_static_s = n - n_dyn_s

        dyn_uri = dynamic_mask_path(bag_id, chunk_id, sweep_id)
        np.save(local_path(dyn_uri), mask)

        total_static += n_static_s
        total_dynamic += n_dyn_s

        if n_static_s > 0 or n_dyn_s > 0:
            cached_xyz = xyz_cache[i]
            cached_intensity = intensity_cache[i]
            if cached_xyz is not None:
                xyz = cached_xyz
                intensity = cached_intensity
            else:
                xyz, intensity = _load_world_xyz_intensity(row["world_path"])
            static_mask = ~mask

            if n_static_s > 0:
                static_xyz_chunks.append(xyz[static_mask])
                if any_intensity:
                    if has_intensity and intensity is not None:
                        static_intensity_chunks.append(
                            intensity[static_mask].astype(np.float32)
                        )
                    else:
                        static_intensity_chunks.append(
                            np.zeros(n_static_s, dtype=np.float32)
                        )

            if n_dyn_s > 0:
                dyn_xyz_chunks.append(xyz[mask])
                # sweep_id-per-point lets downstream recover temporal origin
                # without iterating lidar_proc_index.  int32 is enough for any
                # realistic chunk (32 k sweeps).
                dyn_sweep_id_chunks.append(np.full(n_dyn_s, sweep_id, dtype=np.int32))
                if any_intensity:
                    if has_intensity and intensity is not None:
                        dyn_intensity_chunks.append(intensity[mask].astype(np.float32))
                    else:
                        dyn_intensity_chunks.append(np.zeros(n_dyn_s, dtype=np.float32))

        updated_meta.append(
            ProcessedSweepMeta(
                bag_id=row["bag_id"],
                chunk_id=row["chunk_id"],
                sweep_id=sweep_id,
                lidar_id=row["lidar_id"],
                reference_timestamp_ns=int(row["reference_timestamp_ns"]),
                n_points_total=n,
                n_points_static=n_static_s,
                n_points_dynamic=n_dyn_s,
                n_points_ground=int(row.get("n_points_ground") or 0),
                world_path=row["world_path"],
                dynamic_mask_path=dyn_uri,
                has_intensity=has_intensity,
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

    # Write updated index with static/dynamic counts.
    write_table(
        updated_meta, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id)
    )

    # Write static map.  origin / voxel_size are chunk-local descriptors —
    # different chunks of the same bag use different origins.  reduce.py
    # consumes only `xyz`.
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
    # Sorted int64 array of static voxel keys.  Step C (ground) intersects
    # per-sweep ground points against this so dynamic-vehicle-underside
    # returns don't survive into the chunk ground cloud.
    save_kwargs["static_voxel_keys"] = static_arr

    np.savez_compressed(local_path(out_uri), **save_kwargs)

    # Dynamic map: symmetric to static map; carries sweep_id per point so
    # downstream (proposal_generation LiDAR detector, SLF candidate seeding,
    # tracking ReID) can recover temporal info via a join on lidar_proc_index.
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
        "chunk %s: static=%d dynamic=%d voxel_threshold=%d/%d",
        chunk_id,
        total_static,
        total_dynamic,
        threshold,
        n_sweeps,
    )

    # Shared bit-mask for unpacking packed int64 keys → (vx, vy, vz) int32.
    # Used by both the per-chunk and per-frame occupancy blocks below.
    axis_mask = np.int64((1 << AXIS_BITS) - 1)

    if cfg.save_voxel_occupancy and unique_keys.size > 0:
        # All occupied voxels (static + dynamic) aggregated across the full
        # chunk.  Useful for QA/visualization but NOT what SAM4D's MinkUNet
        # encoder consumes directly — use save_per_frame_voxel_occupancy for
        # that (each file covers one frame, not the whole chunk).
        vz = (unique_keys & axis_mask).astype(np.int32)
        vy = ((unique_keys >> np.int64(AXIS_BITS)) & axis_mask).astype(np.int32)
        vx = ((unique_keys >> np.int64(2 * AXIS_BITS)) & axis_mask).astype(np.int32)
        occ_coords = np.stack([vx, vy, vz], axis=1)  # (N, 3) int32
        np.savez_compressed(
            local_path(voxel_occupancy_path(bag_id, chunk_id)),
            coords=occ_coords,
            origin=origin,
            voxel_size=np.float32(cfg.voxel_size_m),
        )
        log.info(
            "chunk %s: wrote voxel_occupancy.npz (%d occupied voxels)",
            chunk_id,
            occ_coords.shape[0],
        )

    if cfg.save_per_frame_voxel_occupancy and frame_keys:
        n_frames = 0
        for fid, key_list in frame_keys.items():
            merged = np.unique(np.concatenate(key_list))
            if merged.size == 0:
                continue
            vz = (merged & axis_mask).astype(np.int32)
            vy = ((merged >> np.int64(AXIS_BITS)) & axis_mask).astype(np.int32)
            vx = ((merged >> np.int64(2 * AXIS_BITS)) & axis_mask).astype(np.int32)
            frame_coords = np.stack([vx, vy, vz], axis=1)  # (N, 3) int32
            np.savez_compressed(
                local_path(voxel_occupancy_frame_path(bag_id, chunk_id, fid)),
                coords=frame_coords,
                origin=origin,
                voxel_size=np.float32(cfg.voxel_size_m),
            )
            n_frames += 1
        log.info(
            "chunk %s: wrote %d per-frame voxel_occupancy files", chunk_id, n_frames
        )

    return ClassifyResult(
        n_static=total_static,
        n_dynamic=total_dynamic,
        static_map_path=out_uri,
        cache_auto_disabled=cache_auto_disabled,
        estimated_cache_bytes=estimated_bytes,
    )
