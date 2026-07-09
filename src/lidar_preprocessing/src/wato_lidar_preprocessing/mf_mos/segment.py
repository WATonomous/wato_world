"""MF-MOS static/dynamic decomposition (`--seg mos`).

The MF-MOS counterpart to `classify/` (Amanatides-Woo). Fully independent of
the AW path — no ray traversal, no log-odds, no imports from classify/. The
per-sweep dynamic mask comes straight from the MF-MOS moving masks written by
`mf_mos._core.process_chunk`:

    dynamic = mf_mos_moving & ~ground & ~near_ego
    static  = ~mf_mos_moving & ~ground

Patchwork++ ground stays authoritative — ground points belong to ground.npz,
never static_map/dynamic_map — mirroring classify/masking.py so Step C
behaves identically regardless of which segmentation method ran. The same
near-ego gate classify pass 2 applies (cfg.dynamic_min_range_m) suppresses
moving verdicts on ego self-returns; gated points drop from both clouds
rather than being promoted to static.

Outputs are byte-compatible with classify/ (same static_map.npz /
dynamic_map.npz / dynamic_mask.npy / lidar_proc_index columns) so ground,
reduce, and downstream consumers don't care which method produced them.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from wato_common.artifact_store import (
    dynamic_map_path,
    dynamic_mask_path,
    ensure_local_dir,
    lidar_proc_dir,
    lidar_proc_index_path,
    lidar_sweep_path,
    local_path,
    mf_mos_score_path,
    static_map_path,
    voxel_occupancy_frame_path,
    voxel_occupancy_path,
)
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import PROCESSED_SWEEPS_SCHEMA, ProcessedSweepMeta
from wato_lidar_preprocessing.config import ComponentConfig
from wato_lidar_preprocessing.voxel import AXIS_BITS, voxel_indices

log = logging.getLogger(__name__)

_AXIS_MASK = np.int64((1 << AXIS_BITS) - 1)


@dataclass
class MosSegmentResult:
    """Result of the MF-MOS static/dynamic decomposition.

    Field names mirror classify.ClassifyResult (including the cache fields,
    which are always inert here) so the orchestrator can hand either result
    to _write_chunk_summary unchanged.
    """

    n_static: int
    n_dynamic: int
    static_map_path: str
    n_sweeps_no_mask: int = 0
    cache_auto_disabled: bool = False
    estimated_cache_bytes: int = 0


def _coords_from_keys(keys: np.ndarray) -> np.ndarray:
    vz = (keys & _AXIS_MASK).astype(np.int32)
    vy = ((keys >> np.int64(AXIS_BITS)) & _AXIS_MASK).astype(np.int32)
    vx = ((keys >> np.int64(2 * AXIS_BITS)) & _AXIS_MASK).astype(np.int32)
    return np.stack([vx, vy, vz], axis=1)


def _origin_from_index(meta_rows: list[dict]) -> np.ndarray | None:
    """Chunk world-frame origin from the per-sweep bbox columns.

    Mirrors classify.io_helpers.origin_from_index so the voxel keys this
    module packs into static_voxel_keys align with the origin/voxel_size it
    writes into static_map.npz (the only consistency Step C ground requires).
    """
    xmins, ymins, zmins = [], [], []
    for row in meta_rows:
        if row.get("valid") is False:
            continue
        xm = row.get("world_xmin")
        ym = row.get("world_ymin")
        zm = row.get("world_zmin")
        if xm is None or ym is None or zm is None:
            if int(row.get("n_points_total") or 0) > 0:
                raise ValueError(
                    f"sweep {row.get('sweep_id')!r} has points but no world bbox "
                    "columns — re-run deskew (--force) on this chunk."
                )
            continue
        xmins.append(xm)
        ymins.append(ym)
        zmins.append(zm)
    if not xmins:
        return None
    return np.array([min(xmins), min(ymins), min(zmins)], dtype=np.float64)


def _align_raw_to_world(
    raw_arr: np.ndarray,
    bag_id: str,
    chunk_id: str,
    row: dict,
    n_world: int,
    filter_nonfinite: bool,
    what: str,
) -> np.ndarray | None:
    """Align a raw-length per-point MF-MOS array to world-frame length.

    MF-MOS artifacts are (n_raw,) aligned to the raw lidar NPZ. Deskew
    applies a nonfinite filter, so n_world <= n_raw. We re-apply the same
    filter here so the array is index-aligned with the (n_world,) world cloud.

    Returns None if the array can't be reconciled.
    """
    n_raw = raw_arr.shape[0]
    if n_raw == n_world:
        return raw_arr  # common case: no NaN/inf points were dropped

    if n_raw < n_world:
        log.warning(
            "sweep %s: mf_mos %s len %d < world len %d — discarded",
            row.get("sweep_id"),
            what,
            n_raw,
            n_world,
        )
        return None

    # n_raw > n_world: re-apply the nonfinite filter to recover the alignment.
    raw_uri = lidar_sweep_path(
        bag_id,
        chunk_id,
        str(row.get("lidar_id", "")),
        int(row.get("sweep_id", 0)),
    )
    try:
        raw = np.load(local_path(raw_uri))
    except Exception as exc:  # noqa: BLE001
        log.warning("could not load raw sweep for mf_mos alignment: %s", exc)
        return None

    x_r = raw["x"].astype(np.float64)
    y_r = raw["y"].astype(np.float64)
    z_r = raw["z"].astype(np.float64)
    if filter_nonfinite:
        finite = np.isfinite(x_r) & np.isfinite(y_r) & np.isfinite(z_r)
    else:
        finite = np.ones(n_raw, dtype=bool)

    aligned = raw_arr[finite]
    if aligned.shape[0] != n_world:
        log.warning(
            "sweep %s: mf_mos %s world-aligned len %d != world len %d — discarded",
            row.get("sweep_id"),
            what,
            aligned.shape[0],
            n_world,
        )
        return None
    return aligned


def load_mf_mos_world_mask(
    bag_id: str,
    chunk_id: str,
    row: dict,
    n_world: int,
    filter_nonfinite: bool,
) -> np.ndarray | None:
    """Load an MF-MOS raw-length mask and align it to world-frame length.

    Returns None if the mask is unavailable or can't be reconciled (the
    caller treats the sweep as having no MF-MOS verdict).
    """
    mf_path = row.get("mf_mos_mask_path")
    if not mf_path:
        return None
    try:
        mf_raw = np.load(local_path(mf_path))
    except Exception as exc:  # noqa: BLE001
        log.warning("could not load mf_mos mask %s: %s", mf_path, exc)
        return None
    return _align_raw_to_world(
        mf_raw, bag_id, chunk_id, row, n_world, filter_nonfinite, "mask"
    )


def load_mf_mos_world_scores(
    bag_id: str,
    chunk_id: str,
    row: dict,
    n_world: int,
    filter_nonfinite: bool,
) -> np.ndarray | None:
    """Load per-point MF-MOS moving probabilities, world-aligned.

    Scores are optional (written only when cfg.mf_mos.save_scores) — a
    missing file is normal and returns None without a warning.
    """
    score_uri = mf_mos_score_path(bag_id, chunk_id, int(row.get("sweep_id", 0)))
    score_path = local_path(score_uri)
    if not os.path.exists(score_path):
        return None
    try:
        raw = np.load(score_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not load mf_mos scores %s: %s", score_uri, exc)
        return None
    return _align_raw_to_world(
        raw, bag_id, chunk_id, row, n_world, filter_nonfinite, "scores"
    )


def _load_world(world_path_uri: str):
    """world NPZ → (xyz float64, intensity, ground_mask, sweep origin).

    intensity / ground_mask / origin are None when absent from the NPZ.
    `origin` is the sensor position in the world frame (written by deskew),
    needed for the near-ego dynamic gate.
    """
    data = np.load(local_path(world_path_uri))
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1)
    intensity = data["intensity"].astype(np.float32) if "intensity" in data else None
    ground_mask = data["ground_mask"] if "ground_mask" in data else None
    origin = np.asarray(data["origin"], dtype=np.float64) if "origin" in data else None
    return xyz, intensity, ground_mask, origin


def near_ego_mask(
    xyz: np.ndarray, sweep_origin: np.ndarray, min_range_m: float
) -> np.ndarray:
    """True for points within min_range_m horizontal range of the sensor.

    The same gate classify pass 2 applies (cfg.dynamic_min_range_m): within
    this radius the returns are ego self-returns / near clutter, so a moving
    verdict is unreliable regardless of which method produced it.
    """
    d_xy = np.hypot(xyz[:, 0] - sweep_origin[0], xyz[:, 1] - sweep_origin[1])
    return d_xy < min_range_m


def _write_empty_outputs(
    bag_id: str, chunk_id: str, voxel_size: float
) -> MosSegmentResult:
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
    return MosSegmentResult(0, 0, out_uri)


def _invalid_meta_row(row: dict, sweep_id: int) -> dict:
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


def classify_chunk(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
) -> MosSegmentResult:
    """Decompose a chunk into static/dynamic clouds from MF-MOS masks alone.

    Requires mf_mos._core.process_chunk to have run first (it writes the
    per-sweep masks this reads). Writes the same artifacts classify/ does so
    Step C ground and Step D reduce are method-agnostic.
    """
    meta_rows = read_rows(lidar_proc_index_path(bag_id, chunk_id))
    if not meta_rows:
        log.warning("chunk %s: no processed sweeps — writing empty maps", chunk_id)
        return _write_empty_outputs(bag_id, chunk_id, cfg.voxel_size_m)

    origin = _origin_from_index(meta_rows)
    if origin is None:
        return _write_empty_outputs(bag_id, chunk_id, cfg.voxel_size_m)

    voxel_size = cfg.voxel_size_m
    any_intensity = any(bool(r.get("has_intensity", False)) for r in meta_rows)

    static_xyz_chunks: list[np.ndarray] = []
    static_intensity_chunks: list[np.ndarray] = []
    dyn_xyz_chunks: list[np.ndarray] = []
    dyn_intensity_chunks: list[np.ndarray] = []
    dyn_sweep_id_chunks: list[np.ndarray] = []
    static_key_parts: list[np.ndarray] = []
    occ_key_parts: list[np.ndarray] = []
    frame_keys: dict[int, list[np.ndarray]] = {}

    total_static = 0
    total_dynamic = 0
    n_no_mask = 0
    updated_meta: list[dict] = []

    for row in tqdm(meta_rows, desc=f"mos chunk {chunk_id}", unit="sweep"):
        sweep_id = int(row["sweep_id"])
        if row.get("valid") is False:
            updated_meta.append(_invalid_meta_row(row, sweep_id))
            continue

        xyz, intensity, ground_mask, sweep_origin = _load_world(row["world_path"])
        n = xyz.shape[0]
        dyn_uri = dynamic_mask_path(bag_id, chunk_id, sweep_id)
        has_intensity = bool(row.get("has_intensity", False))

        if n == 0:
            np.save(local_path(dyn_uri), np.zeros(0, dtype=bool))
            updated_meta.append(
                _meta_row(row, sweep_id, 0, 0, 0, dyn_uri, has_intensity)
            )
            continue

        mf_mask = load_mf_mos_world_mask(
            bag_id, chunk_id, row, n, cfg.filter_nonfinite_points
        )
        if mf_mask is None:
            # No usable MF-MOS verdict for this sweep → treat as all-static
            # (never fabricate dynamics). Patchwork++ ground still excluded.
            n_no_mask += 1
            mf_mask = np.zeros(n, dtype=bool)

        moving = mf_mask.astype(bool)
        if ground_mask is not None:
            not_ground = ~ground_mask.astype(bool)
        else:
            not_ground = np.ones(n, dtype=bool)

        dyn_mask = moving & not_ground
        # Near-ego dynamic gate, mirroring classify pass 2: within
        # dynamic_min_range_m the returns are ego self-returns / near clutter,
        # so a moving verdict is unreliable. Gated points drop from the
        # dynamic cloud without being promoted to static.
        if (
            cfg.dynamic_min_range_m > 0.0
            and sweep_origin is not None
            and dyn_mask.any()
        ):
            dyn_mask &= ~near_ego_mask(xyz, sweep_origin, cfg.dynamic_min_range_m)
        static_mask = (~moving) & not_ground
        np.save(local_path(dyn_uri), dyn_mask)

        n_static = int(static_mask.sum())
        n_dyn = int(dyn_mask.sum())
        total_static += n_static
        total_dynamic += n_dyn

        keys = voxel_indices(xyz, origin, voxel_size, chunk_id=chunk_id)
        occ_key_parts.append(np.unique(keys))
        fid = row.get("frame_id")
        if fid is not None:
            frame_keys.setdefault(int(fid), []).append(keys)

        if n_static > 0:
            static_xyz_chunks.append(xyz[static_mask])
            static_key_parts.append(np.unique(keys[static_mask]))
            if any_intensity:
                static_intensity_chunks.append(
                    intensity[static_mask].astype(np.float32)
                    if (has_intensity and intensity is not None)
                    else np.zeros(n_static, dtype=np.float32)
                )
        if n_dyn > 0:
            dyn_xyz_chunks.append(xyz[dyn_mask])
            dyn_sweep_id_chunks.append(np.full(n_dyn, sweep_id, dtype=np.int32))
            if any_intensity:
                dyn_intensity_chunks.append(
                    intensity[dyn_mask].astype(np.float32)
                    if (has_intensity and intensity is not None)
                    else np.zeros(n_dyn, dtype=np.float32)
                )

        updated_meta.append(
            _meta_row(row, sweep_id, n, n_static, n_dyn, dyn_uri, has_intensity)
        )

    write_table(
        updated_meta, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id)
    )

    if static_key_parts:
        static_voxel_keys = np.unique(np.concatenate(static_key_parts))
    else:
        static_voxel_keys = np.empty(0, dtype=np.int64)

    out_uri = static_map_path(bag_id, chunk_id)
    static_kwargs: dict[str, np.ndarray] = {
        "voxel_size": np.float32(voxel_size),
        "origin": origin,
        "static_voxel_keys": static_voxel_keys,
    }
    if static_xyz_chunks:
        static_kwargs["xyz"] = np.concatenate(static_xyz_chunks, axis=0)
        if any_intensity and static_intensity_chunks:
            static_kwargs["intensity"] = np.concatenate(static_intensity_chunks)
    else:
        static_kwargs["xyz"] = np.empty((0, 3), dtype=np.float64)
    np.savez_compressed(local_path(out_uri), **static_kwargs)

    dyn_kwargs: dict[str, np.ndarray] = {}
    if dyn_xyz_chunks:
        dyn_kwargs["xyz"] = np.concatenate(dyn_xyz_chunks, axis=0)
        dyn_kwargs["sweep_id"] = np.concatenate(dyn_sweep_id_chunks)
        if any_intensity and dyn_intensity_chunks:
            dyn_kwargs["intensity"] = np.concatenate(dyn_intensity_chunks)
    else:
        dyn_kwargs["xyz"] = np.empty((0, 3), dtype=np.float64)
        dyn_kwargs["sweep_id"] = np.empty(0, dtype=np.int32)
    np.savez_compressed(local_path(dynamic_map_path(bag_id, chunk_id)), **dyn_kwargs)

    if cfg.save_voxel_occupancy and occ_key_parts:
        occ_keys = np.unique(np.concatenate(occ_key_parts))
        np.savez_compressed(
            local_path(voxel_occupancy_path(bag_id, chunk_id)),
            coords=_coords_from_keys(occ_keys),
            origin=origin,
            voxel_size=np.float32(voxel_size),
        )

    if cfg.save_per_frame_voxel_occupancy and frame_keys:
        for fid, key_list in frame_keys.items():
            merged = np.unique(np.concatenate(key_list))
            if merged.size == 0:
                continue
            np.savez_compressed(
                local_path(voxel_occupancy_frame_path(bag_id, chunk_id, fid)),
                coords=_coords_from_keys(merged),
                origin=origin,
                voxel_size=np.float32(voxel_size),
            )

    log.info(
        "chunk %s: mos segmentation static=%d dynamic=%d (no-mask sweeps left static=%d)",
        chunk_id,
        total_static,
        total_dynamic,
        n_no_mask,
    )
    return MosSegmentResult(
        n_static=total_static,
        n_dynamic=total_dynamic,
        static_map_path=out_uri,
        n_sweeps_no_mask=n_no_mask,
    )


def _meta_row(
    row: dict,
    sweep_id: int,
    n_points_total: int,
    n_static: int,
    n_dynamic: int,
    dyn_uri: str,
    has_intensity: bool,
) -> dict:
    return ProcessedSweepMeta(
        bag_id=row["bag_id"],
        chunk_id=row["chunk_id"],
        sweep_id=sweep_id,
        lidar_id=row["lidar_id"],
        reference_timestamp_ns=int(row["reference_timestamp_ns"]),
        n_points_total=n_points_total,
        n_points_static=n_static,
        n_points_dynamic=n_dynamic,
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
        mf_mos_mask_path=row.get("mf_mos_mask_path"),
    ).model_dump()
