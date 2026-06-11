"""Fusion static/dynamic decomposition (`--seg union`).

The third Step-B method. `aw` (classify/) and `mos` (mf_mos/) are mutually
exclusive and never import each other; `union` is the fusion layer, so it
deliberately reads the outputs of both and reuses mf_mos's mask-alignment +
meta-row helpers.

Division of labour:
  * static cloud  = Amanatides-Woo's static map (classify.process_chunk),
    kept verbatim. High precision — a voxel is static only with a
    preponderance of occupied evidence.
  * dynamic cloud = MF-MOS moving points (mf_mos._core.process_chunk), with
      - Patchwork++ ground removed (ground belongs to ground.npz), and
      - every point whose voxel AW confirmed static removed
        (cfg.union.aw_static_veto). The AW static map is the "comparison"
        that rejects MF-MOS false positives hugging static structure.

      dynamic = mf_mos_moving & ~ground & ~aw_static            (default)
      dynamic = (mf_mos_moving | aw_dynamic) & ~ground & ~aw_static
                                                      (keep_aw_dynamic=True)

Prerequisites — the orchestrator runs these first, in this order, so their
artifacts are on disk before classify_chunk reads them:
  1. classify.process_chunk     → static_map.npz (+ static_voxel_keys), and
     the per-sweep AW dynamic_mask.npy this step reads when keep_aw_dynamic.
  2. mf_mos._core.process_chunk → per-sweep *_mf_mos_mask.npy.

This step rewrites ONLY the dynamic side — dynamic_map.npz, the per-sweep
dynamic_mask.npy, and each index row's n_points_dynamic. static_map.npz and
the AW n_points_static counts are left untouched, so Step C ground / Step D
reduce stay method-agnostic exactly as for aw and mos.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from wato_common.artifact_store import (
    dynamic_map_path,
    dynamic_mask_path,
    lidar_proc_index_path,
    local_path,
    static_map_path,
)
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import PROCESSED_SWEEPS_SCHEMA
from wato_lidar_preprocessing.config import ComponentConfig

# union is the fusion layer: reusing mf_mos's mask-alignment + meta-row
# helpers keeps the on-disk schema single-sourced rather than re-deriving it.
from wato_lidar_preprocessing.mf_mos.segment import (
    _invalid_meta_row,
    _load_world,
    _meta_row,
    load_mf_mos_world_mask,
)
from wato_lidar_preprocessing.voxel import voxel_indices

log = logging.getLogger(__name__)


@dataclass
class UnionSegmentResult:
    """Result of the fusion decomposition.

    Field names mirror classify.ClassifyResult / mf_mos.MosSegmentResult
    (including the cache fields, populated by the orchestrator from the AW
    half) so _write_chunk_summary can consume any of the three unchanged.
    """

    n_static: int
    n_dynamic: int
    static_map_path: str
    n_sweeps_no_mask: int = 0
    n_vetoed: int = 0
    cache_auto_disabled: bool = False
    estimated_cache_bytes: int = 0


def _keys_in_sorted(keys: np.ndarray, sorted_keys: np.ndarray) -> np.ndarray:
    """Boolean mask: which `keys` are present in sorted-unique `sorted_keys`.

    Mirrors classify/masking.py's searchsorted membership test so the voxel
    lookup is identical to the one that built static_voxel_keys.
    """
    if sorted_keys.size == 0 or keys.size == 0:
        return np.zeros(keys.shape[0], dtype=bool)
    pos = np.searchsorted(sorted_keys, keys)
    pos = np.clip(pos, 0, sorted_keys.size - 1)
    return sorted_keys[pos] == keys


def classify_chunk(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
) -> UnionSegmentResult:
    """Fuse AW static + MF-MOS dynamic for one chunk.

    Requires classify.process_chunk and mf_mos._core.process_chunk to have run
    first (this reads static_map.npz and the per-sweep MF-MOS masks). Rewrites
    only the dynamic side; static_map.npz is kept verbatim.
    """
    meta_rows = read_rows(lidar_proc_index_path(bag_id, chunk_id))
    sm_uri = static_map_path(bag_id, chunk_id)
    if not meta_rows:
        log.warning("chunk %s: no processed sweeps — union is a no-op", chunk_id)
        return UnionSegmentResult(0, 0, sm_uri)

    # AW static map is the veto basis AND the canonical origin/voxel_size, so
    # the keys we pack here line up bit-for-bit with static_voxel_keys.
    sm = np.load(local_path(sm_uri))
    aw_static_keys = np.sort(np.asarray(sm["static_voxel_keys"], dtype=np.int64))
    origin = np.asarray(sm["origin"], dtype=np.float64)
    voxel_size = float(sm["voxel_size"])
    veto = bool(cfg.union.aw_static_veto) and aw_static_keys.size > 0

    any_intensity = any(bool(r.get("has_intensity", False)) for r in meta_rows)

    dyn_xyz_chunks: list[np.ndarray] = []
    dyn_intensity_chunks: list[np.ndarray] = []
    dyn_sweep_id_chunks: list[np.ndarray] = []
    total_static = 0
    total_dynamic = 0
    total_vetoed = 0
    n_no_mask = 0
    updated_meta: list[dict] = []

    for row in tqdm(meta_rows, desc=f"union chunk {chunk_id}", unit="sweep"):
        sweep_id = int(row["sweep_id"])
        if row.get("valid") is False:
            updated_meta.append(_invalid_meta_row(row, sweep_id))
            continue

        xyz, intensity, ground_mask = _load_world(row["world_path"])
        n = xyz.shape[0]
        dyn_uri = dynamic_mask_path(bag_id, chunk_id, sweep_id)
        has_intensity = bool(row.get("has_intensity", False))
        # union keeps AW's static cloud verbatim, so preserve its per-sweep
        # static count (set by classify Pass 2) rather than recomputing it.
        n_static = int(row.get("n_points_static") or 0)
        total_static += n_static

        if n == 0:
            np.save(local_path(dyn_uri), np.zeros(0, dtype=bool))
            updated_meta.append(
                _meta_row(row, sweep_id, 0, n_static, 0, dyn_uri, has_intensity)
            )
            continue

        # keep_aw_dynamic unions in AW's own verdict — read it BEFORE the
        # np.save below overwrites this same dynamic_mask.npy.
        aw_dyn: np.ndarray | None = None
        if cfg.union.keep_aw_dynamic:
            try:
                loaded = np.load(local_path(dyn_uri)).astype(bool)
                aw_dyn = loaded if loaded.shape[0] == n else None
            except (FileNotFoundError, ValueError) as exc:
                log.warning(
                    "sweep %d: could not read AW dynamic mask for keep_aw_dynamic "
                    "(%s); using MF-MOS only this sweep",
                    sweep_id,
                    exc,
                )

        mf_mask = load_mf_mos_world_mask(
            bag_id, chunk_id, row, n, cfg.filter_nonfinite_points
        )
        if mf_mask is None:
            # No usable MF-MOS verdict → fabricate no dynamics (mirrors
            # mf_mos.segment). keep_aw_dynamic can still admit AW's verdict.
            n_no_mask += 1
            moving = np.zeros(n, dtype=bool)
        else:
            moving = mf_mask.astype(bool)

        candidate = moving | aw_dyn if aw_dyn is not None else moving

        not_ground = (
            ~ground_mask.astype(bool)
            if ground_mask is not None
            else np.ones(n, dtype=bool)
        )
        dyn_mask = candidate & not_ground

        if veto:
            keys = voxel_indices(xyz, origin, voxel_size, chunk_id=chunk_id)
            on_aw_static = _keys_in_sorted(keys, aw_static_keys)
            n_pre = int(dyn_mask.sum())
            dyn_mask &= ~on_aw_static
            total_vetoed += n_pre - int(dyn_mask.sum())

        np.save(local_path(dyn_uri), dyn_mask)
        n_dyn = int(dyn_mask.sum())
        total_dynamic += n_dyn

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

    # static_map.npz is intentionally NOT rewritten — AW owns the static cloud.
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

    log.info(
        "chunk %s: union static=%d (AW) dynamic=%d (MF-MOS%s; %d points vetoed by "
        "AW-static; %d sweeps had no MF-MOS mask)",
        chunk_id,
        total_static,
        total_dynamic,
        "+AW-dyn" if cfg.union.keep_aw_dynamic else "",
        total_vetoed,
        n_no_mask,
    )
    return UnionSegmentResult(
        n_static=total_static,
        n_dynamic=total_dynamic,
        static_map_path=sm_uri,
        n_sweeps_no_mask=n_no_mask,
        n_vetoed=total_vetoed,
    )
