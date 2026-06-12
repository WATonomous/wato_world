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
      - Patchwork++ ground removed (ground belongs to ground.npz),
      - the near-ego gate applied (cfg.dynamic_min_range_m, same as classify
        pass 2 — AW can't veto ego self-returns because near-ego voxels are
        maximally carved and never enter the static set),
      - the ground-height veto applied (cfg.union.ground_height_veto_m):
        points below that height over Step C's ground.npz height grid drop.
        The AW static set is structurally blind to the road — skip_endpoint
        keeps road voxels at n_hits==0, so they are never static — which
        lets MF-MOS road false positives through every voxel veto. Height
        over the aggregated ground grid is the signal that catches them
        (including below-grade artifacts), and
      - every point whose voxel AW confirmed static — or, with
        veto_dilation_voxels >= 1, within that many voxels of one — removed
        (cfg.union.aw_static_veto). The AW static map is the "comparison"
        that rejects MF-MOS false positives hugging static structure.
        Surface returns straddle voxel boundaries, so the exact-voxel test
        alone misses the shell one voxel off the structure; the dilation
        catches it. Two exemptions keep real movers alive:
          * candidates whose own voxel AW classed dynamic (static_map.npz
            dynamic_voxel_keys) are exempt from the *dilated* part — AW
            corroborates the motion, and movers brush past structure all
            the time;
          * points whose MF-MOS moving probability is >= veto_score_exempt
            are exempt from the whole veto: AW's static evidence aggregates
            the whole chunk, so an object parked for most of the chunk that
            drives off gets voxel-static evidence — a high-confidence
            MF-MOS verdict is more trustworthy than chunk-aggregate
            occupancy there.

      dynamic = mf_mos_moving & ~ground & ~near_ego & ~near_ground
                & ~aw_static_dilated                              (default)
      dynamic = (mf_mos_moving | aw_dynamic) & ...   (keep_aw_dynamic=True)

Prerequisites — the orchestrator runs these first, in this order, so their
artifacts are on disk before classify_chunk reads them:
  1. classify.process_chunk     → static_map.npz (static_voxel_keys +
     dynamic_voxel_keys), and the per-sweep aw_dynamic_mask.npy snapshot
     this step reads when keep_aw_dynamic (written on the union path only).
  2. mf_mos._core.process_chunk → per-sweep *_mf_mos_mask.npy
     (+ *_mf_mos_score.npy when save_scores, used by veto_score_exempt).
  3. ground.process_chunk       → ground.npz height grid for the
     ground-height veto. The orchestrator runs Step C before union on the
     union path only; if it's missing the veto is skipped with a warning.

This step rewrites ONLY the dynamic side — dynamic_map.npz, the per-sweep
dynamic_mask.npy, and each index row's n_points_dynamic. static_map.npz and
the AW n_points_static counts are left untouched, so Step C ground / Step D
reduce stay method-agnostic exactly as for aw and mos.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from wato_common.artifact_store import (
    aw_dynamic_mask_path,
    dynamic_map_path,
    dynamic_mask_path,
    ground_path,
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
    load_mf_mos_world_scores,
    near_ego_mask,
)
from wato_lidar_preprocessing.voxel import AXIS_BITS, keys_in_sorted, voxel_indices

if TYPE_CHECKING:
    from wato_lidar_preprocessing.classify import ClassifyResult

log = logging.getLogger(__name__)

_AXIS_MASK = np.int64((1 << AXIS_BITS) - 1)


@dataclass
class UnionSegmentResult:
    """Result of the fusion decomposition.

    Field names mirror classify.ClassifyResult / mf_mos.MosSegmentResult
    (including the cache fields, carried over from the AW half) so
    _write_chunk_summary can consume any of the three unchanged.
    """

    n_static: int
    n_dynamic: int
    static_map_path: str
    n_sweeps_no_mask: int = 0
    n_vetoed: int = 0
    n_ground_vetoed: int = 0
    cache_auto_disabled: bool = False
    estimated_cache_bytes: int = 0


def _load_aw_dynamic_mask(
    bag_id: str, chunk_id: str, sweep_id: int, n: int
) -> np.ndarray | None:
    """AW's own per-sweep dynamic verdict for keep_aw_dynamic.

    Prefers the aw_dynamic_mask.npy snapshot (written by classify on the
    union path). Falls back to dynamic_mask.npy for artifacts that predate
    the snapshot — only correct when classify ran immediately before, since
    union overwrites that file with the fused verdict.
    """
    snap_uri = aw_dynamic_mask_path(bag_id, chunk_id, sweep_id)
    uri = snap_uri
    if not os.path.exists(local_path(snap_uri)):
        uri = dynamic_mask_path(bag_id, chunk_id, sweep_id)
        log.warning(
            "sweep %d: no aw_dynamic_mask.npy snapshot — falling back to "
            "dynamic_mask.npy (pre-snapshot artifacts; re-run classify to fix)",
            sweep_id,
        )
    try:
        loaded = np.load(local_path(uri)).astype(bool)
    except (FileNotFoundError, ValueError) as exc:
        log.warning(
            "sweep %d: could not read AW dynamic mask for keep_aw_dynamic "
            "(%s); using MF-MOS only this sweep",
            sweep_id,
            exc,
        )
        return None
    if loaded.shape[0] != n:
        log.warning(
            "sweep %d: AW dynamic mask len %d != world len %d; using MF-MOS "
            "only this sweep",
            sweep_id,
            loaded.shape[0],
            n,
        )
        return None
    return loaded


def _load_ground_grid(
    bag_id: str, chunk_id: str
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """ground.npz → (height_grid, grid_origin, cell_size) or None.

    None (with a warning) when ground.npz is absent — the orchestrator runs
    Step C before union, but a standalone re-fuse on artifacts that predate
    that ordering won't have it — or when it's a sentinel (Patchwork++
    unavailable upstream), in which case there is no height to veto against.
    """
    path = local_path(ground_path(bag_id, chunk_id))
    if not os.path.exists(path):
        log.warning(
            "chunk %s: ground.npz not found — ground-height veto skipped "
            "(Step C must run before union)",
            chunk_id,
        )
        return None
    g = np.load(path)
    status = str(g["status"]) if "status" in g.files else "ok"
    if status != "ok" or "height_grid" not in g.files:
        log.warning(
            "chunk %s: ground.npz status=%r — ground-height veto skipped",
            chunk_id,
            status,
        )
        return None
    return (
        g["height_grid"],
        np.asarray(g["grid_origin"], dtype=np.float64),
        float(g["cell_size"]),
    )


def _height_above_ground(
    xyz: np.ndarray, grid: tuple[np.ndarray, np.ndarray, float]
) -> np.ndarray:
    """Per-point height over the Step C ground grid.

    height_grid is (H, W) = (y, x) — see ground._core._build_height_grid.
    Points outside the grid clamp to the nearest edge cell; the grid spans
    the bbox of all ground returns, so real out-of-bounds points are rare.
    """
    hg, grid_origin, cell = grid
    ix = np.clip(
        ((xyz[:, 0] - grid_origin[0]) / cell).astype(np.int64), 0, hg.shape[1] - 1
    )
    iy = np.clip(
        ((xyz[:, 1] - grid_origin[1]) / cell).astype(np.int64), 0, hg.shape[0] - 1
    )
    return xyz[:, 2] - hg[iy, ix]


def _near_static_dilated(
    keys: np.ndarray, aw_static_keys: np.ndarray, dilation: int
) -> np.ndarray:
    """True where a voxel within Chebyshev distance `dilation` is AW-static.

    Excludes the centre voxel — the caller tests exact membership separately
    (it carries different exemption rules). Neighbour keys are formed by
    arithmetic on the packed key's axis fields; at the grid edge an index can
    underflow into the adjacent field, which yields a negative or ~260 km
    key that can never match a real static voxel, so no bounds mask needed.
    """
    near = np.zeros(keys.shape[0], dtype=bool)
    if aw_static_keys.size == 0 or keys.size == 0:
        return near
    vz = keys & _AXIS_MASK
    vy = (keys >> np.int64(AXIS_BITS)) & _AXIS_MASK
    vx = keys >> np.int64(2 * AXIS_BITS)
    offsets = range(-dilation, dilation + 1)
    for dx in offsets:
        for dy in offsets:
            for dz in offsets:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                kk = (
                    ((vx + dx) << np.int64(2 * AXIS_BITS))
                    | ((vy + dy) << np.int64(AXIS_BITS))
                    | (vz + dz)
                )
                near |= keys_in_sorted(kk, aw_static_keys)
    return near


def classify_chunk(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
    aw_result: "ClassifyResult | None" = None,
) -> UnionSegmentResult:
    """Fuse AW static + MF-MOS dynamic for one chunk.

    Requires classify.process_chunk, mf_mos._core.process_chunk and (for the
    ground-height veto) ground.process_chunk to have run first. Rewrites only
    the dynamic side; static_map.npz is kept verbatim.

    `aw_result` is the ClassifyResult of the AW half — its cache telemetry is
    carried into the returned UnionSegmentResult for the chunk summary.
    """
    cache_auto_disabled = aw_result.cache_auto_disabled if aw_result else False
    estimated_cache_bytes = aw_result.estimated_cache_bytes if aw_result else 0

    meta_rows = read_rows(lidar_proc_index_path(bag_id, chunk_id))
    sm_uri = static_map_path(bag_id, chunk_id)
    if not meta_rows:
        log.warning("chunk %s: no processed sweeps — union is a no-op", chunk_id)
        return UnionSegmentResult(
            0,
            0,
            sm_uri,
            cache_auto_disabled=cache_auto_disabled,
            estimated_cache_bytes=estimated_cache_bytes,
        )

    # AW static map is the veto basis AND the canonical origin/voxel_size, so
    # the keys we pack here line up bit-for-bit with static_voxel_keys.
    sm = np.load(local_path(sm_uri))
    aw_static_keys = np.sort(np.asarray(sm["static_voxel_keys"], dtype=np.int64))
    origin = np.asarray(sm["origin"], dtype=np.float64)
    voxel_size = float(sm["voxel_size"])
    veto = bool(cfg.union.aw_static_veto) and aw_static_keys.size > 0
    score_exempt = cfg.union.veto_score_exempt
    dilation = int(cfg.union.veto_dilation_voxels)
    if "dynamic_voxel_keys" in sm.files:
        aw_dynamic_keys = np.sort(np.asarray(sm["dynamic_voxel_keys"], dtype=np.int64))
    else:
        aw_dynamic_keys = np.empty(0, dtype=np.int64)
        if veto and dilation > 0:
            log.info(
                "chunk %s: static_map.npz has no dynamic_voxel_keys (pre-export "
                "artifacts or persistence mode) — dilated veto runs without the "
                "AW-dynamic exemption",
                chunk_id,
            )

    min_height = float(cfg.union.ground_height_veto_m)
    ground_grid = _load_ground_grid(bag_id, chunk_id) if min_height > 0.0 else None

    if not cfg.union.aw_static_veto:
        log.warning(
            "chunk %s: aw_static_veto=False — static_map and dynamic_map can "
            "overlap, so n_points_static + n_points_dynamic + n_points_ground "
            "may exceed n_points_total in the chunk summary (A/B mode only)",
            chunk_id,
        )

    any_intensity = any(bool(r.get("has_intensity", False)) for r in meta_rows)

    dyn_xyz_chunks: list[np.ndarray] = []
    dyn_intensity_chunks: list[np.ndarray] = []
    dyn_sweep_id_chunks: list[np.ndarray] = []
    total_static = 0
    total_dynamic = 0
    total_vetoed = 0
    total_ground_vetoed = 0
    n_no_mask = 0
    updated_meta: list[dict] = []

    for row in tqdm(meta_rows, desc=f"union chunk {chunk_id}", unit="sweep"):
        sweep_id = int(row["sweep_id"])
        if row.get("valid") is False:
            updated_meta.append(_invalid_meta_row(row, sweep_id))
            continue

        xyz, intensity, ground_mask, sweep_origin = _load_world(row["world_path"])
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

        aw_dyn: np.ndarray | None = None
        if cfg.union.keep_aw_dynamic:
            aw_dyn = _load_aw_dynamic_mask(bag_id, chunk_id, sweep_id, n)

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

        # Near-ego dynamic gate, mirroring classify pass 2. Must happen here:
        # the AW veto can't catch ego self-returns (near-ego voxels are
        # maximally carved, never static), and it also shrinks the candidate
        # set the vetoes below operate on.
        if (
            cfg.dynamic_min_range_m > 0.0
            and sweep_origin is not None
            and dyn_mask.any()
        ):
            dyn_mask &= ~near_ego_mask(xyz, sweep_origin, cfg.dynamic_min_range_m)

        # Ground-height veto: drop candidates skimming (or below) the Step C
        # ground surface — MF-MOS road false positives that no voxel veto can
        # reach because road voxels are never AW-static.
        if ground_grid is not None and dyn_mask.any():
            idx = np.flatnonzero(dyn_mask)
            low = _height_above_ground(xyz[idx], ground_grid) < min_height
            dyn_mask[idx[low]] = False
            total_ground_vetoed += int(low.sum())

        if veto and dyn_mask.any():
            # Voxelize only the surviving dynamic candidates (typically a few
            # percent of the sweep), not the whole cloud.
            cand_idx = np.flatnonzero(dyn_mask)
            keys = voxel_indices(xyz[cand_idx], origin, voxel_size, chunk_id=chunk_id)
            vetoed = keys_in_sorted(keys, aw_static_keys)
            if dilation > 0:
                # The dilated shell is exempt for candidates whose own voxel
                # AW classed dynamic — AW corroborates the motion there, and
                # movers brushing past structure must not be shredded.
                near = _near_static_dilated(keys, aw_static_keys, dilation)
                vetoed |= near & ~keys_in_sorted(keys, aw_dynamic_keys)
            if score_exempt is not None and vetoed.any():
                scores = load_mf_mos_world_scores(
                    bag_id, chunk_id, row, n, cfg.filter_nonfinite_points
                )
                if scores is not None:
                    # High-confidence MF-MOS movers survive the veto (handles
                    # parked-then-moving objects whose voxels are chunk-static).
                    vetoed &= scores[cand_idx] < score_exempt
            dyn_mask[cand_idx[vetoed]] = False
            total_vetoed += int(vetoed.sum())

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
        "chunk %s: union static=%d (AW) dynamic=%d (MF-MOS%s; %d vetoed by "
        "AW-static[D=%d]; %d vetoed by ground-height<%.2fm; %d sweeps had no "
        "MF-MOS mask)",
        chunk_id,
        total_static,
        total_dynamic,
        "+AW-dyn" if cfg.union.keep_aw_dynamic else "",
        total_vetoed,
        dilation,
        total_ground_vetoed,
        min_height if ground_grid is not None else 0.0,
        n_no_mask,
    )
    return UnionSegmentResult(
        n_static=total_static,
        n_dynamic=total_dynamic,
        static_map_path=sm_uri,
        n_sweeps_no_mask=n_no_mask,
        n_vetoed=total_vetoed,
        n_ground_vetoed=total_ground_vetoed,
        cache_auto_disabled=cache_auto_disabled,
        estimated_cache_bytes=estimated_cache_bytes,
    )
