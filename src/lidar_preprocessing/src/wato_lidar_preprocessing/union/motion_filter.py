"""Post-fusion temporal motion filter for the union dynamic cloud.

Runs AFTER union.classify_chunk's AW-static / ground vetoes, on the accumulated
per-chunk dynamic cloud. The voxel vetoes only reach MF-MOS false positives
that land on the AW static map; structure the static map covers sparsely (far
walls, foliage, below-grade returns) slips through. Two temporal gates catch
it, both exploiting the offline batch setting and targeting *currently-moving*
semantics:

  persistence gate — drop a point whose voxel (persistence_voxel_m) is occupied
                     across >= persistence_max_sweeps DISTINCT sweeps. A moving
                     object sweeps through a 0.5 m voxel in 1-2 sweeps; static
                     structure dwells in the same voxel the whole time it's in
                     view.
  coherence  gate  — drop a point whose per-sweep cluster does not link into a
                     track spanning >= coherence_min_life sweeps. Removes the
                     temporally-incoherent specks that survive persistence.

Both were chosen by measurement (scripts/compare_seg_dynamic): on real Velodyne
data the pair cuts on-static leakage 58% -> ~4%. A velocity gate and an
MF-MOS-AND-AW-dynamic intersection both tested WORSE — per-sweep visibility
drifts a connected-component centroid, faking velocity on static structure — so
neither is implemented. The coherence gate is membership-only for that reason;
the per-sweep size cap (coherence_max_object_m) is the discriminator the
velocity gate lacked.

This module operates on the artifacts union.classify_chunk just wrote — the
accumulated dynamic_map.npz plus each sweep's dynamic_mask.npy — and rewrites
both, plus each index row's n_points_dynamic. It relies on union's write order:
within one sweep the dynamic_map block is in world-point order, identical to the
True positions of that sweep's dynamic_mask, so the keep mask maps back
position-for-position without re-reading the world NPZs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from wato_common.artifact_store import (
    dynamic_map_path,
    dynamic_mask_path,
    lidar_proc_index_path,
    local_path,
)
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import PROCESSED_SWEEPS_SCHEMA
from wato_lidar_preprocessing.config import ComponentConfig, MotionFilterParams
from wato_lidar_preprocessing.mf_mos.segment import _invalid_meta_row, _meta_row
from wato_lidar_preprocessing.voxel import voxel_indices

log = logging.getLogger(__name__)


@dataclass
class MotionFilterResult:
    """Outcome of one chunk's motion filter, threaded into the chunk summary."""

    applied: bool
    n_before: int
    n_after: int
    n_persistence_dropped: int
    n_coherence_dropped: int


def persistence_keep(
    xyz: np.ndarray, sweep_id: np.ndarray, voxel_m: float, max_sweeps: int
) -> np.ndarray:
    """True where a point's voxel is occupied in < max_sweeps distinct sweeps.

    A genuinely-moving point passes through a voxel in 1-2 sweeps; structure
    persists. max_sweeps <= 0 disables the gate (all True).
    """
    n = xyz.shape[0]
    if max_sweeps <= 0 or n == 0:
        return np.ones(n, dtype=bool)
    # Origin at the cloud's min corner keeps every voxel index >= 0 and small,
    # well inside the shared packing's bit budget regardless of map-frame offset.
    origin = np.floor(xyz.min(axis=0))
    keys = voxel_indices(xyz, origin, voxel_m)
    # distinct (voxel, sweep) pairs; the count of a key among the unique pairs
    # is exactly the number of distinct sweeps that voxel was occupied in.
    pairs = np.unique(np.stack([keys, sweep_id.astype(np.int64)], axis=1), axis=0)
    ukeys, sweeps_per_key = np.unique(pairs[:, 0], return_counts=True)
    per_point = sweeps_per_key[np.searchsorted(ukeys, keys)]
    return per_point < max_sweeps


def _cluster_sweep_2d(
    xy: np.ndarray, cell: float, max_object_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Connected-components over a rasterized 2D grid for one sweep.

    Returns (centroids[K,2], point_label[N]) where point_label is the 1-based
    cluster id per point, or 0 for points in a cluster whose extent exceeds
    max_object_m (treated as static structure, never tracked).
    """
    n = xy.shape[0]
    if n == 0:
        return np.empty((0, 2)), np.zeros(0, dtype=np.int64)
    mn = xy.min(axis=0)
    ij = np.floor((xy - mn) / cell).astype(np.int64)
    h = int(ij[:, 1].max()) + 1
    w = int(ij[:, 0].max()) + 1
    grid = np.zeros((h, w), dtype=bool)
    grid[ij[:, 1], ij[:, 0]] = True
    labels, n_lab = ndimage.label(grid, structure=np.ones((3, 3)))
    plab = labels[ij[:, 1], ij[:, 0]]
    centroids = np.empty((n_lab, 2))
    out_label = plab.copy()
    for lab in range(1, n_lab + 1):
        m = plab == lab
        pts = xy[m]
        centroids[lab - 1] = pts.mean(axis=0)
        if (pts.max(axis=0) - pts.min(axis=0)).max() > max_object_m:
            out_label[m] = 0  # too big to be an object — drop from tracking
    return centroids, out_label


def coherence_keep(
    xyz: np.ndarray,
    sweep_id: np.ndarray,
    cell_m: float,
    link_gate_m: float,
    min_life: int,
    max_object_m: float,
) -> np.ndarray:
    """True where a point belongs to a cluster-track spanning >= min_life sweeps.

    Per-sweep connected-component clusters are greedily linked across
    consecutive sweeps by nearest centroid within link_gate_m, unioned into
    tracks, and a track is kept only if it spans >= min_life distinct sweeps.
    Membership only — never a velocity test. min_life <= 0 disables (all True).
    """
    n = xyz.shape[0]
    if min_life <= 0 or n == 0:
        return np.ones(n, dtype=bool)

    sweeps = np.unique(sweep_id)
    gidx = np.arange(n)
    # node = one per-sweep cluster: (sweep, centroid_xy, member global indices)
    node_sweep: list[int] = []
    node_centroid: list[np.ndarray] = []
    node_members: list[np.ndarray] = []
    nodes_by_sweep: dict[int, list[int]] = {}
    for s in sweeps:
        sel = sweep_id == s
        idx = gidx[sel]
        centroids, plab = _cluster_sweep_2d(xyz[sel, :2], cell_m, max_object_m)
        for lab in range(1, centroids.shape[0] + 1):
            m = plab == lab
            if not m.any():  # whole cluster dropped by the size cap
                continue
            ni = len(node_sweep)
            node_sweep.append(int(s))
            node_centroid.append(centroids[lab - 1])
            node_members.append(idx[m])
            nodes_by_sweep.setdefault(int(s), []).append(ni)

    # union-find over linked nodes
    parent = list(range(len(node_sweep)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for k in range(1, len(sweeps)):
        prev = nodes_by_sweep.get(int(sweeps[k - 1]), [])
        cur = nodes_by_sweep.get(int(sweeps[k]), [])
        if not prev or not cur:
            continue
        prev_xy = np.array([node_centroid[i] for i in prev])
        cur_xy = np.array([node_centroid[i] for i in cur])
        dist, j = cKDTree(prev_xy).query(cur_xy, k=1, distance_upper_bound=link_gate_m)
        for ci, (dd, jj) in enumerate(zip(dist, j)):
            if np.isfinite(dd):
                parent[find(cur[ci])] = find(prev[jj])

    track_nodes: dict[int, list[int]] = {}
    for ni in range(len(node_sweep)):
        track_nodes.setdefault(find(ni), []).append(ni)

    keep = np.zeros(n, dtype=bool)
    for nis in track_nodes.values():
        life = len({node_sweep[i] for i in nis})
        if life < min_life:
            continue
        for ni in nis:
            keep[node_members[ni]] = True
    return keep


def compute_keep(
    mf: MotionFilterParams, xyz: np.ndarray, sweep_id: np.ndarray
) -> tuple[np.ndarray, int, int]:
    """Combined persistence+coherence keep mask plus per-gate drop counts.

    Coherence is computed on the full cloud (matching the validated A/B), then
    AND'd with persistence so the reported coherence drops are the points the
    coherence gate removes *beyond* what persistence already did.
    """
    persist = persistence_keep(
        xyz, sweep_id, mf.persistence_voxel_m, mf.persistence_max_sweeps
    )
    coh = coherence_keep(
        xyz,
        sweep_id,
        mf.coherence_cell_m,
        mf.coherence_link_gate_m,
        mf.coherence_min_life,
        mf.coherence_max_object_m,
    )
    keep = persist & coh
    n_persistence_dropped = int((~persist).sum())
    n_coherence_dropped = int((persist & ~coh).sum())
    return keep, n_persistence_dropped, n_coherence_dropped


def filter_dynamic_artifacts(
    cfg: ComponentConfig, bag_id: str, chunk_id: str
) -> MotionFilterResult:
    """Apply the motion filter to a chunk's already-written union artifacts.

    Reads dynamic_map.npz + each per-sweep dynamic_mask.npy, computes the keep
    mask, and rewrites dynamic_map.npz, the per-sweep masks, and each index
    row's n_points_dynamic. A no-op (returns applied=False) when the filter is
    disabled or both gates are off. static_map.npz is never touched.
    """
    mf = cfg.union.motion_filter
    both_gates_off = mf.persistence_max_sweeps <= 0 and mf.coherence_min_life <= 0
    if not mf.enabled or both_gates_off:
        return MotionFilterResult(False, 0, 0, 0, 0)

    dm_uri = dynamic_map_path(bag_id, chunk_id)
    dm = dict(np.load(local_path(dm_uri)))
    xyz = dm["xyz"]
    sweep_id = dm["sweep_id"]
    n_before = int(xyz.shape[0])
    if n_before == 0:
        return MotionFilterResult(True, 0, 0, 0, 0)

    keep, n_persistence_dropped, n_coherence_dropped = compute_keep(mf, xyz, sweep_id)

    # Map the keep mask back onto each sweep's dynamic_mask. Within a sweep the
    # dynamic_map block is in world order == the mask's True positions, so
    # keep[sweep_id == s] aligns to flatnonzero(mask) position-for-position.
    meta_rows = read_rows(lidar_proc_index_path(bag_id, chunk_id))
    updated_meta: list[dict] = []
    for row in meta_rows:
        sweep = int(row["sweep_id"])
        if row.get("valid") is False:
            updated_meta.append(_invalid_meta_row(row, sweep))
            continue
        n_total = int(row.get("n_points_total") or 0)
        n_static = int(row.get("n_points_static") or 0)
        has_intensity = bool(row.get("has_intensity", False))
        dyn_uri = dynamic_mask_path(bag_id, chunk_id, sweep)
        sel = sweep_id == sweep
        if not sel.any():
            # Sweep contributed no dynamics; leave its (all-False) mask as is.
            updated_meta.append(
                _meta_row(row, sweep, n_total, n_static, 0, dyn_uri, has_intensity)
            )
            continue
        keep_s = keep[sel]
        mask = np.load(local_path(dyn_uri)).astype(bool)
        true_pos = np.flatnonzero(mask)
        if true_pos.shape[0] != keep_s.shape[0]:
            # Invariant broken (mask out of sync with dynamic_map) — leave this
            # sweep untouched in BOTH artifacts so they stay consistent.
            log.warning(
                "chunk %s sweep %d: dynamic_mask True-count %d != dynamic_map "
                "block %d — skipping motion filter for this sweep",
                chunk_id,
                sweep,
                true_pos.shape[0],
                keep_s.shape[0],
            )
            keep[sel] = True
            n_dyn = int(mask.sum())
            updated_meta.append(
                _meta_row(row, sweep, n_total, n_static, n_dyn, dyn_uri, has_intensity)
            )
            continue
        mask[true_pos[~keep_s]] = False
        np.save(local_path(dyn_uri), mask)
        n_dyn = int(mask.sum())
        updated_meta.append(
            _meta_row(row, sweep, n_total, n_static, n_dyn, dyn_uri, has_intensity)
        )

    # Rewrite the accumulated dynamic cloud (keep may have been relaxed above).
    new_dm: dict[str, np.ndarray] = {"xyz": xyz[keep], "sweep_id": sweep_id[keep]}
    if "intensity" in dm:
        new_dm["intensity"] = dm["intensity"][keep]
    np.savez_compressed(local_path(dm_uri), **new_dm)
    write_table(
        updated_meta, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id)
    )

    n_after = int(keep.sum())
    log.info(
        "chunk %s: motion filter %d -> %d dynamic pts (persistence dropped %d, "
        "coherence dropped %d)",
        chunk_id,
        n_before,
        n_after,
        n_persistence_dropped,
        n_coherence_dropped,
    )
    return MotionFilterResult(
        True, n_before, n_after, n_persistence_dropped, n_coherence_dropped
    )
