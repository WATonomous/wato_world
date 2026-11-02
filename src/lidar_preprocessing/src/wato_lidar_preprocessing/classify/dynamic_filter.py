"""Step B — point-level cleanup of the voxel verdict's dynamic points.

The voxel verdict sees a chunk's total evidence per voxel, so what still reads
dynamic is thin: isolated returns (mixed pixels, foliage speckle) and structure
the rays keep passing through. Two tests catch those, in memory, before Step B
writes anything:

  cluster     — within a frame a mover is many connected points; speckle isn't.
  persistence — structure holds dynamic points for long; a mover passes through.

Both are computed on the full candidate set and ANDed, and only touch
dynamic_mask / dynamic_map: the voxel-key sets keep the raw verdict, so Step F's
recall bits still see every point removed here. Tuning:
docs/lidar_preprocessing_measurements.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from wato_lidar_preprocessing.config import DynamicFilterParams
from wato_lidar_preprocessing.voxel import voxel_indices


@dataclass
class FilterResult:
    keep: np.ndarray  # bool, aligned to the candidates
    n_cluster_dropped: int
    n_persistence_dropped: int  # beyond what the cluster test already removed


def persistence_keep(
    xyz: np.ndarray, sweep_id: np.ndarray, voxel_m: float, max_sweeps: int
) -> np.ndarray:
    """True where a point's voxel is occupied in < max_sweeps distinct sweeps.

    max_sweeps <= 0 disables the gate (all True).
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


def cluster_keep(
    xyz: np.ndarray, frame: np.ndarray, voxel_m: float, min_pts: int
) -> np.ndarray:
    """True where a point's cluster, within its frame, has >= min_pts points.

    Clusters are 26-connected components of the occupied voxel_m grid.
    min_pts <= 0 disables the gate (all True).
    """
    n = xyz.shape[0]
    if min_pts <= 0 or n == 0:
        return np.ones(n, dtype=bool)
    size = np.zeros(n, dtype=np.int64)
    order = np.argsort(frame, kind="stable")
    _, starts = np.unique(frame[order], return_index=True)
    for idx in np.split(order, starts[1:]):
        vox = np.floor(xyz[idx] / voxel_m).astype(np.int64)
        occupied, voxel_of = np.unique(vox, axis=0, return_inverse=True)
        voxel_of = voxel_of.ravel()
        pairs = cKDTree(occupied).query_pairs(r=1.0, p=np.inf, output_type="ndarray")
        graph = coo_matrix(
            (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
            shape=(len(occupied), len(occupied)),
        )
        _, label = connected_components(graph, directed=False)
        size[idx] = np.bincount(label[voxel_of])[label[voxel_of]]
    return size >= min_pts


def sweep_rate_hz(timestamps_ns: list[int]) -> float:
    """Sweeps per second across every scanner, from a chunk's sweep stamps."""
    if len(timestamps_ns) < 2:
        return 0.0
    span_ns = max(timestamps_ns) - min(timestamps_ns)
    return (len(timestamps_ns) - 1) / (span_ns * 1e-9) if span_ns > 0 else 0.0


def filter_candidates(
    params: DynamicFilterParams,
    xyz: np.ndarray,
    sweep_id: np.ndarray,
    frame: np.ndarray,
    rate_hz: float,
) -> FilterResult:
    """Keep mask over dynamic candidates (xyz, their sweep and frame ids).

    rate_hz (sweep_rate_hz) turns persistence_max_s into a sweep count; below
    two sweeps the persistence test has nothing to compare and is skipped.
    """
    clustered = cluster_keep(xyz, frame, params.cluster_voxel_m, params.min_cluster_pts)
    max_sweeps = int(round(params.persistence_max_s * rate_hz))
    brief = persistence_keep(
        xyz,
        sweep_id,
        params.persistence_voxel_m,
        max_sweeps if max_sweeps >= 2 else 0,
    )
    return FilterResult(
        keep=clustered & brief,
        n_cluster_dropped=int((~clustered).sum()),
        n_persistence_dropped=int((clustered & ~brief).sum()),
    )
