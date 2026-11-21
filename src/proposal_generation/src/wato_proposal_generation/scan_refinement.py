"""SAL Appendix A.1: replace lifted seeds with best-IoU non-ground clusters.

The six-scale bank sees all frame points, independently of labels and motion.
Only IoU strictly above the threshold replaces a seed; otherwise its original
indices survive, including DBSCAN noise. Matching is independent per instance:
no union, class gate, expansion cap, or exclusive-cluster allocation is added.

SAL leaves min_samples, metric, backend, and ties unspecified. We use exact
Euclidean XYZ DBSCAN, counting self, with core-only expansion. Input point
order resolves border ties; published scale order resolves equal mask IoUs.
See docs/research/sal_scan_cluster_refinement.md for these choices and the
adaptation from per-camera masks to this pipeline's merged instance labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from wato_proposal_generation.config import ScanRefinementConfig


@dataclass(frozen=True)
class RefinedMask:
    seed_indices: np.ndarray
    point_indices: np.ndarray
    # Best candidate, even when its IoU fails the replacement threshold.
    radius_m: float | None
    cluster_id: int | None
    iou: float
    # replaced | iou_not_above_threshold | no_overlap | disabled; the separate
    # legacy adapter uses largest_group, never as a failed-match fallback.
    status: str


def _dbscan(
    xyz: np.ndarray, tree: cKDTree, radius_m: float, min_samples: int
) -> np.ndarray:
    """Standard DBSCAN as connected components of the core graph.

    Counts include self. A DBSCAN cluster's cores are exactly a connected
    component of the graph joining cores within radius_m, so components are
    found in one sparse pass (no per-point Python expansion: on WATO frames
    of ~13k returns the expansion loop cost ~0.1 s a scale). Borders receive
    labels but never propagate; a sparse chain of non-core points therefore
    cannot join two core components. Clusters are numbered by their lowest
    core index and a border takes its lowest adjacent cluster, matching
    ordered DBSCAN expansion over the input order.
    """
    core = tree.query_ball_point(xyz, radius_m, p=2, eps=0, return_length=True)
    core = core >= min_samples
    core_indices = np.flatnonzero(core)
    labels = np.full(len(xyz), -1, np.int64)
    if not core_indices.size:
        return labels
    core_tree = cKDTree(xyz[core_indices])
    pairs = core_tree.query_pairs(radius_m, p=2, eps=0, output_type="ndarray")
    n_core = len(core_indices)
    graph = coo_matrix(
        (np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])),
        shape=(n_core, n_core),
    )
    _, component = connected_components(graph, directed=False)
    # Renumber components by their lowest core index (expansion order).
    _, first = np.unique(component, return_index=True)
    rank = np.empty(len(first), np.int64)
    rank[np.argsort(first)] = np.arange(len(first))
    core_labels = rank[component]
    labels[core_indices] = core_labels
    border_indices = np.flatnonzero(~core)
    # A non-core point has fewer than min_samples neighbors in the full scan,
    # so these lists stay small.
    hits = core_tree.query_ball_point(
        xyz[border_indices], radius_m, p=2, eps=0, return_sorted=False
    )
    lengths = np.fromiter((len(h) for h in hits), np.int64, len(hits))
    if lengths.any():
        owner = np.repeat(border_indices, lengths)
        adjacent = core_labels[np.fromiter(chain.from_iterable(hits), np.int64)]
        lowest = np.full(len(xyz), np.iinfo(np.int64).max)
        np.minimum.at(lowest, owner, adjacent)
        attached = border_indices[lengths > 0]
        labels[attached] = lowest[attached]
    return labels


@dataclass(frozen=True)
class Scale:
    """One DBSCAN partition of the frame: labels per row (-1 noise) and the
    size of each cluster."""

    radius_m: float
    labels: np.ndarray
    sizes: np.ndarray


Bank = list[Scale]


def build_bank(xyz: np.ndarray, radii_m: tuple[float, ...], min_samples: int) -> Bank:
    """The frame's DBSCAN partitions, one per radius in the given order.

    xyz must contain the full NON-GROUND frame, in stable sweep/index order:
    refinement and verification both index it by frame row.
    """
    if not len(xyz):
        return []
    tree = cKDTree(xyz)
    bank = []
    for radius in radii_m:
        labels = _dbscan(xyz, tree, radius, min_samples)
        bank.append(Scale(radius, labels, np.bincount(labels[labels >= 0])))
    return bank


def best_match(seed: np.ndarray, scale: Scale) -> tuple[float, int | None]:
    """(IoU, cluster id) of the scale's cluster with the highest point-set IoU
    against the seed's rows; (0.0, None) when every seed row is noise. Equal
    IoUs keep the lower cluster id."""
    hits = scale.labels[seed]
    clusters, intersections = np.unique(hits[hits >= 0], return_counts=True)
    if not clusters.size:
        return 0.0, None
    ious = intersections / (len(seed) + scale.sizes[clusters] - intersections)
    k = int(np.argmax(ious))
    return float(ious[k]), int(clusters[k])


def refine(
    xyz: np.ndarray,
    seeds: dict[str, np.ndarray],
    cfg: ScanRefinementConfig,
    bank: Bank | None = None,
) -> dict[str, RefinedMask]:
    """Refine every nonempty instance against one bank, in shared row indices.

    xyz must contain the full NON-GROUND frame, in stable sweep/index order.
    All overlaps use original row indices; no downsampling or coordinate-based
    deduplication changes the IoU denominators. Empty/disabled banks retain
    original support. Disabling never substitutes the former radius filter.
    ``bank``, when given, is build_bank's for the same xyz and cfg (shared with
    verification); a disabled config ignores it.
    """
    if not seeds:
        return {}
    if not cfg.enabled:
        bank = []
    elif bank is None:
        bank = build_bank(xyz, cfg.radii_m, cfg.min_samples)

    out = {}
    for iid, seed in sorted(seeds.items()):
        best_iou = 0.0
        best_radius = None
        best_cluster = None
        best_labels = None
        for scale in bank:
            iou, cluster = best_match(seed, scale)
            if cluster is not None and iou > best_iou:
                best_iou = iou
                best_radius = scale.radius_m
                best_cluster = cluster
                best_labels = scale.labels
        replaced = best_labels is not None and best_iou > cfg.min_iou
        if replaced:
            indices = np.flatnonzero(best_labels == best_cluster)
            status = "replaced"
        else:
            indices = seed
            if not cfg.enabled:
                status = "disabled"
            elif best_labels is None:
                status = "no_overlap"
            else:
                status = "iou_not_above_threshold"
        out[str(iid)] = RefinedMask(
            seed, indices, best_radius, best_cluster, best_iou, status
        )
    return out
