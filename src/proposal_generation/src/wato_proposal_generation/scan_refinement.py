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

import numpy as np
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
    """Standard DBSCAN with on-demand neighborhoods, avoiding an N² matrix.

    Counts include self. Borders receive labels but never propagate; a sparse
    chain of non-core points therefore cannot join two core components. Each
    core neighborhood is expanded once. Dense core chains can still merge.
    Rebuild the search tree after half its cores have been assigned, excluding
    them from future searches. This changes work, not connectivity: only
    unassigned cores can extend a component. Attach borders afterward to the
    first adjacent core component, matching ordered DBSCAN expansion.
    """
    core = tree.query_ball_point(xyz, radius_m, p=2, eps=0, return_length=True)
    core = core >= min_samples
    core_indices = np.flatnonzero(core)
    core_xyz = xyz[core_indices]
    core_labels = np.full(len(core_indices), -1, np.int64)
    active = np.arange(len(core_indices))
    remaining = len(active)
    core_tree = cKDTree(core_xyz)
    cluster_id = 0
    for start in range(len(core_indices)):
        if core_labels[start] >= 0:
            continue
        core_labels[start] = cluster_id
        remaining -= 1
        pending = [start]
        while pending and remaining:
            if remaining * 2 <= len(active):
                active = np.flatnonzero(core_labels < 0)
                core_tree = cKDTree(core_xyz[active])
            current = pending.pop()
            neighbors = np.asarray(
                core_tree.query_ball_point(core_xyz[current], radius_m, p=2, eps=0),
                dtype=np.int64,
            )
            neighbors = active[neighbors]
            unassigned = neighbors[core_labels[neighbors] < 0]
            core_labels[unassigned] = cluster_id
            remaining -= len(unassigned)
            pending.extend(unassigned.tolist())
        cluster_id += 1
    labels = np.full(len(xyz), -1, np.int64)
    labels[core_indices] = core_labels
    if core_indices.size:
        border_indices = np.flatnonzero(~core)
        # A non-core point has fewer than min_samples neighbors in the full
        # scan, so these lists stay small without discarding dense core edges.
        neighbors = cKDTree(core_xyz).query_ball_point(
            xyz[border_indices], radius_m, p=2, eps=0, return_sorted=False
        )
        for index, hits in zip(border_indices, neighbors):
            if hits:
                labels[index] = core_labels[hits].min()
    return labels


def refine(
    xyz: np.ndarray, seeds: dict[str, np.ndarray], cfg: ScanRefinementConfig
) -> dict[str, RefinedMask]:
    """Refine every nonempty instance against one bank, in shared row indices.

    xyz must contain the full NON-GROUND frame, in stable sweep/index order.
    All overlaps use original row indices; no downsampling or coordinate-based
    deduplication changes the IoU denominators. Empty/disabled banks retain
    original support. Disabling never substitutes the former radius filter.
    """
    if not seeds:
        return {}
    bank: list[tuple[float, np.ndarray, np.ndarray]] = []
    if cfg.enabled:
        tree = cKDTree(xyz)
        for radius in cfg.radii_m:
            labels = _dbscan(xyz, tree, radius, cfg.min_samples)
            sizes = np.bincount(labels[labels >= 0])
            bank.append((radius, labels, sizes))

    out = {}
    for iid, seed in sorted(seeds.items()):
        best_iou = 0.0
        best_radius = None
        best_cluster = None
        best_labels = None
        for radius, labels, sizes in bank:
            hits = labels[seed]
            clusters, intersections = np.unique(hits[hits >= 0], return_counts=True)
            if not clusters.size:
                continue
            ious = intersections / (len(seed) + sizes[clusters] - intersections)
            k = int(np.argmax(ious))
            if ious[k] > best_iou:
                best_iou = float(ious[k])
                best_radius = radius
                best_cluster = int(clusters[k])
                best_labels = labels
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
