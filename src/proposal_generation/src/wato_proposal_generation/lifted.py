"""Source "lifted": a box around each semantic_lifting instance's points.

Per frame, an instance's labelled points across the frame's sweeps. Masklets
bleed onto background (60% of unmerged masklets' points jump > 5 m between
sweeps, docs/semantic_lifting_measurements.md), so only the largest group of
points within cluster_radius_m of each other is the object; the box is fit to
it and lowered to the ground.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from wato_common.artifacts.ground import GroundGrid
from wato_common.artifacts.lidar_sweeps import Frame
from wato_common.artifacts.lifted_labels import LiftedInstance
from wato_common.tracking import fit_bev_box
from wato_proposal_generation.boxes import SourceBox, proposal, to_ground
from wato_proposal_generation.config import LiftedConfig
from wato_proposal_generation.frame_points import FramePoints

PROVENANCE = "lifted"


def instance(instances: dict[str, LiftedInstance], iid: str) -> LiftedInstance:
    try:
        return instances[iid]
    except KeyError:
        raise KeyError(
            f"lifted instance {iid!r} labels points but is missing from "
            "lifted_instances.parquet; re-run semantic_lifting"
        ) from None


def largest_group(xyz: np.ndarray, radius_m: float) -> np.ndarray:
    """Indices of the largest set of points chained within radius_m."""
    n = xyz.shape[0]
    if n <= 1:
        return np.arange(n)
    pairs = cKDTree(xyz).query_pairs(radius_m, output_type="ndarray")
    graph = coo_matrix(
        (np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n)
    )
    _, labels = connected_components(graph, directed=False)
    return np.flatnonzero(labels == np.bincount(labels).argmax())


def proposals(
    frame: Frame,
    fp: FramePoints,
    ground: Optional[GroundGrid],
    instances: dict[str, LiftedInstance],
    cfg: LiftedConfig,
) -> list[SourceBox]:
    out = []
    for iid in np.unique(fp.instance_id[fp.instance_id != ""]):
        iid = str(iid)
        idx = np.flatnonzero(fp.instance_id == iid)
        keep = idx[largest_group(fp.xyz[idx], cfg.cluster_radius_m)]
        if keep.size < cfg.min_points:
            continue
        inst = instance(instances, iid)
        members = np.zeros(fp.n, bool)
        members[keep] = True
        confidence = float(fp.confidence[keep].mean())
        p = proposal(
            f"lifted-{iid}-{frame.reference_sweep_id:06d}",
            frame,
            to_ground(fit_bev_box(fp.xyz[keep]), ground),
            cls=inst.cls,
            # Placeholder until per-source reliability is measured.
            score=confidence,
            provenance=PROVENANCE,
            lidar_point_count=int(keep.size),
            supporting_cam_ids=inst.cam_ids,
            supporting_masklet_ids=inst.masklet_ids,
            lift_confidence=confidence,
        )
        out.append(SourceBox(p, members))
    return out
