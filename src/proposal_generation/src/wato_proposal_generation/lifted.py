"""Source "lifted": a box around each semantic_lifting instance's points.

Per frame, an instance's labelled points across the frame's sweeps. Masklets
bleed onto background (60% of unmerged masklets' points jump > 5 m between
sweeps, docs/semantic_lifting_measurements.md), so only the largest group of
points within the class's cluster radius of each other is the object, minus
any above the class's height cap; the box is fit to it and lowered to the
ground. Last, a box mostly inside a same-class box with more points is
dropped: it is a hidden object boxed on its occluder.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from wato_common.artifacts.ground import GroundGrid
from wato_common.artifacts.lidar_sweeps import Frame
from wato_common.artifacts.lifted_labels import LiftedInstance
from wato_common.geometry.objects import largest_group, to_ground
from wato_common.tracking import bev_inside_fraction, fit_bev_box
from wato_proposal_generation.boxes import SourceBox, box_of, proposal
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


def below_height(
    keep: np.ndarray, xyz: np.ndarray, ground: Optional[GroundGrid], cap_m: float
) -> np.ndarray:
    """The indices in keep at most cap_m above the ground (above the lowest
    of them without a grid)."""
    p = xyz[keep]
    above = ground.height_above(p) if ground is not None else p[:, 2] - p[:, 2].min()
    return keep[above <= cap_m]


def drop_nested(boxes: list[SourceBox], max_fraction: float) -> list[SourceBox]:
    """The boxes minus each one whose BEV footprint lies at least max_fraction
    inside a same-class box with more points (that one kept)."""
    order = sorted(
        range(len(boxes)), key=lambda i: (-boxes[i].proposal.lidar_point_count, i)
    )
    kept: list[int] = []
    for i in order:
        p = boxes[i].proposal
        if not any(
            boxes[k].proposal.cls == p.cls
            and boxes[k].proposal.lidar_point_count > p.lidar_point_count
            and bev_inside_fraction(box_of(p), box_of(boxes[k].proposal))
            >= max_fraction
            for k in kept
        ):
            kept.append(i)
    return [boxes[i] for i in sorted(kept)]


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
        inst = instance(instances, iid)
        idx = np.flatnonzero(fp.instance_id == iid)
        radius = cfg.cluster_radius_by_class_m.get(inst.cls, cfg.cluster_radius_m)
        keep = idx[largest_group(fp.xyz[idx], radius)]
        cap = cfg.max_height_m.get(inst.cls)
        if cap is not None and keep.size:
            keep = below_height(keep, fp.xyz, ground, cap)
        if keep.size < cfg.min_points:
            continue
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
    return drop_nested(out, cfg.max_nested_fraction)
