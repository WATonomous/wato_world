"""Box helpers shared by every proposal source."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from wato_common.artifacts.ground import GroundGrid
from wato_common.artifacts.lidar_sweeps import Frame
from wato_common.artifacts.proposals import Proposal
from wato_common.schemas import Box3D


@dataclass(frozen=True)
class SourceBox:
    """A source's proposal and the frame points it claims, for fusion."""

    proposal: Proposal
    members: np.ndarray  # bool (P,) over the frame's FramePoints


def to_ground(box: Box3D, ground: Optional[GroundGrid]) -> Box3D:
    """The box with its bottom lowered to the ground under its centre.

    Neither source's points reach the road: lifted labels exclude Patchwork++
    ground, and Step F's boxes stop at its height floor (0.25 m on WATO), so
    wheels and feet are cut off. The bottom is never raised. No grid (Step C
    found no ground for the chunk): the box is unchanged.
    """
    if ground is None:
        return box
    ground_z = float(ground.height_at(np.array([[box.cx, box.cy]]))[0])
    top = box.cz + box.h / 2.0
    bottom = min(box.cz - box.h / 2.0, ground_z)
    return Box3D(
        cx=box.cx,
        cy=box.cy,
        cz=(top + bottom) / 2.0,
        w=box.w,
        l=box.l,
        h=top - bottom,
        heading=box.heading,
    )


def proposal(
    proposal_id: str,
    frame: Frame,
    box: Box3D,
    *,
    cls: Optional[str],
    score: float,
    provenance: str,
    lidar_point_count: int,
    supporting_cam_ids: tuple[str, ...] = (),
    supporting_masklet_ids: tuple[str, ...] = (),
    **evidence,
) -> Proposal:
    """A proposal row for `box` in `frame`; `evidence` fills the source's
    evidence columns (ProposalRow)."""
    return Proposal(
        proposal_id=proposal_id,
        sweep_id=frame.reference_sweep_id,
        frame_id=frame.frame_id,
        sweep_ids=frame.sweep_ids,
        reference_timestamp_ns=frame.reference_timestamp_ns,
        cx=box.cx,
        cy=box.cy,
        cz=box.cz,
        w=box.w,
        l=box.l,
        h=box.h,
        heading=box.heading,
        cls=cls,
        score=float(score),
        provenance=provenance,
        lidar_point_count=int(lidar_point_count),
        supporting_cam_ids=tuple(supporting_cam_ids),
        supporting_masklet_ids=tuple(supporting_masklet_ids),
        **evidence,
    )


def box_of(p: Proposal) -> Box3D:
    return Box3D(cx=p.cx, cy=p.cy, cz=p.cz, w=p.w, l=p.l, h=p.h, heading=p.heading)
