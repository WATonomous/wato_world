"""A chunk's fused proposals as per-frame detections for the tracker.

Each fused row is one detection. Its member rows say what backs it: a
lidar_mos member is Step F motion evidence, a lifted member carries SAM2's
instance id (the matching key) and its confidence. A detection is strong when
two sources agree on it or its lift is confident (config `strong`).

Only frames with at least one fused row are stepped; a track coasting over
empty frames is still predicted at each stepped frame's time.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from wato_common.artifacts import proposals
from wato_common.artifacts.proposals import Proposal
from wato_common.schemas import Box3D
from wato_tracking.config import StrongConfig

LIFTED = "lifted"
LIDAR_MOS = "lidar_mos"


@dataclass(frozen=True)
class Detection:
    proposal: Proposal  # the fused row
    strong: bool
    key: Optional[str]  # the lifted member's instance id
    two_source: bool
    has_lidar_mos: bool

    @property
    def box(self) -> Box3D:
        p = self.proposal
        return Box3D(cx=p.cx, cy=p.cy, cz=p.cz, w=p.w, l=p.l, h=p.h, heading=p.heading)


@dataclass(frozen=True)
class DetFrame:
    sweep_id: int  # the frame's reference sweep
    t_ns: int
    dets: tuple[Detection, ...]


def instance_of(lifted_proposal_id: str) -> str:
    """The instance id inside a lifted row's id, "lifted-<instance>-<sweep:06d>"."""
    return lifted_proposal_id[len(LIFTED) + 1 :].rsplit("-", 1)[0]


def detection(
    fused: Proposal, by_id: dict[str, Proposal], cfg: StrongConfig
) -> Detection:
    members = [by_id[m] for m in fused.member_proposal_ids]
    lift = next((m for m in members if m.provenance == LIFTED), None)
    has_mos = any(m.provenance == LIDAR_MOS for m in members)
    two_source = lift is not None and has_mos
    confident = (
        lift is not None
        and (lift.lift_confidence or 0.0) >= cfg.min_lift_confidence
        and (lift.lidar_point_count or 0) >= cfg.min_lift_points
    )
    return Detection(
        proposal=fused,
        strong=two_source or confident,
        key=instance_of(lift.proposal_id) if lift is not None else None,
        two_source=two_source,
        has_lidar_mos=has_mos,
    )


def frames(rows: list[Proposal], cfg: StrongConfig) -> list[DetFrame]:
    """The chunk's frames with fused rows, in time order."""
    by_id = {p.proposal_id: p for p in rows}
    by_sweep: dict[int, list[Detection]] = defaultdict(list)
    for p in proposals.fused(rows):
        by_sweep[p.sweep_id].append(detection(p, by_id, cfg))
    out = [
        DetFrame(sid, dets[0].proposal.reference_timestamp_ns, tuple(dets))
        for sid, dets in by_sweep.items()
    ]
    return sorted(out, key=lambda f: (f.t_ns, f.sweep_id))
