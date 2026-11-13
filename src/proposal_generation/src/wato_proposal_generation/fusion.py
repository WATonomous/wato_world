"""Fusion: one fused proposal per object in a frame.

A lidar_mos box and a lifted box are the same object when they share enough of
the frame's LiDAR points (config `fusion`); pairs are taken greedily, most
shared points first, each box in at most one pair. A pair becomes one box fit
to both boxes' points, with the lifted box's class. A box no other source
matched is fused on its own, class-less if it has no class, so every object
either source found has a fused row.

A fusion_detector box (detector.py) then joins the object it shares the most
points with, by the same rule, one to one. It is recorded as supporting
provenance; the object's geometry, class, score and point count remain those
of the motion/lifted sources. An unmatched detector box can supply a new
object when fusion_detector.add_unmatched is set.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from wato_common.artifacts.ground import GroundGrid
from wato_common.artifacts.lidar_sweeps import Frame
from wato_common.artifacts.proposals import FUSED, Proposal
from wato_common.tracking import fit_bev_box
from wato_common.geometry.objects import to_ground
from wato_proposal_generation.boxes import SourceBox, proposal
from wato_proposal_generation.config import FusionConfig
from wato_proposal_generation.frame_points import FramePoints


def _pairs(
    a: list[np.ndarray], b: list[np.ndarray], cfg: FusionConfig
) -> list[tuple[int, int]]:
    """(a index, b index) pairs of member masks sharing enough points, one to
    one, most shared points first."""
    candidates = []
    for i, ma in enumerate(a):
        n_a = int(ma.sum())
        for j, mb in enumerate(b):
            n_b = int(mb.sum())
            if n_a == 0 or n_b == 0:
                continue
            shared = int((ma & mb).sum())
            fraction = shared / min(n_a, n_b)
            if shared >= cfg.min_shared_points and fraction >= cfg.min_shared_fraction:
                candidates.append((shared, fraction, i, j))
    pairs = []
    used_a: set[int] = set()
    used_b: set[int] = set()
    for _, _, i, j in sorted(candidates, key=lambda c: (-c[0], -c[1], c[2], c[3])):
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((i, j))
    return pairs


def match(
    mos: list[SourceBox], lifted: list[SourceBox], cfg: FusionConfig
) -> list[tuple[int, int]]:
    """(mos index, lifted index) pairs, one to one."""
    return _pairs([m.members for m in mos], [f.members for f in lifted], cfg)


def _alone(p: Proposal, fused_id: str) -> Proposal:
    """A fused row standing for a single source row."""
    return replace(
        p,
        proposal_id=fused_id,
        provenance=FUSED,
        member_proposal_ids=(p.proposal_id,),
        motion_score=None,
        n_sources=None,
        frac_persistent=None,
        track_life=None,
        lift_confidence=None,
        lift_seed_point_count=None,
    )


@dataclass(frozen=True)
class _Object:
    """What the lidar_mos / lifted matching found, before the detector joins."""

    mos: Optional[SourceBox]
    lifted: Optional[SourceBox]

    @property
    def members(self) -> np.ndarray:
        if self.mos is not None and self.lifted is not None:
            return self.mos.members | self.lifted.members
        return (self.mos or self.lifted).members


def fuse(
    frame: Frame,
    mos: list[SourceBox],
    lifted: list[SourceBox],
    fp: FramePoints,
    ground: Optional[GroundGrid],
    cfg: FusionConfig,
    detector: Optional[list[SourceBox]] = None,
    add_unmatched_detector: bool = True,
    unmatched_detector_min_score: float = 0.6,
    unmatched_detector_min_points: int = 5,
) -> list[Proposal]:
    pairs = match(mos, lifted, cfg)
    matched_m = {i for i, _ in pairs}
    matched_f = {j for _, j in pairs}
    # In emission order: pairs, then lone lidar_mos, then lone lifted boxes.
    objects = [_Object(mos[i], lifted[j]) for i, j in pairs]
    objects += [_Object(m, None) for i, m in enumerate(mos) if i not in matched_m]
    objects += [_Object(None, f) for j, f in enumerate(lifted) if j not in matched_f]

    detector = detector or []
    joined = dict(
        _pairs([o.members for o in objects], [d.members for d in detector], cfg)
    )

    out: list[Proposal] = []

    def next_id() -> str:
        return f"fused-{frame.reference_sweep_id:06d}-{len(out)}"

    for k, o in enumerate(objects):
        if o.mos is not None and o.lifted is not None:
            m, f = o.mos.proposal, o.lifted.proposal
            union = o.members
            out.append(
                replace(
                    proposal(
                        next_id(),
                        frame,
                        to_ground(fit_bev_box(fp.xyz[union]), ground),
                        cls=f.cls,
                        score=(m.score + f.score) / 2.0,
                        provenance=FUSED,
                        lidar_point_count=int(union.sum()),
                        supporting_cam_ids=f.supporting_cam_ids,
                        supporting_masklet_ids=f.supporting_masklet_ids,
                    ),
                    member_proposal_ids=(m.proposal_id, f.proposal_id),
                )
            )
        else:
            out.append(_alone((o.mos or o.lifted).proposal, next_id()))
        if k in joined:
            p = out[-1]
            out[-1] = replace(
                p,
                member_proposal_ids=p.member_proposal_ids
                + (detector[joined[k]].proposal.proposal_id,),
            )
    if add_unmatched_detector:
        taken = set(joined.values())
        for j, d in enumerate(detector):
            if (
                j not in taken
                and d.proposal.score >= unmatched_detector_min_score
                and int(d.members.sum()) >= unmatched_detector_min_points
            ):
                out.append(_alone(d.proposal, next_id()))
    return out


def counts(proposals: list[Proposal]) -> dict[str, int]:
    """Rows per provenance, and fused rows by how many sources they merge."""
    out: dict[str, int] = {}
    for p in proposals:
        out[p.provenance] = out.get(p.provenance, 0) + 1
        if p.provenance == FUSED and len(p.member_proposal_ids) > 1:
            out["fused_matched"] = out.get("fused_matched", 0) + 1
    return out
