"""Fusion: one fused proposal per object in a frame.

A lidar_mos box and a lifted box are the same object when they share enough of
the frame's LiDAR points (config `fusion`); pairs are taken greedily, most
shared points first, each box in at most one pair. A pair becomes one box fit
to both boxes' points, with the lifted box's class. A box no other source
matched is fused on its own, class-less if it has no class, so every object
either source found has a fused row.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from wato_common.artifacts.ground import GroundGrid
from wato_common.artifacts.lidar_sweeps import Frame
from wato_common.artifacts.proposals import FUSED, Proposal
from wato_common.tracking import fit_bev_box
from wato_proposal_generation.boxes import SourceBox, proposal, to_ground
from wato_proposal_generation.config import FusionConfig
from wato_proposal_generation.frame_points import FramePoints


def match(
    mos: list[SourceBox], lifted: list[SourceBox], cfg: FusionConfig
) -> list[tuple[int, int]]:
    """(mos index, lifted index) pairs, one to one."""
    candidates = []
    for i, m in enumerate(mos):
        n_m = int(m.members.sum())
        for j, f in enumerate(lifted):
            n_f = int(f.members.sum())
            if n_m == 0 or n_f == 0:
                continue
            shared = int((m.members & f.members).sum())
            fraction = shared / min(n_m, n_f)
            if shared >= cfg.min_shared_points and fraction >= cfg.min_shared_fraction:
                candidates.append((shared, fraction, i, j))
    pairs = []
    used_m: set[int] = set()
    used_f: set[int] = set()
    for _, _, i, j in sorted(candidates, key=lambda c: (-c[0], -c[1], c[2], c[3])):
        if i in used_m or j in used_f:
            continue
        used_m.add(i)
        used_f.add(j)
        pairs.append((i, j))
    return pairs


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
    )


def fuse(
    frame: Frame,
    mos: list[SourceBox],
    lifted: list[SourceBox],
    fp: FramePoints,
    ground: Optional[GroundGrid],
    cfg: FusionConfig,
) -> list[Proposal]:
    pairs = match(mos, lifted, cfg)
    out = []

    def next_id() -> str:
        return f"fused-{frame.reference_sweep_id:06d}-{len(out)}"

    for i, j in pairs:
        m, f = mos[i].proposal, lifted[j].proposal
        union = mos[i].members | lifted[j].members
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
    matched_m = {i for i, _ in pairs}
    matched_f = {j for _, j in pairs}
    for i, m in enumerate(mos):
        if i not in matched_m:
            out.append(_alone(m.proposal, next_id()))
    for j, f in enumerate(lifted):
        if j not in matched_f:
            out.append(_alone(f.proposal, next_id()))
    return out


def counts(proposals: list[Proposal]) -> dict[str, int]:
    """Rows per provenance, and fused rows by how many sources they merge."""
    out: dict[str, int] = {}
    for p in proposals:
        out[p.provenance] = out.get(p.provenance, 0) + 1
        if p.provenance == FUSED and len(p.member_proposal_ids) > 1:
            out["fused_matched"] = out.get("fused_matched", 0) + 1
    return out
