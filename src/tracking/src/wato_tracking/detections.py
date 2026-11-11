"""A chunk's fused proposals as per-frame detections for the tracker.

Each fused row is one detection. Its member rows say what backs it: a
lidar_mos member is Step F motion evidence, a lifted member carries SAM2's
instance id (the matching key) and its confidence, a fusion_detector member
carries BEVFusion's score. A detection is strong when lidar_mos and lifted
agree on it, its lift is confident, or its detector score is high (config
`strong`). The detector and lifting share cameras, so their agreement alone
doesn't count as two sources. A box backed only by the detector starts a
track only when scored high enough (ByteTrack's birth rule); otherwise it can
only extend one.

Only frames with at least one fused row are stepped; a track coasting over
empty frames is still predicted at each stepped frame's time.

With `duplicates` on, each frame's boxes go through CenterPoint's circle NMS
first: a second box on one object, beside the box its track takes, would
otherwise start a second track (IoU 0 at ~1 m, so no overlap rule reaches it).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from wato_common.artifacts import proposals
from wato_common.artifacts.proposals import Proposal
from wato_common.schemas import Box3D
from wato_tracking.config import DuplicatesConfig, StrongConfig

log = logging.getLogger(__name__)

LIFTED = "lifted"
LIDAR_MOS = "lidar_mos"
FUSION_DETECTOR = "fusion_detector"


@dataclass(frozen=True)
class Detection:
    proposal: Proposal  # the fused row
    strong: bool
    key: Optional[str]  # the lifted member's instance id
    two_source: bool
    has_lidar_mos: bool
    detector_score: Optional[float] = None  # its fusion_detector member's
    can_birth: bool = True  # may start a track when no track takes it

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
    det = next((m for m in members if m.provenance == FUSION_DETECTOR), None)
    two_source = lift is not None and has_mos
    # Recovered scan geometry must not manufacture strong camera evidence.
    # Before scan refinement the total count was labeled support, so retain
    # that interpretation for legacy artifacts with no separate seed count.
    lift_points = 0
    if lift is not None:
        lift_points = lift.lift_seed_point_count
        if lift_points is None:
            lift_points = lift.lidar_point_count
    confident = (
        lift is not None
        and (lift.lift_confidence or 0.0) >= cfg.min_lift_confidence
        and (lift_points or 0) >= cfg.min_lift_points
    )
    detected = det is not None and det.score >= cfg.min_detector_score
    detector_only = det is not None and lift is None and not has_mos
    return Detection(
        proposal=fused,
        strong=two_source or confident or detected,
        key=instance_of(lift.proposal_id) if lift is not None else None,
        two_source=two_source,
        has_lidar_mos=has_mos,
        detector_score=det.score if det is not None else None,
        can_birth=not detector_only or det.score >= cfg.min_detector_birth_score,
    )


def suppress_duplicates(
    dets: list[Detection], cfg: DuplicatesConfig
) -> list[Detection]:
    """CenterPoint's circle NMS over one frame's detections (config
    `duplicates`): in rank order, each kept box drops the lower-ranked boxes
    of its class group whose BEV centres lie within sqrt(min_radius) of it.
    Boxes with no class, or a class in no group, are always kept."""
    radius = {c: g.min_radius for g in cfg.groups for c in g.classes}
    group = {c: k for k, g in enumerate(cfg.groups) for c in g.classes}
    order = sorted(
        range(len(dets)),
        key=lambda i: (
            not dets[i].strong,
            -(dets[i].proposal.lidar_point_count or 0),
            i,
        ),
    )
    dropped = [False] * len(dets)
    for a, i in enumerate(order):
        ci = dets[i].proposal.cls
        if dropped[i] or ci not in group:
            continue
        pi = dets[i].proposal
        for j in order[a + 1 :]:
            pj = dets[j].proposal
            if dropped[j] or group.get(pj.cls) != group[ci]:
                continue
            if (pi.cx - pj.cx) ** 2 + (pi.cy - pj.cy) ** 2 <= radius[ci]:
                dropped[j] = True
    return [d for d, x in zip(dets, dropped) if not x]


def frames(
    rows: list[Proposal],
    cfg: StrongConfig,
    duplicates: Optional[DuplicatesConfig] = None,
) -> list[DetFrame]:
    """The chunk's frames with fused rows, in time order."""
    by_id = {p.proposal_id: p for p in rows}
    by_sweep: dict[int, list[Detection]] = defaultdict(list)
    for p in proposals.fused(rows):
        by_sweep[p.sweep_id].append(detection(p, by_id, cfg))
    if duplicates is not None and duplicates.enabled:
        n = sum(map(len, by_sweep.values()))
        by_sweep = {
            sid: suppress_duplicates(dets, duplicates) for sid, dets in by_sweep.items()
        }
        kept = sum(map(len, by_sweep.values()))
        log.info("duplicates: %d of %d fused boxes suppressed", n - kept, n)
    out = [
        DetFrame(sid, dets[0].proposal.reference_timestamp_ns, tuple(dets))
        for sid, dets in by_sweep.items()
    ]
    return sorted(out, key=lambda f: (f.t_ns, f.sweep_id))
