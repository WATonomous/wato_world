"""Source "lidar_mos": lidar_preprocessing Step F's moving-object clusters.

A cluster that passes the gates (config `lidar_mos`) is one proposal: Step F's
box, lowered to the ground. It takes a class from semantic_lifting when enough
of its points carry a lifted label, by a confidence-weighted vote.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np

from wato_common.artifacts.ground import GroundGrid
from wato_common.artifacts.lidar_sweeps import CanonicalLidarError, Frame
from wato_common.artifacts.lifted_labels import LiftedInstance
from wato_common.artifacts.motion_proposals import MotionCluster
from wato_common.schemas import Box3D
from wato_proposal_generation.boxes import SourceBox, proposal, to_ground
from wato_proposal_generation.config import LidarMosConfig
from wato_proposal_generation.frame_points import FramePoints
from wato_proposal_generation.lifted import instance

PROVENANCE = "lidar_mos"


def gate(clusters: list[MotionCluster], cfg: LidarMosConfig) -> list[MotionCluster]:
    return [
        c
        for c in clusters
        if c.motion_score > cfg.min_motion_score
        and c.n_sources >= cfg.min_n_sources
        and c.frac_persistent < cfg.max_frac_persistent
    ]


def by_frame(
    clusters: list[MotionCluster], frames: list[Frame]
) -> dict[int, list[MotionCluster]]:
    """Clusters keyed by their frame's reference sweep_id.

    Step F clustered per frame_sync frame and stamped each cluster with the
    frame's reference time; a cluster that doesn't land on one of `frames` at
    that time means those frames were grouped on another canonical LiDAR.
    """
    frame_of_sweep = {sid: f for f in frames for sid in f.sweep_ids}
    out: dict[int, list[MotionCluster]] = {}
    for c in clusters:
        f = frame_of_sweep.get(c.sweep_ids[0]) if c.sweep_ids else None
        if (
            f is None
            or not set(c.sweep_ids) <= set(f.sweep_ids)
            or c.reference_timestamp_ns != f.reference_timestamp_ns
        ):
            raise CanonicalLidarError(
                f"Step F cluster {c.cluster_id} (sweeps {list(c.sweep_ids)}, "
                f"t={c.reference_timestamp_ns}) matches no frame grouped on this "
                "config's canonical LiDAR; lidar_preprocessing's "
                "frame_sync.canonical_lidar differs from it"
            )
        out.setdefault(f.reference_sweep_id, []).append(c)
    return out


def _vote(
    members: np.ndarray,
    fp: FramePoints,
    instances: dict[str, LiftedInstance],
    min_labeled_fraction: float,
) -> Optional[LiftedInstance]:
    """The lifted instance whose class wins the cluster's points, if enough
    of them are labelled."""
    n = int(members.sum())
    labeled = members & (fp.instance_id != "")
    if n == 0 or labeled.sum() / n < min_labeled_fraction:
        return None
    by_class: dict[str, float] = defaultdict(float)
    by_instance: dict[str, float] = defaultdict(float)
    for iid, cls, conf in zip(
        fp.instance_id[labeled], fp.cls[labeled], fp.confidence[labeled]
    ):
        by_class[str(cls)] += float(conf)
        by_instance[str(iid)] += float(conf)
    winner = max(sorted(by_class), key=by_class.__getitem__)
    candidates = [
        iid for iid in sorted(by_instance) if instance(instances, iid).cls == winner
    ] or sorted(by_instance)
    return instance(instances, max(candidates, key=by_instance.__getitem__))


def proposals(
    frame: Frame,
    clusters: list[MotionCluster],
    fp: FramePoints,
    ground: Optional[GroundGrid],
    instances: dict[str, LiftedInstance],
    cfg: LidarMosConfig,
) -> list[SourceBox]:
    out = []
    for c in clusters:
        members = fp.cluster_id == c.cluster_id
        box = Box3D(cx=c.cx, cy=c.cy, cz=c.cz, w=c.w, l=c.l, h=c.h, heading=c.heading)
        inst = _vote(members, fp, instances, cfg.min_labeled_fraction)
        p = proposal(
            f"mos-{c.cluster_id}",
            frame,
            to_ground(box, ground),
            cls=inst.cls if inst else None,
            # Placeholder until per-source reliability is measured: Chen's
            # criterion is motion_score > 1, saturating at 2.
            score=min(1.0, c.motion_score / 2.0),
            provenance=PROVENANCE,
            lidar_point_count=int(members.sum()),
            supporting_cam_ids=inst.cam_ids if inst else (),
            supporting_masklet_ids=inst.masklet_ids if inst else (),
            motion_score=c.motion_score,
            n_sources=c.n_sources,
            frac_persistent=c.frac_persistent,
            track_life=c.track_life,
        )
        out.append(SourceBox(p, members))
    return out
