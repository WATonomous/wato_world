"""Source "lifted": a box around each semantic_lifting instance's points.

The opt-in scan_clusters method uses SAL-style matching to replace support with
its best-IoU full-scan DBSCAN cluster when IoU > the threshold, retaining the
original support otherwise. A class height cap then filters box support; the
box is fit to it and lowered to ground. The default largest_group method keeps
radius-connected support per independent hypothesis. Larger hypotheses are
considered first; two filters compare their final support only to already
accepted survivors and reject another object's:
- a fragment: nearly all its points lie against a much bigger instance's
  points. A small mask beside a big one (a far car past a near car's edge, a
  low-score duplicate) claims a few of the big object's edge points, which
  box as a thin column beside it.
- a nested box: one mostly inside a same-class box with more points, a hidden
  object boxed on its occluder.

Coverage suppression, exclusive support, and the geometry filters share one
acceptance pass. Each box is fit to its proposed final support before its
geometry checks. Rejected hypotheses never take points away from another
object or suppress later candidates. The provisional exclusive semantic
labels are not used as seeds when retained claims are present.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from scipy.spatial import cKDTree

from wato_common.artifacts.ground import GroundGrid
from wato_common.artifacts.lidar_sweeps import Frame
from wato_common.artifacts.lifted_labels import LiftedInstance
from wato_common.geometry.objects import largest_group, to_ground
from wato_common.tracking import bev_inside_fraction, fit_bev_box
from wato_proposal_generation.boxes import SourceBox, box_of, proposal
from wato_proposal_generation.config import LiftedConfig
from wato_proposal_generation.frame_points import FramePoints
from wato_proposal_generation.overlap import resolve_support
from wato_proposal_generation.scan_refinement import RefinedMask, refine

PROVENANCE = "lifted"


def instance(instances: dict[str, LiftedInstance], iid: str) -> LiftedInstance:
    try:
        return instances[iid]
    except KeyError:
        raise KeyError(
            f"lifted instance {iid!r} labels points but is missing from "
            "lifted_instances.parquet; re-run semantic_lifting"
        ) from None


def refine_support(
    fp: FramePoints, instances: dict[str, LiftedInstance], cfg: LiftedConfig
) -> dict[str, RefinedMask]:
    """Select the published matcher or legacy cleanup for an explicit ablation.

    The legacy method remains separate; it is never a fallback for failed SAL
    matches. Its audit IoU is the retained component's overlap with its seed.
    """
    seeds = fp.seeds()
    if cfg.support_method == "scan_clusters":
        return refine(
            fp.xyz, {iid: s.point_indices for iid, s in seeds.items()}, cfg.refinement
        )
    out = {}
    for iid, claims in sorted(seeds.items()):
        seed = claims.point_indices
        cls = instance(instances, iid).cls
        radius = cfg.cluster_radius_by_class_m.get(cls, cfg.cluster_radius_m)
        keep = seed[largest_group(fp.xyz[seed], radius)]
        out[iid] = RefinedMask(
            seed, keep, radius, None, len(keep) / len(seed), "largest_group"
        )
    return out


def below_height(
    keep: np.ndarray, xyz: np.ndarray, ground: Optional[GroundGrid], cap_m: float
) -> np.ndarray:
    """The indices in keep at most cap_m above the ground (above the lowest
    of them without a grid)."""
    p = xyz[keep]
    above = ground.height_above(p) if ground is not None else p[:, 2] - p[:, 2].min()
    return keep[above <= cap_m]


def proposals(
    frame: Frame,
    fp: FramePoints,
    ground: Optional[GroundGrid],
    instances: dict[str, LiftedInstance],
    cfg: LiftedConfig,
    refinements: Optional[dict[str, RefinedMask]] = None,
) -> list[SourceBox]:
    if refinements is None:
        refinements = refine_support(fp, instances, cfg)
    seeds = fp.seeds()

    def make_box(iid: str, keep: np.ndarray) -> SourceBox:
        inst = instance(instances, iid)
        refined = refinements[iid]
        idx = refined.seed_indices
        members = np.zeros(fp.n, bool)
        members[keep] = True
        # Cluster-acquired points have no new camera evidence. Preserve the
        # original seed's confidence; only retained seed returns count toward
        # tracking's strong-lift gate.
        evidence = seeds[iid]
        confidence = float(
            evidence.confidences.mean()
            if cfg.support_method == "scan_clusters"
            else evidence.confidences[np.isin(evidence.point_indices, keep)].mean()
        )
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
            lift_seed_point_count=int(np.intersect1d(idx, keep).size),
        )
        return SourceBox(p, members)

    supports = {}
    for iid, refined in sorted(refinements.items()):
        inst = instance(instances, iid)
        keep = refined.point_indices
        cap = cfg.max_height_m.get(inst.cls)
        if cap is not None and keep.size:
            keep = below_height(keep, fp.xyz, ground, cap)
        supports[iid] = keep

    boxes: dict[str, SourceBox] = {}

    def accept(iid: str, keep: np.ndarray, accepted: dict[str, np.ndarray]) -> bool:
        # No candidate may suppress another and then disappear at a later
        # geometry gate. Check the proposed final box against survivors only;
        # after acceptance this box and its ownership are permanent.
        box = make_box(iid, keep)
        previous = [boxes[other] for other in accepted]
        p = box.proposal
        n = p.lidar_point_count
        if any(
            other.proposal.cls == p.cls
            and other.proposal.lidar_point_count > n
            and bev_inside_fraction(box_of(p), box_of(other.proposal))
            >= cfg.max_nested_fraction
            for other in previous
        ):
            return False
        bigger = [
            other
            for other in previous
            if other.proposal.lidar_point_count >= cfg.fragment_min_ratio * n
        ]
        if bigger:
            near = cKDTree(
                fp.xyz[np.logical_or.reduce([other.members for other in bigger])]
            )
            d, _ = near.query(fp.xyz[keep], distance_upper_bound=cfg.fragment_radius_m)
            if np.mean(np.isfinite(d)) >= cfg.fragment_min_fraction:
                return False
        boxes[iid] = box
        return True

    owned = resolve_support(
        supports,
        min_points=cfg.min_points,
        max_iom=cfg.max_support_iom,
        accept=accept,
    )
    return [boxes[iid] for iid in sorted(owned)]


def semantic_view(fp: FramePoints, boxes: list[SourceBox]) -> FramePoints:
    """Camera evidence retained by the final 3D owners, for motion class votes.

    A rejected candidate must not class an unmatched motion proposal through
    the provisional label raster. Geometric additions have no camera evidence
    and remain unlabeled in this view. Source artifacts stay unchanged.
    """
    seeds = fp.seeds()
    ids = np.full(fp.n, "", dtype=f"U{max(map(len, seeds), default=1)}")
    classes = np.full(
        fp.n, "", dtype=f"U{max((len(b.proposal.cls) for b in boxes), default=1)}"
    )
    confidence = np.zeros(fp.n, np.float32)
    for box in boxes:
        iid = min(box.proposal.supporting_masklet_ids)
        seed = seeds[iid]
        retained = box.members[seed.point_indices]
        idx = seed.point_indices[retained]
        ids[idx] = iid
        classes[idx] = box.proposal.cls
        confidence[idx] = seed.confidences[retained]
    return replace(fp, instance_id=ids, cls=classes, confidence=confidence)
