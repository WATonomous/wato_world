"""Does a distinct LiDAR segment back each lifted instance in this frame?

A camera mask says what an object is and which rays it covers; lifting gives
it the visible returns along those rays, which is the object only if the
LiDAR hit it. Behind a hedge, too far, or between beams, the returns are the
hedge in front or the tree line behind. Two published tests check the
association against the frame's own segmentation:

- SAL (Appendix A.1): the instance's lifted seed matches a scan cluster at
  point-set IoU strictly above 0.5 (best IoU at each DBSCAN scale);
- FGR (ICRA 2021, Sec. 3.1): that cluster lies at least theta_seg = 0.8 inside
  the 2D box frustum; a component reaching beyond it belongs to another object
  (a hedge strip, a wall, the background).

At each scale of the bank, the cluster with the best IoU against the seed is
tested; the instance is verified when any scale passes both, as FGR keeps the
best component over its radii: a coarse scale can merge a car with the hedge
in front while a finer one separates them. Of the passing scales, the verdict
names the one with the best IoU, SAL's own choice across its ensemble (the
first scale on ties), so the cluster it names is the object's segment rather
than the coarsest one that still fits the frustum. The seed is the instance's
candidate claims (lifted_candidates), whatever support method boxes it; the
frustum is lifting's per-instance rectangle relation. Geometry is unchanged.
docs/research/lidar_verified_lifting.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wato_proposal_generation.config import VerificationConfig
from wato_proposal_generation.scan_refinement import Bank, best_match

VERIFIED = "verified"
NO_OVERLAP = "no_overlap"  # every seed row is DBSCAN noise at every scale
IOU_NOT_ABOVE = "iou_not_above_threshold"
CONTAINMENT_BELOW = "containment_below_threshold"


@dataclass(frozen=True)
class Verdict:
    verified: bool
    status: str
    # Of the scales that verified it, the best-IoU one; else the best-IoU
    # failed scale (diagnostics). The first scale on ties.
    radius_m: float | None
    cluster_id: int | None
    iou: float
    containment: float | None
    cluster_points: int | None
    frustum_points: int


def verify(
    seeds: dict[str, np.ndarray],
    frustums: dict[str, np.ndarray],
    bank: Bank,
    n_rows: int,
    cfg: VerificationConfig,
) -> dict[str, Verdict]:
    """One verdict per seeded instance.

    Args:
        seeds: instance id -> frame rows of its candidate claims.
        frustums: instance id -> frame rows inside its masks' rectangles;
            every seed row is among them (lifted_candidates' invariant).
        bank: scan_refinement.build_bank over the frame's non-ground rows.
        n_rows: number of frame rows the bank labels.
    """
    out = {}
    for iid, seed in sorted(seeds.items()):
        frustum = frustums.get(iid, np.zeros(0, np.int64))
        in_frustum = np.zeros(n_rows, bool)
        in_frustum[frustum] = True
        best_passed: Verdict | None = None
        best_failed: Verdict | None = None
        for scale in bank:
            iou, cluster = best_match(seed, scale)
            if cluster is None:
                continue
            members = scale.labels == cluster
            size = int(members.sum())
            containment = float(in_frustum[members].sum()) / size
            passed = iou > cfg.min_iou and containment >= cfg.min_containment
            candidate = Verdict(
                verified=passed,
                status=(
                    VERIFIED
                    if passed
                    else IOU_NOT_ABOVE
                    if iou <= cfg.min_iou
                    else CONTAINMENT_BELOW
                ),
                radius_m=scale.radius_m,
                cluster_id=cluster,
                iou=iou,
                containment=containment,
                cluster_points=size,
                frustum_points=int(frustum.size),
            )
            if passed:
                if best_passed is None or iou > best_passed.iou:
                    best_passed = candidate
            elif best_failed is None or iou > best_failed.iou:
                best_failed = candidate
        verdict = best_passed or best_failed
        if verdict is None:
            verdict = Verdict(
                verified=False,
                status=NO_OVERLAP,
                radius_m=None,
                cluster_id=None,
                iou=0.0,
                containment=None,
                cluster_points=None,
                frustum_points=int(frustum.size),
            )
        out[str(iid)] = verdict
    return out
