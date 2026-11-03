"""Matching predicted boxes to ground truth at one instant, and scoring it.

Matching follows the nuScenes detection benchmark: predictions in descending
score take the nearest unmatched ground-truth box whose BEV centre lies within
the distance threshold. It is class-agnostic (several sources are class-less);
class agreement is reported on the matches. Ground truth outside the taxonomy
(cls None: barriers, cones, ...) is "ignore": a prediction matched to it is
neither a hit nor a false positive, and it is never a miss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from wato_common.schemas import Box3D
from wato_common.tracking.boxes import iou_3d

# nuScenes' centre-distance thresholds (m).
DISTANCE_THRESHOLDS_M = (0.5, 1.0, 2.0, 4.0)
# Range bands (m from the ego at that instant), as in the measurement docs.
RANGE_BANDS_M = ((0.0, 15.0), (15.0, 30.0), (30.0, 50.0), (50.0, math.inf))


@dataclass(frozen=True)
class Scored:
    """A box with what scoring needs to know about it."""

    box: Box3D
    score: float = 1.0
    cls: Optional[str] = None
    range_m: float = 0.0


@dataclass(frozen=True)
class FrameMatch:
    pairs: tuple[tuple[int, int], ...]  # (pred idx, gt idx), cared-for GT only
    ignored_preds: tuple[int, ...]  # matched to ignore-GT
    false_positives: tuple[int, ...]
    misses: tuple[int, ...]  # cared-for GT idx left unmatched


def match_frame(
    preds: Sequence[Scored], gts: Sequence[Scored], max_dist_m: float
) -> FrameMatch:
    order = sorted(range(len(preds)), key=lambda i: (-preds[i].score, i))
    taken = np.zeros(len(gts), bool)
    g_xy = np.array([[g.box.cx, g.box.cy] for g in gts]).reshape(-1, 2)
    pairs, ignored, fps = [], [], []
    for i in order:
        if len(gts) == 0:
            fps.append(i)
            continue
        d = np.hypot(g_xy[:, 0] - preds[i].box.cx, g_xy[:, 1] - preds[i].box.cy)
        d[taken] = np.inf
        j = int(np.argmin(d))
        if d[j] > max_dist_m:
            fps.append(i)
            continue
        taken[j] = True
        if gts[j].cls is None:
            ignored.append(i)
        else:
            pairs.append((i, j))
    misses = [j for j, g in enumerate(gts) if g.cls is not None and not taken[j]]
    return FrameMatch(tuple(pairs), tuple(ignored), tuple(sorted(fps)), tuple(misses))


@dataclass
class Tally:
    """Hits, false positives and misses, plus errors on the hits."""

    n_gt: int = 0
    n_pred: int = 0
    hits: int = 0
    false_positives: int = 0
    centre_err_m: list[float] = field(default_factory=list)
    iou: list[float] = field(default_factory=list)
    dl: list[float] = field(default_factory=list)  # pred − gt length
    dw: list[float] = field(default_factory=list)
    dh: list[float] = field(default_factory=list)
    dz: list[float] = field(default_factory=list)  # pred − gt centre height
    class_agrees: list[bool] = field(default_factory=list)  # classed preds only

    def summary(self) -> dict:
        def med(v: list) -> Optional[float]:
            return float(np.median(v)) if v else None

        return dict(
            n_gt=self.n_gt,
            n_pred=self.n_pred,
            hits=self.hits,
            false_positives=self.false_positives,
            recall=self.hits / self.n_gt if self.n_gt else None,
            precision=(
                self.hits / (self.hits + self.false_positives)
                if self.hits + self.false_positives
                else None
            ),
            median_centre_err_m=med(self.centre_err_m),
            median_iou=med(self.iou),
            median_dl_m=med(self.dl),
            median_dw_m=med(self.dw),
            median_dh_m=med(self.dh),
            median_dz_m=med(self.dz),
            class_accuracy=(
                float(np.mean(self.class_agrees)) if self.class_agrees else None
            ),
        )


def band_of(range_m: float) -> str:
    for lo, hi in RANGE_BANDS_M:
        if lo <= range_m < hi:
            return f"{lo:g}-{hi:g}" if math.isfinite(hi) else f"{lo:g}+"
    raise ValueError(f"range {range_m} is negative")


def tally_frame(
    tallies: dict[str, Tally],
    preds: Sequence[Scored],
    gts: Sequence[Scored],
    max_dist_m: float,
    by_class: Optional[dict[str, Tally]] = None,
) -> None:
    """Add one instant to per-band tallies (key "all" plus each range band).

    A miss and a hit fall in the ground truth's band, a false positive in the
    prediction's. by_class, when given, gets the ground-truth side per
    ground-truth class (n_gt, hits and the errors on them; false positives
    have no ground-truth class, so its precision is meaningless).
    """
    m = match_frame(preds, gts, max_dist_m)

    def at(key: str) -> Tally:
        return tallies.setdefault(key, Tally())

    cared = [j for j, g in enumerate(gts) if g.cls is not None]
    for j in cared:
        for key in ("all", band_of(gts[j].range_m)):
            at(key).n_gt += 1
        if by_class is not None:
            by_class.setdefault(gts[j].cls, Tally()).n_gt += 1
    for i in range(len(preds)):
        if i in m.ignored_preds:
            continue
        for key in ("all", band_of(preds[i].range_m)):
            at(key).n_pred += 1
    for i in m.false_positives:
        for key in ("all", band_of(preds[i].range_m)):
            at(key).false_positives += 1
    for i, j in m.pairs:
        p, g = preds[i], gts[j]
        targets = [at("all"), at(band_of(g.range_m))]
        if by_class is not None:
            targets.append(by_class.setdefault(g.cls, Tally()))
        for t in targets:
            t.hits += 1
            t.centre_err_m.append(math.hypot(p.box.cx - g.box.cx, p.box.cy - g.box.cy))
            t.iou.append(iou_3d(p.box, g.box))
            t.dl.append(p.box.l - g.box.l)
            t.dw.append(p.box.w - g.box.w)
            t.dh.append(p.box.h - g.box.h)
            t.dz.append(p.box.cz - g.box.cz)
            if p.cls is not None:
                t.class_agrees.append(p.cls == g.cls)
