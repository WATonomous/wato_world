"""Appearance ReID: do two boxes' masklets look like one object?

SAM2 gives a box an identity (its lifted instance, tracker.py's key) only
while one masklet lasts. When the object changes camera, or SAM2 loses it and
a later detection starts a new masklet, the new box has a new instance, and
geometry alone decides whether it continues the track. perception_2d's
descriptors (wato_common.artifacts.masklets.MaskletEmbeddings: one unit
vector per masklet per embedded image) say whether the two masklets look
alike.

How the published trackers use such a cue, and what is taken from each:

- **Distance** between two masklets is the smallest cosine distance over
  their descriptor pairs: DeepSORT's gallery (arXiv 1703.07402 §2.4, a budget
  of the last 100 descriptors per track) and GIAOTracker's GlobalLink
  (arXiv 2202.11983 Eq. 5). A masklet keeps at most `budget` descriptors,
  evenly over its images; a track's gallery is the masklets of every box it
  took. A mean per masklet separates nearby same-class objects far worse
  (docs/tracking_measurements.md).
- **Fusion with geometry** is BoT-SORT's (arXiv 2206.14651 Eq. 12-13):
  appearance counts only for a pair that is close (here: within the
  association's distance gate) and alike (cosine distance under θ_emb); such
  a pair costs min(geometric cost, d_cos / 2). It never prices a pair the
  distance gate refuses, and a pair that doesn't look alike keeps its
  geometric cost.
- **θ_emb** has no published value for these descriptors: BoT-SORT read its
  0.25 off the distance histograms of matching and non-matching pairs of
  *its* ReID model (Fig. 5). `calibrate` builds the same two histograms from
  the bag itself, with labels the pipeline already has (Kuo et al., CVPR 2010,
  label a tracker's own output this way): two masklets lifting merged into
  one instance are one object; two of one class in one image with different
  instances are two. θ_emb is where the two error rates are equal. A fixed
  `appearance.max_cosine_distance` replaces the calibration.

Ids are chunk-local, so an Appearance covers one chunk; the threshold is
calibrated over the bag.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Optional, Sequence

import numpy as np

from wato_common.artifacts import masklets, proposals
from wato_common.artifacts.masklets import MaskletEmbeddings
from wato_common.artifacts.proposals import Proposal
from wato_tracking.detections import LIFTED, instance_of

log = logging.getLogger(__name__)

MemberCost = Callable[[Sequence[str], Sequence[str]], Optional[float]]


class Appearance:
    """One chunk's masklets' descriptor banks and the distances between them."""

    def __init__(self, embeddings: MaskletEmbeddings, budget: int) -> None:
        if budget < 1:
            raise ValueError(f"budget {budget}: a bank needs a descriptor")
        self.model = embeddings.model
        self.pooling = embeddings.pooling
        rows: dict[str, list[int]] = defaultdict(list)
        for k, m in enumerate(embeddings.masklet_ids.tolist()):
            rows[m].append(k)
        t = embeddings.camera_timestamp_ns
        self._bank: dict[str, np.ndarray] = {}
        for m, idx in rows.items():
            idx = sorted(idx, key=lambda k: int(t[k]))
            if len(idx) > budget:
                pick = np.unique(np.linspace(0, len(idx) - 1, budget).round())
                idx = [idx[int(k)] for k in pick]
            self._bank[m] = embeddings.features[idx].astype(np.float32)
        # Who shares an image with whom: (cam_id, camera_seq) -> masklets.
        self.images: dict[tuple[str, int], list[str]] = defaultdict(list)
        for m, cam, seq in zip(
            embeddings.masklet_ids.tolist(),
            embeddings.cam_ids.tolist(),
            embeddings.camera_seq.tolist(),
        ):
            self.images[(cam, seq)].append(m)
        self._cache: dict[tuple[str, str], float] = {}

    def __contains__(self, masklet_id: str) -> bool:
        return masklet_id in self._bank

    def distance(self, a: str, b: str) -> Optional[float]:
        """Smallest cosine distance between a's and b's descriptors; None
        when either masklet has none."""
        if a not in self._bank or b not in self._bank:
            return None
        if a == b:
            return 0.0
        key = (a, b) if a < b else (b, a)
        d = self._cache.get(key)
        if d is None:
            d = 1.0 - float((self._bank[a] @ self._bank[b].T).max())
            self._cache[key] = d = max(d, 0.0)
        return d

    def between(self, a: Sequence[str], b: Sequence[str]) -> Optional[float]:
        """Smallest distance between a masklet of a and one of b."""
        found = [d for x in a for y in b if (d := self.distance(x, y)) is not None]
        return min(found) if found else None

    def member_cost(self, max_cosine_distance: float) -> MemberCost:
        """BoT-SORT's appearance term for MultiObjectTracker and relink: half
        the cosine distance of a pair that looks alike, None otherwise."""

        def cost(a: Sequence[str], b: Sequence[str]) -> Optional[float]:
            d = self.between(a, b)
            return 0.5 * d if d is not None and d < max_cosine_distance else None

        return cost


def load(bag_id: str, chunk_id: str, budget: int) -> Appearance:
    """The chunk's Appearance. Raises FileNotFoundError when perception_2d
    wrote no descriptors for it."""
    return Appearance(masklets.read_embeddings(bag_id, chunk_id), budget)


@dataclass(frozen=True)
class Calibration:
    max_cosine_distance: float
    n_same: int  # pairs lifting calls one object
    n_different: int  # same-class pairs sharing an image
    # Share of each kind on the wrong side of the threshold (equal by
    # construction, up to the pairs' granularity).
    same_rejected: float
    different_accepted: float


def self_labelled_pairs(
    appearance: Appearance, rows: list[Proposal]
) -> tuple[list[float], list[float]]:
    """(same, different): descriptor distances of the chunk's masklet pairs
    that the pipeline itself can label.

    same: two masklets of one lifted instance (lifting merged them from the
    LiDAR points both claim: an object seen by two cameras, or picked up again
    by a new masklet). different: two masklets with a mask in one image, of
    one class and of different instances. A masklet in no lifted row has no
    instance and is in neither.
    """
    instance: dict[str, str] = {}
    cls: dict[str, Optional[str]] = {}
    of_instance: dict[str, dict[str, None]] = defaultdict(dict)
    for p in rows:
        if p.provenance != LIFTED:
            continue
        key = instance_of(p.proposal_id)
        for m in p.supporting_masklet_ids:
            if m in appearance:
                instance[m] = key
                cls[m] = p.cls
                of_instance[key][m] = None
    same = [
        appearance.distance(a, b)
        for members in of_instance.values()
        for a, b in combinations(members, 2)
    ]
    pairs: dict[tuple[str, str], None] = {}
    for present in appearance.images.values():
        for a, b in combinations(sorted(m for m in present if m in instance), 2):
            if instance[a] != instance[b] and cls[a] == cls[b]:
                pairs[(a, b)] = None
    different = [appearance.distance(a, b) for a, b in pairs]
    return same, different


def equal_error_threshold(
    same: Sequence[float], different: Sequence[float]
) -> tuple[float, float, float]:
    """(threshold, same pairs above it, different pairs under it), at the
    distance where the two shares are closest to equal. A pair is called the
    same object when its distance is under the threshold."""
    s, d = np.sort(np.asarray(same)), np.sort(np.asarray(different))
    cuts = np.unique(np.concatenate([s, d]))
    rejected = 1.0 - np.searchsorted(s, cuts, side="left") / len(s)
    accepted = np.searchsorted(d, cuts, side="left") / len(d)
    k = int(np.argmin(np.abs(rejected - accepted)))
    return float(cuts[k]), float(rejected[k]), float(accepted[k])


def calibrate(chunks: Sequence[tuple[Appearance, list[Proposal]]]) -> Calibration:
    """θ_emb from the bag's own labelled pairs (module docstring).

    Raises ValueError when the bag has no pair of one kind: one camera with
    nothing merged across masklets, or never two objects of a class in one
    image. Set appearance.max_cosine_distance, or turn appearance off.
    """
    same: list[float] = []
    different: list[float] = []
    for appearance, rows in chunks:
        s, d = self_labelled_pairs(appearance, rows)
        same += s
        different += d
    if not same or not different:
        raise ValueError(
            f"appearance: {len(same)} same-object and {len(different)} "
            "different-object masklet pairs to calibrate on; set "
            "appearance.max_cosine_distance or disable appearance"
        )
    threshold, rejected, accepted = equal_error_threshold(same, different)
    return Calibration(threshold, len(same), len(different), rejected, accepted)


def for_bag(
    bag_id: str,
    chunk_ids: Sequence[str],
    budget: int,
    max_cosine_distance: Optional[float],
) -> tuple[dict[str, MemberCost], Optional[Calibration]]:
    """Each chunk's member_cost, and the calibration behind the threshold
    (None when the config fixes it)."""
    loaded = {cid: load(bag_id, cid, budget) for cid in chunk_ids}
    kinds = {(a.model, a.pooling) for a in loaded.values()}
    if len(kinds) > 1:
        raise ValueError(
            f"bag {bag_id!r}: chunks embedded with different descriptors "
            f"{sorted(kinds)}; re-run perception_2d's embed command"
        )
    calibration = None
    if max_cosine_distance is None:
        calibration = calibrate(
            [(a, proposals.read(bag_id, cid)) for cid, a in loaded.items()]
        )
        max_cosine_distance = calibration.max_cosine_distance
        log.info(
            "appearance: θ_emb %.3f from %d same-object and %d different-object "
            "masklet pairs (%.0f%% / %.0f%% on the wrong side)",
            max_cosine_distance,
            calibration.n_same,
            calibration.n_different,
            100 * calibration.same_rejected,
            100 * calibration.different_accepted,
        )
    return (
        {cid: a.member_cost(max_cosine_distance) for cid, a in loaded.items()},
        calibration,
    )
