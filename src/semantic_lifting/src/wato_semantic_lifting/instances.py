"""Cross-camera instances: one object's masklets merged into one instance.

perception_2d tracks objects per camera, and `global_object_id` is only
assigned by `tracking`, which runs after lifting — but proposal_generation
groups LiDAR points by lifted instance to fit one box per object, so a car
seen by two cameras must be one instance here. The evidence is geometric:
two masklets from different cameras that both claim the same visible LiDAR
points are looking at the same surface.

A pair merges when, over the chunk, they share at least min_shared_points
points and those are at least min_shared_fraction of the smaller masklet's
claims in the sweeps where they overlap — so a road-sized mask that touches a
car at its edge (a few of the car's points) doesn't absorb it. Only masklets
of one class merge. Strongest pairs go first, and a merge is refused if the
group would then hold two masklets that are different objects:

- two masklets one camera saw in the same frame;
- two masklets whose claimed points are, at the median over the sweeps where
  both claim some, more than max_separation_m apart. A SAM2 masklet can drift
  from one object onto another (perception_2d's track-ending problem); without
  this, it links camera B's masklet on the first object to camera C's on the
  second, and chains of such links joined cars 60-125 m apart.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from wato_common.artifacts.masklets import MaskletRecord
from wato_semantic_lifting.voting import Votes


class CoClaims:
    """Accumulates, over a chunk's sweeps, the points each pair of masklets
    both claim, and where each masklet's claimed points are. Pairs are
    (label, label), ascending; a camera votes once per point, so the two are
    always from different cameras."""

    def __init__(self) -> None:
        self.shared: Counter[tuple[int, int]] = Counter()
        # Σ over the sweeps where the pair shares points of the smaller of the
        # two masklets' claims in that sweep.
        self.smaller_claim: Counter[tuple[int, int]] = Counter()
        # label -> sweep_id -> median (x, y) of the points it claimed.
        self.positions: dict[int, dict[int, tuple[float, float]]] = {}

    def add(self, sweep_id: int, votes: Votes, xyz: np.ndarray) -> None:
        """One sweep's votes and its world-frame points."""
        if len(votes) == 0:
            return
        for label in np.unique(votes.label_idx).tolist():
            pts = xyz[votes.point_idx[votes.label_idx == label], :2]
            x, y = np.median(pts, axis=0).tolist()
            self.positions.setdefault(label, {})[sweep_id] = (x, y)
        order = np.lexsort((votes.label_idx, votes.point_idx))
        pts, labels = votes.point_idx[order], votes.label_idx[order]
        starts = np.flatnonzero(np.r_[True, pts[1:] != pts[:-1]])
        sizes = np.diff(np.r_[starts, len(pts)])
        here: Counter[tuple[int, int]] = Counter()
        for s, k in zip(starts[sizes > 1].tolist(), sizes[sizes > 1].tolist()):
            ls = labels[s : s + k].tolist()
            for i in range(k):
                for j in range(i + 1, k):
                    if ls[i] != ls[j]:
                        here[(ls[i], ls[j])] += 1
        if not here:
            return
        claims = Counter(votes.label_idx.tolist())
        for (a, b), n in here.items():
            self.shared[(a, b)] += n
            self.smaller_claim[(a, b)] += min(claims[a], claims[b])


def _separation_m(co: CoClaims, a: int, b: int) -> float:
    """Median distance between where a's and b's claimed points are, over the
    sweeps where both claim some; 0 when they never do."""
    pa, pb = co.positions.get(a, {}), co.positions.get(b, {})
    common = pa.keys() & pb.keys()
    if not common:
        return 0.0
    return float(np.median([np.hypot(*np.subtract(pa[s], pb[s])) for s in common]))


def merge_masklets(
    records: list[MaskletRecord],
    co: CoClaims,
    *,
    min_shared_points: int,
    min_shared_fraction: float,
    max_separation_m: float,
) -> np.ndarray:
    """(L,) int64: each masklet's instance, as the index of the group's
    masklet with the smallest masklet_id (a lone masklet is its own)."""
    parent = list(range(len(records)))
    # Per group root: its members, and camera -> frames they cover.
    members = [[i] for i in range(len(records))]
    frames = [{r.cam_id: set(r.frames_present)} for r in records]

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    candidates = sorted(
        (
            pair
            for pair, n in co.shared.items()
            if n >= min_shared_points
            and n >= min_shared_fraction * co.smaller_claim[pair]
            and records[pair[0]].cls == records[pair[1]].cls
        ),
        key=lambda pair: (-co.shared[pair], pair),
    )
    for a, b in candidates:
        ra, rb = root(a), root(b)
        if ra == rb:
            continue
        fa, fb = frames[ra], frames[rb]
        if any(cam in fb and seqs & fb[cam] for cam, seqs in fa.items()):
            continue
        if any(
            _separation_m(co, p, q) > max_separation_m
            for p in members[ra]
            for q in members[rb]
        ):
            continue
        parent[rb] = ra
        members[ra].extend(members[rb])
        for cam, seqs in fb.items():
            fa.setdefault(cam, set()).update(seqs)

    groups: dict[int, list[int]] = {}
    for i in range(len(records)):
        groups.setdefault(root(i), []).append(i)
    instance = np.empty(len(records), dtype=np.int64)
    for members in groups.values():
        instance[members] = min(members, key=lambda i: records[i].masklet_id)
    return instance
