"""Cross-camera vote resolution for LiDAR point labeling.

Each camera casts at most one vote per visible LiDAR point: the masklet whose
mask the point lands in (innermost mask wins, projection.label_raster),
weighted by that masklet's det_score. Votes are resolved per point:

  1. class: the class with the highest summed score across cameras wins.
     Masklets are per-camera, so cameras agree on a class, not on a masklet;
     cross-camera identity (instances.py) is not required for agreement.
  2. instance: among the winning class's votes, the instance (merged masklet
     group) with the highest summed score.

confidence = winning class score / (winning + runner-up class score), or 1.0
when every vote is for one class. Ties go to the lower class / instance index.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Votes:
    """One sweep's votes, one per (point, camera)."""

    point_idx: np.ndarray  # int64 (V,) index into the sweep's world NPZ
    cam_idx: np.ndarray  # int64 (V,) the voting camera (only distinctness matters)
    label_idx: np.ndarray  # int64 (V,) masklet index (MaskletIndex.records order)
    score: np.ndarray  # float64 (V,) the masklet's det_score

    @classmethod
    def concat(cls, parts: list["Votes"]) -> "Votes":
        if not parts:
            return cls.empty()
        return cls(
            *(
                np.concatenate([getattr(p, f) for p in parts])
                for f in ("point_idx", "cam_idx", "label_idx", "score")
            )
        )

    @classmethod
    def empty(cls) -> "Votes":
        z = np.zeros(0, np.int64)
        return cls(z, z, z, np.zeros(0, np.float64))

    def __len__(self) -> int:
        return int(self.point_idx.shape[0])


@dataclass(frozen=True)
class Resolved:
    """One label per voted point, ascending point_idx."""

    point_idx: np.ndarray  # int64 (M,)
    instance: np.ndarray  # int64 (M,) instance index
    cls: np.ndarray  # int64 (M,) class index
    confidence: np.ndarray  # float64 (M,)
    n_supporting_cameras: np.ndarray  # int64 (M,) cameras voting the winning class
    n_disagreeing_cameras: np.ndarray  # int64 (M,) cameras voting another class


def _group_sums(
    point: np.ndarray, key: np.ndarray, score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum score per (point, key): the unique (point, key) pairs (sorted by
    point), their score sums and their vote counts."""
    pairs, inv = np.unique(np.stack([point, key], axis=1), axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    return pairs, np.bincount(inv, weights=score), np.bincount(inv)


def _best_per_point(
    pairs: np.ndarray, sums: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each point: index of its highest-sum group (ties → lower key), and
    that of the runner-up (-1 when none). Returns (points, best, runner_up)."""
    order = np.lexsort((pairs[:, 1], -sums, pairs[:, 0]))
    pts = pairs[order, 0]
    first = np.flatnonzero(np.r_[True, pts[1:] != pts[:-1]])
    second = first + 1
    has_second = np.zeros(first.shape, bool)
    in_range = second < len(order)
    has_second[in_range] = pts[second[in_range]] == pts[first[in_range]]
    runner = np.full(first.shape, -1, np.int64)
    runner[has_second] = order[second[has_second]]
    return pts[first], order[first], runner


def resolve(
    votes: Votes, label_cls: np.ndarray, label_instance: np.ndarray
) -> Resolved:
    """Resolve one sweep's votes into per-point labels.

    Args:
        votes: every camera's votes for this sweep.
        label_cls: (L,) class index of each label (masklet).
        label_instance: (L,) instance index of each label; masklets merged
            across cameras share one.
    """
    if len(votes) == 0:
        z = np.zeros(0, np.int64)
        return Resolved(z, z, z, np.zeros(0, np.float64), z, z)
    pc = np.stack([votes.point_idx, votes.cam_idx], axis=1)
    if len(np.unique(pc, axis=0)) != len(votes):
        raise ValueError("votes must hold at most one vote per camera per point")

    cls = label_cls[votes.label_idx]
    pairs, sums, counts = _group_sums(votes.point_idx, cls, votes.score)
    points, best, runner = _best_per_point(pairs, sums)
    win_sum = sums[best]
    run_sum = np.where(runner >= 0, sums[np.maximum(runner, 0)], 0.0)
    confidence = np.where(run_sum > 0, win_sum / (win_sum + run_sum), 1.0)
    vote_point = np.searchsorted(points, votes.point_idx)
    n_votes = np.bincount(vote_point)
    n_support = counts[best]
    win_cls = pairs[best, 1]

    # Instance: only the winning class's votes compete.
    keep = cls == win_cls[vote_point]
    inst = label_instance[votes.label_idx[keep]]
    ipairs, isums, _ = _group_sums(votes.point_idx[keep], inst, votes.score[keep])
    _, ibest, _ = _best_per_point(ipairs, isums)  # same points, same order

    return Resolved(
        point_idx=points,
        instance=ipairs[ibest, 1],
        cls=win_cls,
        confidence=confidence,
        n_supporting_cameras=n_support,
        n_disagreeing_cameras=n_votes - n_support,
    )
