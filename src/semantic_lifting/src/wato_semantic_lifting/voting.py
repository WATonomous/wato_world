"""Cross-camera vote accumulation for LiDAR point labeling.

Each camera casts a vote for each visible LiDAR point: (instance_id, class, score).
This module resolves disagreements and produces per-point labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PointVote:
    """A single camera's label vote for one LiDAR point."""

    point_idx: int
    instance_id: str   # global_object_id from masklet
    cls: str
    score: float       # sam3_score * depth_confidence
    cam_id: str
    mask_area: int     # pixel count of the source mask


@dataclass
class LabeledPoint:
    """Resolved label for one LiDAR point."""

    point_idx: int
    instance_id: str
    cls: str
    confidence: float
    n_supporting_cameras: int
    n_disagreeing_cameras: int


def accumulate_votes(
    votes: list[PointVote],
    n_points: int,
) -> list[LabeledPoint]:
    """Resolve per-point votes from multiple cameras into final labels.

    Resolution rules (per design doc):
    - Overlap: innermost mask (smallest area) wins within a single camera.
    - Cross-camera disagreement: highest cumulative score wins.
    - Confidence = winning_score / (winning_score + runner_up_score), or 1.0
      if only one instance voted.

    Args:
        votes: all votes from all cameras for this sweep.
        n_points: total number of LiDAR points (for indexing).

    Returns:
        List of LabeledPoint for every point that received at least one vote.
    """
    # Group votes by point index.
    by_point: dict[int, list[PointVote]] = {}
    for v in votes:
        by_point.setdefault(v.point_idx, []).append(v)

    results: list[LabeledPoint] = []
    for pt_idx, pt_votes in by_point.items():
        # Aggregate scores per (instance_id, cls) pair.
        score_map: dict[tuple[str, str], float] = {}
        cam_set: dict[tuple[str, str], set[str]] = {}
        for v in pt_votes:
            key = (v.instance_id, v.cls)
            score_map[key] = score_map.get(key, 0.0) + v.score
            cam_set.setdefault(key, set()).add(v.cam_id)

        # Pick winner by cumulative score.
        sorted_keys = sorted(score_map, key=lambda k: score_map[k], reverse=True)
        winner_key = sorted_keys[0]
        winner_score = score_map[winner_key]
        runner_up_score = score_map[sorted_keys[1]] if len(sorted_keys) > 1 else 0.0

        confidence = (
            winner_score / (winner_score + runner_up_score)
            if runner_up_score > 0
            else 1.0
        )

        n_supporting = len(cam_set[winner_key])
        n_disagreeing = len({k for k in sorted_keys if k != winner_key})

        results.append(
            LabeledPoint(
                point_idx=pt_idx,
                instance_id=winner_key[0],
                cls=winner_key[1],
                confidence=float(confidence),
                n_supporting_cameras=n_supporting,
                n_disagreeing_cameras=n_disagreeing,
            )
        )

    return results
