"""Per-chunk statistics helpers for semantic_lifting."""

from __future__ import annotations

from wato_common.schemas import LiftedStatsRow
from wato_semantic_lifting.voting import LabeledPoint


def build_stats_row(
    bag_id: str,
    chunk_id: str,
    sweep_id: int,
    n_points_total: int,
    labeled_points: list[LabeledPoint],
    n_points_in_any_mask: int,
    n_points_failed_visibility: int,
    n_cameras_used: int,
) -> dict:
    """Build a LiftedStatsRow dict for one sweep."""
    n_labeled = len(labeled_points)
    n_disagreement = sum(1 for p in labeled_points if p.n_disagreeing_cameras > 0)
    mean_conf = (
        float(sum(p.confidence for p in labeled_points) / n_labeled)
        if n_labeled > 0
        else 0.0
    )
    return LiftedStatsRow(
        sweep_id=str(sweep_id),
        bag_id=bag_id,
        chunk_id=chunk_id,
        n_points_total=n_points_total,
        n_points_labeled=n_labeled,
        n_points_in_any_mask=n_points_in_any_mask,
        n_points_failed_visibility=n_points_failed_visibility,
        n_points_disagreement=n_disagreement,
        mean_confidence_labeled=mean_conf,
        n_cameras_used=n_cameras_used,
    ).model_dump()
