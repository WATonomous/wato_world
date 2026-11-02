"""Per-sweep statistics helpers for semantic_lifting."""

from __future__ import annotations

from wato_common.schemas import LiftedStatsRow
from wato_semantic_lifting.voting import Resolved


def build_stats_row(
    bag_id: str,
    chunk_id: str,
    sweep_id: int,
    *,
    n_points_total: int,
    resolved: Resolved,
    n_points_in_any_mask: int,
    n_cameras_used: int,
) -> LiftedStatsRow:
    """The LiftedStatsRow for one sweep (field meanings: schemas.LiftedStatsRow)."""
    n_labeled = int(resolved.point_idx.shape[0])
    return LiftedStatsRow(
        sweep_id=sweep_id,
        bag_id=bag_id,
        chunk_id=chunk_id,
        n_points_total=n_points_total,
        n_points_labeled=n_labeled,
        n_points_in_any_mask=n_points_in_any_mask,
        n_points_failed_visibility=n_points_in_any_mask - n_labeled,
        n_points_disagreement=int((resolved.n_disagreeing_cameras > 0).sum()),
        mean_confidence_labeled=(
            float(resolved.confidence.mean()) if n_labeled else 0.0
        ),
        n_cameras_used=n_cameras_used,
    )
