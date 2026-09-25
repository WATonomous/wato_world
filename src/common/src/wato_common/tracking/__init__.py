"""Shared 3D box tracking primitives (geometry only).

One motion model and one association rule for every stage that reasons
about object motion — lidar_preprocessing's motion-proposal scoring today,
the `tracking` component later — so the two never disagree about what
"moved" means.
"""

from wato_common.tracking.association import (
    AssociationGates,
    MultiObjectTracker,
    Track,
    TrackObservation,
    associate,
    cost_terms,
)
from wato_common.tracking.boxes import (
    MIN_EXTENT_M,
    bev_corners,
    box_volume,
    fit_bev_box,
    iou_3d,
    points_in_box,
    wrap_heading_half_pi,
)
from wato_common.tracking.kalman import BoxKalmanFilter, KalmanNoise

__all__ = [
    "MIN_EXTENT_M",
    "AssociationGates",
    "BoxKalmanFilter",
    "KalmanNoise",
    "MultiObjectTracker",
    "Track",
    "TrackObservation",
    "associate",
    "bev_corners",
    "box_volume",
    "cost_terms",
    "fit_bev_box",
    "iou_3d",
    "points_in_box",
    "wrap_heading_half_pi",
]
