"""Shared 3D box tracking primitives (geometry only).

One motion model and one association rule for every stage that reasons
about object motion — lidar_preprocessing's motion-proposal scoring and the
`tracking` component — so the two never disagree about what "moved" means.
tracking's extra options (GIoU cost, never-retiring tracks) are opt-in.
"""

from wato_common.tracking.association import (
    CHEN,
    GIOU_3D,
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
    bev_inside_fraction,
    box_corners_3d,
    box_volume,
    fit_bev_box,
    giou_3d,
    giou_3d_cannot_exceed,
    iou_3d,
    iou_bev,
    points_in_box,
    wrap_heading_half_pi,
)
from wato_common.tracking.kalman import BoxKalmanFilter, KalmanNoise

__all__ = [
    "CHEN",
    "GIOU_3D",
    "MIN_EXTENT_M",
    "AssociationGates",
    "BoxKalmanFilter",
    "KalmanNoise",
    "MultiObjectTracker",
    "Track",
    "TrackObservation",
    "associate",
    "bev_corners",
    "bev_inside_fraction",
    "box_corners_3d",
    "box_volume",
    "cost_terms",
    "fit_bev_box",
    "giou_3d",
    "giou_3d_cannot_exceed",
    "iou_3d",
    "iou_bev",
    "points_in_box",
    "wrap_heading_half_pi",
]
