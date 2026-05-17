"""World-to-body-frame transforms for object-centric processing.

The body frame of a 3D bounding box has its origin at the box centre, its
x-axis aligned with the box's heading (yaw), y-axis perpendicular to heading
in the horizontal plane, and z-axis aligned with world z (so "up" is
preserved).  This is the convention assumed by 3DAL, DetZero, and
LabelFormer for per-object processing.

These helpers are vectorised over arrays of points so they're cheap to call
inside the per-track aggregation loop in `label_refinement`.
"""

from __future__ import annotations

import numpy as np


def heading_to_rotation(heading: float) -> np.ndarray:
    """Yaw-only 3x3 rotation around world z-axis (right-handed).

    A point on the world +x axis rotated by `heading` ends up at
    `(cos(h), sin(h), 0)` — i.e. heading is measured CCW from +x.  This
    matches the convention used by `frame_index.parquet:world_T_ego_flat`
    and by every 3D box parquet schema in `schemas.py`.
    """
    c, s = np.cos(heading), np.sin(heading)
    return np.array(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def world_to_body(
    pts_world: np.ndarray,
    center: np.ndarray,
    heading: float,
) -> np.ndarray:
    """Transform an (N, 3) world-frame point array into the box's body frame.

    body = R(-heading) @ (world - center).  Implemented as right-multiply by
    R so the (N, 3) layout stays contiguous.  Returns float64 to match the
    LiDAR world-frame precision in `lidar_preprocessing/deskew.py`.
    """
    if pts_world.ndim != 2 or pts_world.shape[1] != 3:
        raise ValueError(f"pts_world must be (N, 3); got {pts_world.shape}")
    center = np.asarray(center, dtype=np.float64).reshape(3)
    R = heading_to_rotation(heading)
    # (pts - c) @ R == R.T @ (pts - c).T -> (3,N) -> .T => same shape as input.
    return (np.asarray(pts_world, dtype=np.float64) - center) @ R


def body_to_world(
    pts_body: np.ndarray,
    center: np.ndarray,
    heading: float,
) -> np.ndarray:
    """Inverse of `world_to_body`."""
    if pts_body.ndim != 2 or pts_body.shape[1] != 3:
        raise ValueError(f"pts_body must be (N, 3); got {pts_body.shape}")
    center = np.asarray(center, dtype=np.float64).reshape(3)
    R = heading_to_rotation(heading)
    return np.asarray(pts_body, dtype=np.float64) @ R.T + center


def enlarged_box_indices(
    pts_world: np.ndarray,
    center: np.ndarray,
    size: np.ndarray,
    heading: float,
    margin: float = 1.5,
) -> np.ndarray:
    """Boolean mask (N,) of points inside an axis-aligned-in-body-frame box.

    The box is centred at `center`, rotated by `heading`, with extents
    `size * margin` (margin > 1 enlarges, < 1 shrinks).  3DAL/DetZero
    aggregation uses margin=1.5 to be tolerant of upstream-tracker pose
    noise while staying tight enough to exclude neighbouring objects.
    """
    if pts_world.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    body = world_to_body(pts_world, center, heading)
    half = np.asarray(size, dtype=np.float64).reshape(3) * (margin / 2.0)
    return (
        (np.abs(body[:, 0]) < half[0]) &
        (np.abs(body[:, 1]) < half[1]) &
        (np.abs(body[:, 2]) < half[2])
    )
