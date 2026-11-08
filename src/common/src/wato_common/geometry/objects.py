"""Helpers for boxing one object's LiDAR points.

Shared by proposal_generation (boxes around lifted instances and Step F
clusters) and label_refinement (one box from a parked track's points over
every frame).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from wato_common.artifacts.ground import GroundGrid
from wato_common.schemas import Box3D


def largest_group(xyz: np.ndarray, radius_m: float) -> np.ndarray:
    """Indices of the largest set of points chained within radius_m."""
    n = xyz.shape[0]
    if n <= 1:
        return np.arange(n)
    pairs = cKDTree(xyz).query_pairs(radius_m, output_type="ndarray")
    graph = coo_matrix(
        (np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n)
    )
    _, labels = connected_components(graph, directed=False)
    return np.flatnonzero(labels == np.bincount(labels).argmax())


def to_ground(box: Box3D, ground: Optional[GroundGrid]) -> Box3D:
    """The box with its bottom lowered to the ground under its centre.

    Object points rarely reach the road: lifted labels exclude Patchwork++
    ground, and Step F's boxes stop at its height floor (0.25 m on WATO), so
    wheels and feet are cut off. The bottom is never raised. No grid (Step C
    found no ground for the chunk): the box is unchanged.
    """
    if ground is None:
        return box
    ground_z = float(ground.height_at(np.array([[box.cx, box.cy]]))[0])
    top = box.cz + box.h / 2.0
    bottom = min(box.cz - box.h / 2.0, ground_z)
    return Box3D(
        cx=box.cx,
        cy=box.cy,
        cz=(top + bottom) / 2.0,
        w=box.w,
        l=box.l,
        h=top - bottom,
        heading=box.heading,
    )
