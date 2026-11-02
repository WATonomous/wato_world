"""Occlusion-aware visibility test (UniLiPs Eq. 1).

A LiDAR point is visible from a camera if nothing sits in front of it: its
camera depth is no greater than the nearest surface around its pixel plus a
tolerance,

    visible ⟺ d_cam ≤ min(D(N(u, v))) + tolerance

which stops a point on a wall behind a car from taking the car's label.

Two sources of D, because neither works everywhere
(docs/semantic_lifting_measurements.md):

- perception_2d's metric depth map: dense, but its single per-frame
  inverse-depth fit is anchored on mostly near LiDAR points, so its error
  grows ~d² (median 0.1 m at 5-10 m, 2 m at 20-30 m, 20 m at 30-50 m on the
  WATO rig). Only its surfaces nearer than depth_map_max_range_m occlude.
- the sweep's own LiDAR returns, projected into the image: exact at any
  range but sparse, so the nearest non-ground return within a small window
  of cells around the point's pixel, with a tolerance that grows with range
  (the window's footprint, and a surface's depth change across it, scale
  with d).

A point must pass both.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import minimum_filter

from wato_semantic_lifting.config import VisibilityConfig


def pooled_occluder(depth_m: np.ndarray, neighborhood: int) -> np.ndarray:
    """(H, W) float32: the nearest depth-map surface in each pixel's
    (2·neighborhood+1)² patch.

    depth_m == 0 is perception_2d's value for a non-positive fitted inverse
    depth — farther than the fit can represent (sky, far background) — so it
    is +inf here, never an occluder.
    """
    d = depth_m.astype(np.float32)
    d[d <= 0] = np.inf
    return minimum_filter(d, size=2 * neighborhood + 1, mode="nearest")


def depth_map_visible(
    d_cam: np.ndarray, occluder_m: np.ndarray, tolerance_m: float
) -> np.ndarray:
    """(N,) bool: Eq. 1 against the pooled depth map at each point's pixel."""
    return d_cam <= occluder_m + tolerance_m


def lidar_occluder(
    ui: np.ndarray,
    vi: np.ndarray,
    d_cam: np.ndarray,
    can_occlude: np.ndarray,
    size: tuple[int, int],
    cell_px: int,
    neighborhood_cells: int,
) -> np.ndarray:
    """(N,) float32: per projected point, the nearest LiDAR return among the
    occluding points in the (2·neighborhood_cells+1)² cells of cell_px pixels
    around it (+inf when there is none).

    Args:
        ui, vi: (N,) pixel column / row of every projected point of the sweep.
        d_cam: (N,) camera depths.
        can_occlude: (N,) bool; False for ground, which can't hide what stands
            on it but sits in front of its lower edge in the image.
        size: (W, H) of the image.
    """
    W, H = size
    gw, gh = -(-W // cell_px), -(-H // cell_px)
    cell = (vi // cell_px) * gw + (ui // cell_px)
    zbuf = np.full(gh * gw, np.inf, dtype=np.float32)
    np.minimum.at(zbuf, cell[can_occlude], d_cam[can_occlude].astype(np.float32))
    pooled = minimum_filter(
        zbuf.reshape(gh, gw), size=2 * neighborhood_cells + 1, mode="nearest"
    )
    return pooled.reshape(-1)[cell]


def visible(
    d_cam: np.ndarray,
    map_occluder_m: np.ndarray,
    lidar_occluder_m: np.ndarray,
    cfg: VisibilityConfig,
) -> np.ndarray:
    """(N,) bool: no trusted depth-map surface and no LiDAR return in front."""
    near_map = map_occluder_m < cfg.depth_map_max_range_m
    map_ok = ~near_map | depth_map_visible(d_cam, map_occluder_m, cfg.tolerance_m)
    lidar_ok = (
        d_cam <= lidar_occluder_m * (1.0 + cfg.lidar_tolerance_per_m) + cfg.tolerance_m
    )
    return map_ok & lidar_ok
