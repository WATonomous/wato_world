"""Occlusion-aware visibility test (UniLiPs Eq.1).

A LiDAR point is considered visible from a camera if its projected depth
is no greater than the minimum depth in a neighborhood of the depth map
plus a tolerance:

    visible ⟺ d_cam ≤ min(D(N(u, v))) + tolerance_m

This test filters out LiDAR points that are occluded by closer surfaces
(e.g. a point on a wall behind a car).
"""

from __future__ import annotations

import numpy as np


def visibility_test(
    d_cam: np.ndarray,
    depth_map: np.ndarray,
    uv: np.ndarray,
    tolerance_m: float = 0.5,
    neighborhood: int = 3,
) -> np.ndarray:
    """Apply UniLiPs Eq.1 occlusion test.

    Args:
        d_cam: (N,) float32 camera-frame depths of projected LiDAR points.
        depth_map: (H, W) float32 metric depth map from depth_2d artifact.
        uv: (N, 2) float32 pixel coordinates of the projected points.
        tolerance_m: additive tolerance in metres (default 0.5 m).
        neighborhood: half-size of the patch for min-depth pooling (default 3,
            giving a 7×7 neighborhood).

    Returns:
        (N,) bool array — True for points that pass the visibility test.
    """
    H, W = depth_map.shape
    N = d_cam.shape[0]
    visible = np.zeros(N, dtype=bool)

    if N == 0:
        return visible

    ui = np.clip(np.round(uv[:, 0]).astype(int), 0, W - 1)
    vi = np.clip(np.round(uv[:, 1]).astype(int), 0, H - 1)

    # Precompute min-pooled depth map to avoid per-point nested loops.
    # Uses a sliding-window minimum via cumulative min over rows then cols.
    half = neighborhood
    min_depth = _min_pool(depth_map, half)

    local_min = min_depth[vi, ui]

    # Points where the depth map is zero (no measurement) are considered
    # non-occluded — the depth artifact may have gaps.
    no_measurement = local_min <= 0.0
    visible = (d_cam <= local_min + tolerance_m) | no_measurement
    return visible


def _min_pool(depth: np.ndarray, half: int) -> np.ndarray:
    """Return a map where each pixel holds the minimum in its (2*half+1)^2 patch."""
    from scipy.ndimage import minimum_filter

    try:
        return minimum_filter(depth, size=2 * half + 1, mode="nearest")
    except Exception:  # noqa: BLE001
        # Fallback: no pooling (point-wise depth lookup).
        return depth
