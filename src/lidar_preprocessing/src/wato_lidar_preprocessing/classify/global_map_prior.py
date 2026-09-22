"""Global static map prior for two-pass classification (UniLiPs IWU).

cKDTree over the bag-level global_static_map.npz: query_sweep returns, per
point, whether it matches a known static surface and its range to the sensor.
The caller turns range into a credibility weight (SensorModel.range_weight)
and applies a one-time prior shift (l_map_prior) to matched voxels — touching
log_odds only, so the has_hits gate stays backed by real returns.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)


class GlobalMapPrior:
    """KDTree wrapper over global_static_map.npz.

    Built once per bag (or once per worker process in pass 2). The KDTree
    is skipped when the map is empty — query_sweep then returns all-False
    map_hit so callers don't need to special-case it.
    """

    def __init__(
        self,
        global_map_xyz: np.ndarray,
        match_radius_m: float = 0.30,
    ) -> None:
        self.match_radius_m = float(match_radius_m)

        if global_map_xyz.shape[0] == 0:
            self._tree: cKDTree | None = None
            log.warning("GlobalMapPrior: global map is empty — prior has no effect")
        else:
            self._tree = cKDTree(global_map_xyz.astype(np.float64))
            log.info(
                "GlobalMapPrior: built KDTree over %d static map points "
                "(match_radius=%.2fm)",
                global_map_xyz.shape[0],
                self.match_radius_m,
            )

    @classmethod
    def from_npz(
        cls,
        path: str,
        match_radius_m: float = 0.30,
    ) -> "GlobalMapPrior":
        data = np.load(path)
        return cls(data["xyz"], match_radius_m=match_radius_m)

    def query_sweep(
        self,
        xyz: np.ndarray,
        sensor_origin: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """For each point, determine global-map match and range to the sensor.

        Args:
            xyz:           (N, 3) float, world-frame sweep points
            sensor_origin: (3,) float, sensor position for this sweep
                           (the LiDAR origin in world frame)

        Returns:
            map_hit: (N,) bool    — True iff nearest map point is within match_radius_m
            ranges:  (N,) float64 — Euclidean distance to sensor_origin. The
                                    caller converts this to a credibility weight
                                    via SensorModel.range_weight.
        """
        n = xyz.shape[0]

        dx = xyz[:, 0] - sensor_origin[0]
        dy = xyz[:, 1] - sensor_origin[1]
        dz = xyz[:, 2] - sensor_origin[2]
        ranges = np.sqrt(dx * dx + dy * dy + dz * dz)

        if self._tree is None or n == 0:
            return np.zeros(n, dtype=bool), ranges

        dists, _ = self._tree.query(xyz, k=1, workers=-1)
        map_hit = dists <= self.match_radius_m
        return map_hit, ranges
