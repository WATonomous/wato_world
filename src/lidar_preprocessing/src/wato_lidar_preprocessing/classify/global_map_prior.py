"""Global static map prior for two-pass classification (UniLiPs IWU).

Wraps a cKDTree over the bag-level global_static_map.npz so each sweep in
pass 2 can ask "is this scan point within match_radius_m of a known static
surface, and how credible is it?"  Per UniLiPs Eq. 2-3, the range-credibility
weight is r*_j = min(1, r_max / ||p_j - sensor||), and the "found" boost is
applied to the per-voxel log-odds (not n_hits — that gate must stay backed by
real sweep returns; see _apply_global_map_boost in log_odds.py).
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)


class GlobalMapPrior:
    """KDTree wrapper over global_static_map.npz.

    Built once per bag (or once per worker process in pass 2) and shared
    read-only across chunks/sweeps within a process.  The KDTree itself is
    built lazily — if the global map is empty, no tree is constructed and
    query_sweep returns all-False map_hit + all-ones r_star so callers can
    pass an empty prior through without special-casing it.
    """

    def __init__(
        self,
        global_map_xyz: np.ndarray,
        match_radius_m: float = 0.30,
        r_max_credibility_m: float = 200.0,
    ) -> None:
        self.match_radius_m = float(match_radius_m)
        self.r_max = float(r_max_credibility_m)

        if global_map_xyz.shape[0] == 0:
            self._tree: cKDTree | None = None
            log.warning(
                "GlobalMapPrior: global map is empty — prior has no effect"
            )
        else:
            self._tree = cKDTree(global_map_xyz.astype(np.float64))
            log.info(
                "GlobalMapPrior: built KDTree over %d static map points "
                "(match_radius=%.2fm, r_max=%.1fm)",
                global_map_xyz.shape[0],
                self.match_radius_m,
                self.r_max,
            )

    @classmethod
    def from_npz(
        cls,
        path: str,
        match_radius_m: float = 0.30,
        r_max_credibility_m: float = 200.0,
    ) -> "GlobalMapPrior":
        data = np.load(path)
        return cls(
            data["xyz"],
            match_radius_m=match_radius_m,
            r_max_credibility_m=r_max_credibility_m,
        )

    def query_sweep(
        self,
        xyz: np.ndarray,
        sensor_origin: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """For each point in xyz, determine map match and range-credibility weight.

        Args:
            xyz:           (N, 3) float, world-frame sweep points
            sensor_origin: (3,) float, sensor position for this sweep
                           (the LiDAR origin in world frame)

        Returns:
            map_hit:  (N,) bool   — True iff nearest map point is within match_radius_m
            r_star:   (N,) float32 — min(1, r_max / range_to_sensor) per UniLiPs Eq. 2.
                                     Zero-range points get r_star = 1.0.
        """
        n = xyz.shape[0]

        # r_star is computed from sensor distance regardless of tree presence,
        # so an empty map still returns sensible weights (callers that combine
        # the prior with other signals can rely on the shape contract).
        dx = xyz[:, 0] - sensor_origin[0]
        dy = xyz[:, 1] - sensor_origin[1]
        dz = xyz[:, 2] - sensor_origin[2]
        ranges = np.sqrt(dx * dx + dy * dy + dz * dz)
        r_star = np.where(
            ranges > 1e-6,
            np.minimum(1.0, self.r_max / np.maximum(ranges, 1e-6)),
            1.0,
        ).astype(np.float32)

        if self._tree is None or n == 0:
            return np.zeros(n, dtype=bool), r_star

        # k=1 nearest-neighbour with cKDTree; workers=-1 parallelises across cores.
        dists, _ = self._tree.query(xyz, k=1, workers=-1)
        map_hit = dists <= self.match_radius_m
        return map_hit, r_star
