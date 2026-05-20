"""Running map-point accumulator for MapMOS inference.

MapMOS doesn't take a fixed N-past-sweeps window — it takes a current
scan plus a *running map* of registered points from prior scans, each
tagged with the scan index it came from. This module wraps PRBonn's
`VoxelHashMap` Python binding so our accumulator semantics match the
distribution the pretrained model was trained on.

The binding's public Python wrapper (`mapmos.mapping.VoxelHashMap`)
only exposes the pose-overload of `Update`, which transforms input
points from sensor frame to world frame. Since our deskew step already
puts points in world frame, we reach into the private `_internal_map`
attribute to call the origin-overload (which assumes world-frame input
and uses `origin` only for far-point pruning).

API verified against PRBonn/MapMOS @ commit
8947300698c61257ddb1e1e9f927382f0c0a0bac, 2025-09-11:
  - src/mapmos/mapping.py:28-66 (Python wrapper)
  - src/mapmos/pybind/VoxelHashMap.cpp:91-106 (two Update overloads)
  - src/mapmos/pybind/mapmos_pybind.cpp:65-83 (binding registration)
  - src/mapmos/odometry.py:95-97 (get_map_points reference shape)
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


# Defaults match PRBonn's OdometryConfig (config/config.py:34-38).
# These govern the map storage density and pruning radius — they are
# INDEPENDENT of the MinkUNet inference voxel size (which is locked at
# EXPECTED_VOXEL_SIZE_M = 0.1m in weights.py).
DEFAULT_VOXEL_SIZE_M: float = 0.5
DEFAULT_MAX_RANGE_M: float = 100.0
DEFAULT_MAX_POINTS_PER_VOXEL: int = 20


class MapAccumulator:
    """Running map of registered points + per-point scan_index labels.

    Scope: one accumulator per chunk. The instance is constructed at the
    top of mapmos.pipeline.process_chunk, fed scan-by-scan in sweep
    order, and discarded when the chunk finishes. For multi-chunk bags
    with continuous trajectories, the first sweep of chunk N+1 starts
    with an empty accumulator — re-process the last K sweeps of chunk N
    before running inference if you need warm history at the boundary
    (plan §Step 3 Phase 5).

    NOT thread-safe. Sweep processing is sequential within a chunk today.
    """

    def __init__(
        self,
        voxel_size: float = DEFAULT_VOXEL_SIZE_M,
        max_range_m: float = DEFAULT_MAX_RANGE_M,
        max_points_per_voxel: int = DEFAULT_MAX_POINTS_PER_VOXEL,
    ):
        # Lazy import — geometry-only code paths must not require the
        # mapmos pybind binding to be installed (plan non-negotiable #27).
        from mapmos.mapping import VoxelHashMap

        self._vhm = VoxelHashMap(
            voxel_size=voxel_size,
            max_distance=max_range_m,
            max_points_per_voxel=max_points_per_voxel,
        )
        self._voxel_size = voxel_size
        self._max_range_m = max_range_m

    def add_scan(
        self,
        world_xyz: np.ndarray,
        ego_world_xyz: np.ndarray,
        scan_index: int,
    ) -> None:
        """Register one scan into the accumulator.

        Args:
            world_xyz:     (N, 3) float64 — already-deskewed world-frame
                           points. NOT range-filtered; the VoxelHashMap
                           prunes voxels too far from `ego_world_xyz`
                           internally.
            ego_world_xyz: (3,)  float64 — sensor world-frame position
                           at scan time (the `origin` field from the
                           deskew world NPZ).
            scan_index:    integer scan index. Stored alongside each
                           point so get_map_points can return it later.

        Uses the origin-overload of VoxelHashMap.Update (binding's `_update`
        with a 3-vector origin, not the 4x4 pose overload). The pose
        overload would multiply our already-world-frame points by another
        transform — wrong.
        """
        # Lazy import of the pybind helper that converts np arrays to
        # the C++ std::vector<Eigen::Vector3d> the binding expects.
        from mapmos.pybind import mapmos_pybind

        if world_xyz.size == 0:
            return
        # The C++ binding dispatches by argument type: 3-element vector
        # -> origin-overload; 4x4 matrix -> pose-overload. We pass a
        # (3,) numpy array so it binds to the former.
        ego = np.asarray(ego_world_xyz, dtype=np.float64).reshape(3)
        # noinspection PyProtectedMember
        self._vhm._internal_map._update(
            mapmos_pybind._Vector3dVector(np.asarray(world_xyz, dtype=np.float64)),
            ego,
            int(scan_index),
        )

    def get_map_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (points_world_frame, scan_indices).

        Shapes:
            points:        (M, 3) float64
            scan_indices:  (M,)   int (downcast from the binding's stored
                                       integer timestamp)

        Both arrays are aligned 1:1 — the i-th index corresponds to the
        i-th point. Suitable to feed directly into MapMOSNet.predict as
        `map_input` + `map_indices`.
        """
        points, timestamps = self._vhm.point_cloud_with_timestamps()
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        timestamps = np.asarray(timestamps).reshape(-1).astype(np.int64)
        return points, timestamps

    def prune(self, ego_world_xyz: np.ndarray) -> None:
        """Evict voxels far from the current ego position.

        Mostly a no-op when add_scan already pruned (since the
        origin-overload of Update also prunes). Exposed separately for
        callers that want to prune without adding new points (e.g.,
        between chunks).
        """
        ego = np.asarray(ego_world_xyz, dtype=np.float64).reshape(3)
        self._vhm.remove_voxels_far_from_location(ego)

    def empty(self) -> bool:
        return bool(self._vhm.empty())

    def clear(self) -> None:
        self._vhm.clear()
