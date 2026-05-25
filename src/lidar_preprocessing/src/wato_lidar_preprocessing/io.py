"""Artifact I/O helpers for lidar_preprocessing.

Thin wrappers over artifact_store path functions so callers don't need to
import from wato_common directly.
"""

from __future__ import annotations

import numpy as np

from wato_common.artifact_store import (
    dynamic_map_path,
    dynamic_mask_path,
    global_ground_path,
    global_static_map_path,
    ground_path,
    lidar_proc_index_path,
    lidar_proc_summary_path,
    lidar_world_path,
    local_path,
    static_map_path,
    voxel_diag_path,
)
from wato_common.io.parquet_io import read_rows


def load_world_sweep(
    bag_id: str, chunk_id: str, sweep_id: int
) -> dict[str, np.ndarray]:
    """Load a deskewed world-frame sweep NPZ."""
    return dict(np.load(local_path(lidar_world_path(bag_id, chunk_id, sweep_id))))


def load_dynamic_mask(bag_id: str, chunk_id: str, sweep_id: int) -> np.ndarray:
    """Load per-point dynamic boolean mask (True = dynamic) for a sweep."""
    return np.load(local_path(dynamic_mask_path(bag_id, chunk_id, sweep_id)))


def load_static_map(bag_id: str, chunk_id: str) -> dict[str, np.ndarray]:
    """Load the static cloud NPZ for a chunk."""
    return dict(np.load(local_path(static_map_path(bag_id, chunk_id))))


def load_dynamic_map(bag_id: str, chunk_id: str) -> dict[str, np.ndarray]:
    """Load the per-chunk dynamic cloud NPZ.

    Keys:
      xyz       float64 (M, 3) — dynamic-classified world-frame points
      sweep_id  int32   (M,)   — originating sweep_id per point
      intensity float32 (M,)   — only present when any contributing sweep
                                 had intensity (mirrors static_map.npz).
    """
    return dict(np.load(local_path(dynamic_map_path(bag_id, chunk_id))))


def load_ground(bag_id: str, chunk_id: str) -> dict[str, np.ndarray]:
    """Load the ground height grid NPZ for a chunk."""
    return dict(np.load(local_path(ground_path(bag_id, chunk_id))))


def load_global_static_map(bag_id: str) -> np.ndarray:
    """Load the bag-level global static map; returns xyz (N,3) float64."""
    data = np.load(local_path(global_static_map_path(bag_id)))
    return data["xyz"]


def load_global_ground(bag_id: str) -> dict[str, np.ndarray]:
    """Load the bag-level ground height grid NPZ.

    Schema mirrors per-chunk ground.npz (height_grid, normal_grid, grid_origin,
    cell_size, ground_xyz) so consumers can use a single loader regardless of
    scope.  Produced by `wato_lidar_preprocessing reduce`.
    """
    return dict(np.load(local_path(global_ground_path(bag_id))))


def load_voxel_diag(bag_id: str, chunk_id: str) -> dict[str, np.ndarray]:
    """Load per-voxel diagnostics NPZ (full classification stats).

    Schema:
      keys           int64 (N,)     — packed voxel keys
      coords         int32 (N, 3)   — unpacked voxel indices
      origin         float64 (3,)   — chunk-level world-frame origin
      voxel_size     float32
      log_odds       float32 (N,)
      p_occ          float32 (N,)   — sigmoid(log_odds)
      n_obs          int32 (N,)
      n_hits         int32 (N,)
      classification int8 (N,)      — CLASS_* codes from occupancy_export

    Only written when cfg.save_voxel_diagnostics is True; otherwise raises
    FileNotFoundError.
    """
    return dict(np.load(local_path(voxel_diag_path(bag_id, chunk_id))))


def load_proc_index(bag_id: str, chunk_id: str) -> list[dict]:
    """Load the per-sweep processed sweep metadata for a chunk."""
    return read_rows(lidar_proc_index_path(bag_id, chunk_id))


def load_proc_summary(bag_id: str, chunk_id: str) -> dict:
    """Load the chunk-level rollup written by the pipeline.

    Single-row parquet, returned as a dict.  Raises FileNotFoundError if
    the chunk hasn't been processed yet.
    """
    rows = read_rows(lidar_proc_summary_path(bag_id, chunk_id))
    if not rows:
        raise FileNotFoundError(
            f"lidar_proc_summary for chunk {chunk_id!r} of bag {bag_id!r} is empty "
            "or does not exist."
        )
    return rows[0]
