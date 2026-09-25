"""Artifact I/O helpers for lidar_preprocessing.

Thin wrappers over artifact_store path functions so callers don't need to
import from wato_common directly.
"""

from __future__ import annotations

import numpy as np

from wato_common.artifact_store import (
    dynamic_map_path,
    dynamic_mask_path,
    global_iwu_path,
    global_static_map_path,
    ground_path,
    lidar_world_path,
    local_path,
    motion_clusters_path,
    motion_proposals_path,
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


def load_motion_proposals(
    bag_id: str, chunk_id: str, sweep_id: int
) -> dict[str, np.ndarray]:
    """Load Step F's per-sweep motion proposals (aligned to the world NPZ).

    Keys:
      source_bits uint8 (N,) — which heuristics flagged the point; decode with
                               wato_lidar_preprocessing.motion_proposals.decode_bits
      cluster_id  int32 (N,) — row in motion_clusters.parquet, −1 = none

    Recall-oriented and false-positive tolerant. Use dynamic_mask.npy, not
    this, wherever a point must be trusted static (e.g. depth anchors).
    """
    return dict(np.load(local_path(motion_proposals_path(bag_id, chunk_id, sweep_id))))


def load_motion_clusters(bag_id: str, chunk_id: str) -> list[dict]:
    """Load Step F's per-chunk cluster rows (MotionClusterRow dicts)."""
    return read_rows(motion_clusters_path(bag_id, chunk_id))


def load_global_iwu(bag_id: str) -> dict[str, np.ndarray]:
    """Load Step E's bag-level IWU result.

    Keys: xyz (N,3), p_static, n_match, n_seen_through, evicted (bool), plus
    the constants it ran with (alpha, tau_static, update_rate_hz,
    n_sweeps_used).
    """
    return dict(np.load(local_path(global_iwu_path(bag_id))))
