"""Artifact I/O helpers for perception_2d.

Reads upstream artifacts from ingest + lidar_preprocessing. Calibration, the
camera frames (frame_index) and processed LiDAR sweeps are read, and
perception_2d's own outputs written, through wato_common.artifacts.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from wato_common.artifact_store import chunks_index_path
from wato_common.artifacts.lidar_sweeps import ProcessedSweeps
from wato_common.io.parquet_io import read_rows

log = logging.getLogger(__name__)


def load_chunks(bag_id: str) -> list[dict]:
    return read_rows(chunks_index_path(bag_id))


# Per-chunk caches so the lidar-proc index isn't re-read (and sweeps aren't
# re-loaded) once per camera-frame. On nuScenes each sweep is referenced by ~6
# cameras; without these the same parquet scan + npz load happens ~6×.
# Cleared per chunk via clear_lidar_caches() to bound memory.
_sweeps_cache: dict[tuple[str, str], ProcessedSweeps] = {}
_static_points_cache: dict[tuple[str, str, int], Optional[np.ndarray]] = {}


def _sweeps(bag_id: str, chunk_id: str) -> ProcessedSweeps:
    """The chunk's processed-sweep index, read once."""
    key = (bag_id, chunk_id)
    if key not in _sweeps_cache:
        _sweeps_cache[key] = ProcessedSweeps.load(bag_id, chunk_id)
    return _sweeps_cache[key]


def clear_lidar_caches(bag_id: str, chunk_id: str) -> None:
    """Drop this chunk's cached lidar index and static points (call at chunk end)."""
    _sweeps_cache.pop((bag_id, chunk_id), None)
    stale = [k for k in _static_points_cache if k[0] == bag_id and k[1] == chunk_id]
    for k in stale:
        _static_points_cache.pop(k, None)


def load_static_lidar_points(
    bag_id: str, chunk_id: str, sweep_id: int
) -> Optional[np.ndarray]:
    """Return (N, 3) float64 world-frame static points for one sweep, or None.

    Static points are the complement of lidar_preprocessing's dynamic mask —
    used as depth alignment anchors (dynamic points introduce ~75cm error at
    25ms desync).

    Cached per (bag, chunk, sweep) so cameras sharing a sweep don't each reload
    it. Callers must treat the returned array as read-only.
    """
    key = (bag_id, chunk_id, sweep_id)
    if key in _static_points_cache:
        return _static_points_cache[key]
    result = _compute_static_lidar_points(bag_id, chunk_id, sweep_id)
    _static_points_cache[key] = result
    return result


def _compute_static_lidar_points(
    bag_id: str, chunk_id: str, sweep_id: int
) -> Optional[np.ndarray]:
    sweeps = _sweeps(bag_id, chunk_id)
    if sweeps.get(sweep_id) is None:
        return None
    try:
        world, dynamic = sweeps.world_and_dynamic_mask(sweep_id)
    except FileNotFoundError as exc:
        log.warning("sweep %d: indexed but %s — no depth anchors", sweep_id, exc)
        return None
    static = ~dynamic
    if not static.any():
        return None
    return world.xyz[static]
