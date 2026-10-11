"""Artifact I/O helpers for perception_2d.

Reads upstream artifacts from ingest + lidar_preprocessing. Calibration and
processed LiDAR sweeps are read, and perception_2d's own outputs written,
through wato_common.artifacts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from wato_common.artifact_store import (
    chunks_index_path,
    frame_index_path,
    local_path,
)
from wato_common.artifacts.lidar_sweeps import ProcessedSweeps
from wato_common.io.parquet_io import read_rows

log = logging.getLogger(__name__)


@dataclass
class CameraFrameInfo:
    """One camera frame entry from frame_index.parquet."""

    frame_id: str
    bag_id: str
    chunk_id: str
    sweep_id: int
    cam_id: str
    image_path: str
    camera_seq: int
    # When the image was taken.  The ego pose for projecting into this image is
    # looked up at this time (wato_common.pose_lookup), not taken from
    # frame_index's world_T_ego, which is the LiDAR sweep's pose.
    camera_timestamp_ns: Optional[int]
    valid_camera: bool


def load_chunks(bag_id: str) -> list[dict]:
    return read_rows(chunks_index_path(bag_id))


def load_frame_index(bag_id: str, chunk_id: str) -> list[CameraFrameInfo]:
    rows = read_rows(frame_index_path(bag_id, chunk_id))
    result = []
    for r in rows:
        if not r.get("valid_camera", False):
            continue
        cam_ts = r.get("camera_timestamp_ns")
        result.append(
            CameraFrameInfo(
                frame_id=str(r.get("frame_id", "")),
                bag_id=bag_id,
                chunk_id=chunk_id,
                sweep_id=int(r["sweep_id"]),
                cam_id=str(r["cam_id"]),
                image_path=local_path(str(r["image_path"])),
                camera_seq=int(r["camera_seq"]),
                camera_timestamp_ns=int(cam_ts) if cam_ts is not None else None,
                valid_camera=bool(r.get("valid_camera", False)),
            )
        )
    return result


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
