"""Artifact I/O helpers for perception_2d.

Reads upstream artifacts from ingest + lidar_preprocessing.
Writes: detections_2d.parquet, tracklets_2d.parquet, masks_2d/ directory.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from wato_common.artifact_store import (
    calibration_path,
    chunks_index_path,
    detections_2d_path,
    ensure_local_dir,
    frame_index_path,
    lidar_proc_index_path,
    local_path,
    masks_2d_dir,
    tracklets_2d_path,
)
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import MASKLET_SCHEMA


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
    world_T_ego_flat: Optional[list[float]]  # row-major 4×4, None if invalid
    valid_camera: bool
    valid_pose: bool


@dataclass
class CalibrationInfo:
    """Per-camera calibration loaded from calibration.json."""

    K: np.ndarray  # (3, 3) float64 intrinsic
    ego_T_cam: np.ndarray  # (4, 4) float64 SE(3)


def load_chunks(bag_id: str) -> list[dict]:
    return read_rows(chunks_index_path(bag_id))


def load_frame_index(bag_id: str, chunk_id: str) -> list[CameraFrameInfo]:
    rows = read_rows(frame_index_path(bag_id, chunk_id))
    result = []
    for r in rows:
        if not r.get("valid_camera", False):
            continue
        flat = r.get("world_T_ego_flat")
        if flat is not None:
            flat = list(flat)
        result.append(
            CameraFrameInfo(
                frame_id=str(r.get("frame_id", "")),
                bag_id=bag_id,
                chunk_id=chunk_id,
                sweep_id=int(r["sweep_id"]),
                cam_id=str(r["cam_id"]),
                image_path=local_path(str(r["image_path"])),
                camera_seq=int(r["camera_seq"]),
                world_T_ego_flat=flat,
                valid_camera=bool(r.get("valid_camera", False)),
                valid_pose=bool(r.get("valid_pose", False)),
            )
        )
    return result


def load_calibration(bag_id: str) -> dict[str, CalibrationInfo]:
    """Load per-camera K and ego_T_cam from calibration.json."""
    calib_file = local_path(calibration_path(bag_id))
    with open(calib_file, "r", encoding="utf-8") as fh:
        calib = json.load(fh)
    result: dict[str, CalibrationInfo] = {}
    for cam_id, entry in calib.get("cameras", {}).items():
        K = np.asarray(entry["K"], dtype=np.float64)
        ego_T_cam = np.asarray(entry["ego_T_cam"], dtype=np.float64)
        result[cam_id] = CalibrationInfo(K=K, ego_T_cam=ego_T_cam)
    return result


# Per-chunk caches so the lidar-proc index isn't re-read (and sweeps aren't
# re-loaded) once per camera-frame. On nuScenes each sweep is referenced by ~6
# cameras; without these the same parquet scan + npz load happens ~6×.
# Cleared per chunk via clear_lidar_caches() to bound memory.
_proc_index_cache: dict[tuple[str, str], dict[int, dict]] = {}
_static_points_cache: dict[tuple[str, str, int], Optional[np.ndarray]] = {}


def _proc_index(bag_id: str, chunk_id: str) -> dict[int, dict]:
    """Return {sweep_id: row} for a chunk's lidar-proc index, built once."""
    key = (bag_id, chunk_id)
    cached = _proc_index_cache.get(key)
    if cached is None:
        cached = {}
        for r in read_rows(lidar_proc_index_path(bag_id, chunk_id)):
            cached.setdefault(int(r["sweep_id"]), r)  # first-match, as before
        _proc_index_cache[key] = cached
    return cached


def _load_lidar_proc_row(
    bag_id: str, chunk_id: str, sweep_id: int
) -> Optional[dict]:
    return _proc_index(bag_id, chunk_id).get(sweep_id)


def clear_lidar_caches(bag_id: str, chunk_id: str) -> None:
    """Drop this chunk's cached lidar index and static points (call at chunk end)."""
    _proc_index_cache.pop((bag_id, chunk_id), None)
    stale = [
        k for k in _static_points_cache
        if k[0] == bag_id and k[1] == chunk_id
    ]
    for k in stale:
        _static_points_cache.pop(k, None)


def load_dynamic_lidar_points(
    bag_id: str, chunk_id: str, sweep_id: int
) -> Optional[np.ndarray]:
    """Return (N, 3) float64 world-frame dynamic points for one sweep, or None."""
    row = _load_lidar_proc_row(bag_id, chunk_id, sweep_id)
    if row is None or row.get("valid") is False:
        return None
    world_p = local_path(str(row["world_path"]))
    dyn_p = local_path(str(row["dynamic_mask_path"]))
    if not os.path.exists(world_p) or not os.path.exists(dyn_p):
        return None
    data = np.load(world_p)
    mask = np.load(dyn_p)
    if mask.sum() == 0:
        return None
    xyz = np.stack([data["x"][mask], data["y"][mask], data["z"][mask]], axis=1)
    return xyz.astype(np.float64)


def load_static_lidar_points(
    bag_id: str, chunk_id: str, sweep_id: int
) -> Optional[np.ndarray]:
    """Return (N, 3) float64 world-frame static points for one sweep, or None.

    Static points are the complement of the dynamic mask — used as depth
    alignment anchors (dynamic points introduce ~75cm error at 25ms desync).

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
    row = _load_lidar_proc_row(bag_id, chunk_id, sweep_id)
    if row is None or row.get("valid") is False:
        return None
    world_p = local_path(str(row["world_path"]))
    dyn_p = local_path(str(row["dynamic_mask_path"]))
    if not os.path.exists(world_p) or not os.path.exists(dyn_p):
        return None
    data = np.load(world_p)
    dynamic_mask = np.load(dyn_p)
    static_mask = ~dynamic_mask
    if static_mask.sum() == 0:
        return None
    xyz = np.stack(
        [data["x"][static_mask], data["y"][static_mask], data["z"][static_mask]],
        axis=1,
    )
    return xyz.astype(np.float64)


def masks_2d_masklet_dir(bag_id: str, chunk_id: str, masklet_id: str) -> str:
    """Return the directory for one masklet's per-frame PNG files."""
    base = ensure_local_dir(masks_2d_dir(bag_id, chunk_id))
    d = os.path.join(base, masklet_id)
    os.makedirs(d, exist_ok=True)
    return d


def write_detections(bag_id: str, chunk_id: str, rows: list[dict]) -> None:
    write_table(rows, MASKLET_SCHEMA, detections_2d_path(bag_id, chunk_id))


def write_tracklets(bag_id: str, chunk_id: str, rows: list[dict]) -> None:
    write_table(rows, MASKLET_SCHEMA, tracklets_2d_path(bag_id, chunk_id))
