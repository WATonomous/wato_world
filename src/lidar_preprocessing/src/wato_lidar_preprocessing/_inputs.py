"""Shared I/O helpers used by deskew and mf_mos."""

from __future__ import annotations

import json
import logging

import numpy as np

from wato_common.artifact_store import (
    calibration_path,
    frame_index_path,
    local_path,
    poses_path,
)
from wato_common.geometry import PoseSample, unflatten_se3
from wato_common.io.parquet_io import read_rows

log = logging.getLogger(__name__)


def load_pose_valid_sweep_ids(bag_id: str, chunk_id: str) -> set[int] | None:
    """Sweep_ids that ingest marked pose-valid in frame_index.parquet.

    frame_index carries one row per (sweep_id, cam_id) with a per-sweep
    ``valid_pose`` flag: ingest sets it False when no ego pose could be
    interpolated within ``max_pose_gap_ms`` of the sweep timestamp (see ingest's
    frame_index / pose_interpolation). Deskew honors this so the transient
    window at the start of a bag — before SLAM converges and emits poses — is
    skipped rather than silently clamped to the nearest available pose by
    ``batch_interpolate_poses``.

    Returns None when frame_index.parquet is absent (older artifacts), signalling
    callers to fall back to processing every sweep. A sweep counts as valid if
    any of its rows is pose-valid (all cam rows of a sweep share the flag).
    """
    try:
        rows = read_rows(frame_index_path(bag_id, chunk_id))
    except FileNotFoundError:
        log.warning(
            "chunk %s has no frame_index.parquet; cannot honor valid_pose, "
            "processing all sweeps",
            chunk_id,
        )
        return None
    return {int(r["sweep_id"]) for r in rows if r.get("valid_pose", False)}


def load_pose_samples(bag_id: str, chunk_id: str) -> list[PoseSample]:
    """Load valid poses from poses.parquet and return sorted by timestamp."""
    rows = read_rows(poses_path(bag_id, chunk_id))
    samples: list[PoseSample] = []
    for r in rows:
        if not r.get("valid", True):
            continue
        flat = r["world_T_ego_flat"]
        T = unflatten_se3(flat)
        # Matrix translation must agree with the (x,y,z) columns. Disagreement
        # = broken upstream writer; we trust the matrix and warn.
        xyz = np.array([r["x"], r["y"], r["z"]], dtype=np.float64)
        if not np.allclose(T[:3, 3], xyz, atol=1e-6):
            log.warning(
                "pose row at t=%d: world_T_ego translation %s != (x,y,z)=%s",
                int(r["timestamp_ns"]),
                T[:3, 3].tolist(),
                xyz.tolist(),
            )
        samples.append(
            PoseSample(
                timestamp_ns=int(r["timestamp_ns"]),
                translation=T[:3, 3].copy(),
                quat_xyzw=np.array(
                    [r["qx"], r["qy"], r["qz"], r["qw"]], dtype=np.float64
                ),
            )
        )
    samples.sort(key=lambda s: s.timestamp_ns)
    return samples


def load_ego_T_lidar(bag_id: str, lidar_id: str) -> np.ndarray:
    """Return the 4×4 ego_T_lidar extrinsic from calibration.json."""
    calib_file = local_path(calibration_path(bag_id))
    with open(calib_file, "r", encoding="utf-8") as fh:
        calib = json.load(fh)
    lidar_entry = calib.get("lidars", {}).get(lidar_id)
    if lidar_entry is None:
        raise KeyError(f"lidar {lidar_id!r} not found in calibration.json")
    if lidar_entry.get("ego_T_lidar") is None:
        raise ValueError(
            f"lidar {lidar_id!r} has null ego_T_lidar in calibration.json "
            "(extrinsic unresolved — check /tf_static coverage)"
        )
    return np.asarray(lidar_entry["ego_T_lidar"], dtype=np.float64)


def load_ego_T_lidar_dict(bag_id: str, lidar_ids: set[str]) -> dict[str, np.ndarray]:
    """Load ego_T_lidar for multiple lidar IDs from one calibration.json read.

    Missing / unresolved lidar_ids are logged as errors and absent from the
    returned dict; callers should skip those sweeps.
    """
    calib_file = local_path(calibration_path(bag_id))
    with open(calib_file, "r", encoding="utf-8") as fh:
        calib = json.load(fh)
    result: dict[str, np.ndarray] = {}
    for lid in lidar_ids:
        try:
            lidar_entry = calib.get("lidars", {}).get(lid)
            if lidar_entry is None:
                raise KeyError(f"lidar {lid!r} not found in calibration.json")
            if lidar_entry.get("ego_T_lidar") is None:
                raise ValueError(
                    f"lidar {lid!r} has null ego_T_lidar in calibration.json "
                    "(extrinsic unresolved — check /tf_static coverage)"
                )
            result[lid] = np.asarray(lidar_entry["ego_T_lidar"], dtype=np.float64)
        except (KeyError, ValueError) as exc:
            log.error("lidar %s: %s", lid, exc)
    return result
