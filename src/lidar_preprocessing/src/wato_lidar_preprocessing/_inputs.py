"""Shared I/O helpers used by deskew and mf_mos."""

from __future__ import annotations

import logging

import numpy as np

from wato_common.artifact_store import frame_index_path
from wato_common.artifacts.calibration import Calibration
from wato_common.geometry import PoseSample
from wato_common.io.parquet_io import read_rows
from wato_common.pose_lookup import PoseLookup

log = logging.getLogger(__name__)


def load_pose_valid_sweep_ids(bag_id: str, chunk_id: str) -> set[int] | None:
    """Sweep_ids that ingest marked pose-valid in frame_index.parquet.

    frame_index carries one row per (sweep_id, cam_id) with a per-sweep
    ``valid_pose`` flag: ingest sets it False when the sweep falls outside the
    pose span, the two poses around it are more than
    ``pose_requirements.max_bracket_ms`` apart, or they imply a pose jump (see
    wato_common.pose_lookup and ingest README "Pose requirements"). Deskew honors
    this so e.g. the window at the start of a bag — before SLAM converges and
    emits poses — is skipped rather than silently clamped to the nearest
    available pose by ``batch_interpolate_poses``.

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
    """Valid poses from poses.parquet, sorted by timestamp.

    Read through wato_common.pose_lookup, the loader every component shares.
    """
    return PoseLookup.load(bag_id, chunk_id).samples


def load_ego_T_lidar_dict(bag_id: str, lidar_ids: set[str]) -> dict[str, np.ndarray]:
    """ego_T_lidar for each LiDAR, from the bag's calibration.

    Missing / unresolved lidar_ids are logged as errors and absent from the
    returned dict; callers skip those sweeps.
    """
    calib = Calibration.load(bag_id)
    result: dict[str, np.ndarray] = {}
    for lid in lidar_ids:
        try:
            result[lid] = calib.ego_T_lidar(lid)
        except (KeyError, ValueError) as exc:
            log.error("lidar %s: %s", lid, exc)
    return result
