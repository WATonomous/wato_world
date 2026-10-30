"""Shared I/O helpers."""

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

    Deskew skips the others rather than let ``batch_interpolate_poses`` clamp
    them to the nearest pose. None when frame_index.parquet is absent: callers
    then process every sweep.
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
    """Valid poses from poses.parquet (via wato_common.pose_lookup), sorted by time."""
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
