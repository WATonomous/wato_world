"""World <-> query-sensor-frame transforms for MapMOS inference.

MapMOS was trained on sensor-frame inputs (a query scan stamped 0.0s plus
the previous N scans expressed in the query's sensor frame). Step A
produces world-frame NPZs, so inference needs the round trip.

Reuses `_load_ego_T_lidar` / `_load_pose_samples` from deskew so there's
exactly one place that knows how to read calibration + poses (plan: reuse
existing utilities).
"""

from __future__ import annotations

import logging

import numpy as np

from wato_common.geometry import PoseSample, batch_interpolate_poses
from wato_lidar_preprocessing.deskew import _load_ego_T_lidar, _load_pose_samples

log = logging.getLogger(__name__)

# Soft sanity bound. After transform, the centroid of a sensor-frame
# point cloud should sit near the LiDAR origin (≈ (0, 0, 0) plus mount
# noise). A centroid > 50 m away strongly suggests a wrong extrinsic.
# Plan non-negotiable #9.
_SENSOR_CENTROID_SANITY_M: float = 50.0


def load_pose_samples_for_chunk(bag_id: str, chunk_id: str) -> list[PoseSample]:
    """Load the per-chunk pose interpolation table ONCE per chunk.

    Plan non-negotiable #15: the table is identical for every sweep, so
    callers MUST load it before the sweep loop, not per sweep.
    """
    return _load_pose_samples(bag_id, chunk_id)


def compute_sensor_T_world(
    bag_id: str,
    sweep_row: dict,
    pose_samples: list[PoseSample],
) -> np.ndarray:
    """4x4 transform from world frame to the sweep's LiDAR sensor frame."""
    ego_T_lidar = _load_ego_T_lidar(bag_id, sweep_row["lidar_id"])
    world_T_ego_batch = batch_interpolate_poses(
        pose_samples,
        np.array([int(sweep_row["reference_timestamp_ns"])], dtype=np.int64),
    )
    world_T_ego = world_T_ego_batch[0]
    world_T_sensor = world_T_ego @ ego_T_lidar
    return np.linalg.inv(world_T_sensor)


def world_to_sensor(xyz_world: np.ndarray, sensor_T_world: np.ndarray) -> np.ndarray:
    """Apply 4x4 sensor_T_world to (N,3) world-frame points."""
    if xyz_world.size == 0:
        return xyz_world.astype(np.float64, copy=False)
    return (sensor_T_world[:3, :3] @ xyz_world.T).T + sensor_T_world[:3, 3]


def assert_sensor_centroid_sane(xyz_sensor: np.ndarray, sweep_row: dict) -> None:
    """Hard-fail when the sensor-frame centroid drifts far from the origin.

    Catches wrong-extrinsic mistakes early — on a multi-LiDAR rig it is
    very easy to pass the wrong lidar_id and silently rotate the entire
    cloud by 90 degrees or translate it by ~1 m. Plan non-negotiable #9.
    """
    if xyz_sensor.shape[0] == 0:
        return
    centroid = xyz_sensor.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm > _SENSOR_CENTROID_SANITY_M:
        raise RuntimeError(
            f"sensor-frame centroid {centroid.tolist()} (norm={norm:.2f}m) "
            f"is far from origin — likely wrong extrinsic for "
            f"lidar_id={sweep_row.get('lidar_id')!r} sweep_id={sweep_row.get('sweep_id')!r}"
        )
