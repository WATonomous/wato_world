"""Returns too close to the LiDAR to belong to anything but the vehicle."""

from __future__ import annotations

import numpy as np


def close_to_sensor(xyz_sensor: np.ndarray, half_side_m: float) -> np.ndarray:
    """(N,) bool: points inside the axis-aligned square |x| < r, |y| < r of the
    sensor frame, any height.

    The nuScenes devkit's ``LidarPointCloud.remove_close(radius)`` (and
    mmdet3d's / OpenPCDet's copies of it) drops exactly these: returns off the
    roof, the sensor mount and the vehicle body. r <= 0 selects nothing.
    Non-finite points are never selected.
    """
    xyz = np.asarray(xyz_sensor)
    if xyz.ndim != 2 or xyz.shape[1] < 2:
        raise ValueError("xyz_sensor must be an (N, >=2) array")
    if half_side_m <= 0:
        return np.zeros(xyz.shape[0], dtype=bool)
    with np.errstate(invalid="ignore"):
        return (np.abs(xyz[:, 0]) < half_side_m) & (np.abs(xyz[:, 1]) < half_side_m)
