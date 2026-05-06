"""Read poses.parquet → produce world_T_ego at requested timestamps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wato_common.artifact_store import poses_path
from wato_common.geometry import PoseSample, interpolate_pose
from wato_common.io.parquet_io import read_rows


@dataclass
class InterpolatedPose:
    target_timestamp_ns: int
    world_T_ego: np.ndarray  # 4x4
    interp_error_ns: float
    pose_timestamp_ns: int  # nearest source sample
    valid: bool


def load_samples(bag_id: str, chunk_id: str) -> list[PoseSample]:
    rows = read_rows(poses_path(bag_id, chunk_id))
    samples = [
        PoseSample(
            timestamp_ns=int(r["timestamp_ns"]),
            translation=np.array([r["x"], r["y"], r["z"]], dtype=np.float64),
            quat_xyzw=np.array([r["qx"], r["qy"], r["qz"], r["qw"]], dtype=np.float64),
        )
        for r in rows
        if r.get("valid", True)
    ]
    samples.sort(key=lambda s: s.timestamp_ns)
    return samples


def interpolate_at(
    samples: list[PoseSample],
    target_ts_ns: int,
    *,
    max_gap_ns: int = 200_000_000,  # 200 ms
) -> InterpolatedPose:
    """Estimate pose at a target timestamp.

    `valid=False` if the nearest pose sample is more than `max_gap_ns` away
    (downstream code should drop these frames).
    """
    if not samples:
        return InterpolatedPose(target_ts_ns, np.eye(4), float("inf"), 0, valid=False)

    T, err = interpolate_pose(samples, target_ts_ns)
    nearest_ts = min(
        samples, key=lambda s: abs(s.timestamp_ns - target_ts_ns)
    ).timestamp_ns
    return InterpolatedPose(
        target_timestamp_ns=target_ts_ns,
        world_T_ego=T,
        interp_error_ns=err,
        pose_timestamp_ns=int(nearest_ts),
        valid=err <= max_gap_ns,
    )
