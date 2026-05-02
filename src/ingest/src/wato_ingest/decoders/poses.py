"""Extract ego pose over time from the configured pose topic.

The pose source is set by `topics.pose` in ingest.yaml.  Canonical source is
eidos's `slam/odometry` (`nav_msgs/Odometry`), which is the SLAM-optimized
pose in the map frame stamped at LiDAR keyframe sensor time.

We do NOT consume /tf or /tf_static here — eidos does not publish to TF, and
the eidos_transform-published TF stream is wall-clock-stamped, which would
desync from the LiDAR sweeps we're aligning against.  See
wato_monorepo/src/world_modeling/eidos/README.md for the architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from wato_common.artifact_store import ensure_local_dir, local_path, poses_path
from wato_common.geometry import flatten_se3, make_se3
from wato_common.io.parquet_io import write_table
from wato_common.io.rosbag_reader import messages
from wato_common.schemas import POSES_SCHEMA, PoseRow
from wato_ingest.config import IngestConfig


@dataclass
class PoseExtractionResult:
    rows_written: int
    output_uri: str


def _header_ts_ns(header) -> int:
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def extract(
    bag_path: str,
    bag_id: str,
    chunk_id: str,
    *,
    t_start_ns: int,
    t_end_ns: int,
    cfg: IngestConfig,
) -> PoseExtractionResult:
    """Read every Odometry message on `cfg.topics.pose` in the chunk's window
    and write `poses.parquet`.  One row per message, deduplicated on timestamp.
    """
    rows: list[dict] = []

    with messages(
        bag_path,
        storage_id=cfg.storage_id,
        topics=[cfg.topics.pose],
        t_start_ns=t_start_ns,
        t_end_ns=t_end_ns,
    ) as iterator:
        for _topic, msg, _record_ts_ns in iterator:
            if type(msg).__name__ != "Odometry":
                continue
            rows.append(_pose_row(bag_id, chunk_id, msg))

    rows = _dedupe_and_sort(rows)
    out_uri = poses_path(bag_id, chunk_id)
    ensure_local_dir(local_path(out_uri).rsplit("/", 1)[0])
    write_table(rows, POSES_SCHEMA, out_uri)
    return PoseExtractionResult(rows_written=len(rows), output_uri=out_uri)


def _pose_row(bag_id: str, chunk_id: str, msg) -> dict:
    pose = msg.pose.pose
    T = make_se3(
        np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64),
        (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w),
    )
    return PoseRow(
        bag_id=bag_id,
        chunk_id=chunk_id,
        timestamp_ns=_header_ts_ns(msg.header),
        x=float(T[0, 3]),
        y=float(T[1, 3]),
        z=float(T[2, 3]),
        qx=float(pose.orientation.x),
        qy=float(pose.orientation.y),
        qz=float(pose.orientation.z),
        qw=float(pose.orientation.w),
        world_T_ego_flat=flatten_se3(T),
        source=msg.header.frame_id or "odom",
        valid=True,
    ).model_dump()


def _dedupe_and_sort(rows: Iterable[dict]) -> list[dict]:
    """Drop duplicate timestamps (keeping the first) and sort ascending."""
    seen: set[int] = set()
    out: list[dict] = []
    for r in sorted(rows, key=lambda r: r["timestamp_ns"]):
        if r["timestamp_ns"] in seen:
            continue
        seen.add(r["timestamp_ns"])
        out.append(r)
    return out
