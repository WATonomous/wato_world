"""Decode LiDAR PointCloud2 messages into per-sweep .npz files."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wato_common.artifact_store import (
    ensure_local_dir,
    lidar_dir,
    lidar_sweep_path,
    lidar_sweeps_path,
    local_path,
)
from wato_common.io.parquet_io import write_table
from wato_common.io.pointcloud2 import decode_pointcloud2, fields_to_specs
from wato_common.io.rosbag_reader import messages
from wato_common.schemas import LIDAR_SWEEPS_SCHEMA, LidarSweepRow
from wato_ingest.config import IngestConfig


@dataclass
class LidarDecodeResult:
    lidar_id: str
    sweeps_written: int
    output_dir: str


# Per-point-time field synonyms across vendors.
_POINT_TIME_FIELDS = ("t", "time", "timestamp", "time_stamp", "t_offset_us")


def _header_ts_ns(header) -> int:
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def decode_chunk(
    bag_path: str,
    bag_id: str,
    chunk_id: str,
    *,
    t_start_ns: int,
    t_end_ns: int,
    cfg: IngestConfig,
) -> list[LidarDecodeResult]:
    lidar_topics = cfg.topics.lidars
    topic_to_lidar = {v: k for k, v in lidar_topics.items()}
    if not topic_to_lidar:
        return []

    for lidar_id in lidar_topics:
        ensure_local_dir(lidar_dir(bag_id, chunk_id))

    sweep_per_lidar: dict[str, int] = {lid: 0 for lid in lidar_topics}
    rows: list[dict] = []

    with messages(
        bag_path,
        storage_id=cfg.storage_id,
        topics=list(topic_to_lidar.keys()),
        t_start_ns=t_start_ns,
        t_end_ns=t_end_ns,
    ) as iterator:
        for topic, msg, record_ts_ns in iterator:
            lidar_id = topic_to_lidar[topic]
            seq = sweep_per_lidar[lidar_id]
            row = _write_sweep(
                msg=msg,
                bag_id=bag_id,
                chunk_id=chunk_id,
                lidar_id=lidar_id,
                sweep_id=seq,
                record_ts_ns=record_ts_ns,
            )
            if row is not None:
                rows.append(row)
                sweep_per_lidar[lidar_id] = seq + 1

    write_table(rows, LIDAR_SWEEPS_SCHEMA, lidar_sweeps_path(bag_id, chunk_id))

    return [
        LidarDecodeResult(
            lidar_id=lidar_id,
            sweeps_written=sweep_per_lidar[lidar_id],
            output_dir=lidar_dir(bag_id, chunk_id),
        )
        for lidar_id in lidar_topics
    ]


def _write_sweep(
    *,
    msg,
    bag_id: str,
    chunk_id: str,
    lidar_id: str,
    sweep_id: int,
    record_ts_ns: int,
) -> dict | None:
    if type(msg).__name__ != "PointCloud2":
        return None
    try:
        header_ts = _header_ts_ns(msg.header)
    except AttributeError:
        return None

    field_specs = fields_to_specs(msg.fields)
    decoded = decode_pointcloud2(
        bytes(msg.data),
        field_specs,
        point_step=int(msg.point_step),
        width=int(msg.width),
        height=int(msg.height),
        is_bigendian=bool(msg.is_bigendian),
    )

    if not all(k in decoded for k in ("x", "y", "z")):
        return None

    field_names = set(decoded.keys())
    has_intensity = "intensity" in field_names
    has_ring = "ring" in field_names
    point_time_field = next((f for f in _POINT_TIME_FIELDS if f in field_names), None)
    has_point_time = point_time_field is not None

    n = decoded["x"].shape[0]
    if n == 0:
        return None

    save_kwargs: dict[str, np.ndarray] = {
        "x": decoded["x"].astype(np.float32),
        "y": decoded["y"].astype(np.float32),
        "z": decoded["z"].astype(np.float32),
    }
    if has_intensity:
        save_kwargs["intensity"] = decoded["intensity"].astype(np.float32)
    if has_ring:
        save_kwargs["ring"] = decoded["ring"].astype(np.uint16)
    if has_point_time:
        save_kwargs["t_offset_us"] = decoded[point_time_field].astype(np.float64)

    out_uri = lidar_sweep_path(bag_id, chunk_id, sweep_id)
    np.savez_compressed(local_path(out_uri), **save_kwargs)

    ranges = np.linalg.norm(np.stack([save_kwargs["x"], save_kwargs["y"], save_kwargs["z"]], axis=1), axis=1)

    row = LidarSweepRow(
        bag_id=bag_id,
        chunk_id=chunk_id,
        lidar_id=lidar_id,
        sweep_id=sweep_id,
        lidar_path=out_uri,
        header_timestamp_ns=header_ts,
        record_timestamp_ns=record_ts_ns,
        num_points=int(n),
        has_ring=has_ring,
        has_intensity=has_intensity,
        has_point_time=has_point_time,
        min_range_m=float(ranges.min()) if n > 0 else 0.0,
        max_range_m=float(ranges.max()) if n > 0 else 0.0,
        valid=True,
    )
    return row.model_dump()
