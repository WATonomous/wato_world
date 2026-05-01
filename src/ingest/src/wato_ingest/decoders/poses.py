"""Extract ego pose over time from /tf, /tf_static, and /odom.

Strategy:
1. Build a static-tf cache from /tf_static (transforms that don't change).
2. Walk /tf and /odom in time order.  Whenever we see a transform connecting
   `world_frame` -> `ego_frame` (directly or via a single hop), emit a PoseRow.
3. /odom is preferred when it carries pose-with-covariance and the chain is
   simple (odom -> base_link, with base_link being our ego frame).

If the bag is unusual, override `world_frame` and `ego_frame` via
IngestConfig.extra_fields.
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


def _transform_se3(transform) -> np.ndarray:
    t = transform.translation
    q = transform.rotation
    return make_se3(
        np.array([t.x, t.y, t.z], dtype=np.float64),
        (q.x, q.y, q.z, q.w),
    )


def extract(
    bag_path: str,
    bag_id: str,
    chunk_id: str,
    *,
    t_start_ns: int,
    t_end_ns: int,
    cfg: IngestConfig,
    world_frame: str = "map",
    ego_frame: str = "base_link",
) -> PoseExtractionResult:
    rows: list[dict] = []

    # Walk /tf in time order; emit a PoseRow whenever we observe a transform
    # whose (parent, child) chain implies world_frame -> ego_frame.
    with messages(
        bag_path,
        storage_id=cfg.storage_id,
        topics=[cfg.topics.tf, cfg.topics.tf_static, cfg.topics.odom],
        t_start_ns=t_start_ns,
        t_end_ns=t_end_ns,
    ) as iterator:
        # Cache of latest non-static parent->child SE3.
        link_cache: dict[tuple[str, str], np.ndarray] = {}

        for topic, msg, record_ts_ns in iterator:
            mt = type(msg).__name__
            if mt == "TFMessage":
                for ts in msg.transforms:
                    parent = ts.header.frame_id
                    child = ts.child_frame_id
                    link_cache[(parent, child)] = _transform_se3(ts.transform)
                    # Try to resolve world -> ego whenever any link updates.
                    T = _resolve_chain(link_cache, world_frame, ego_frame)
                    if T is not None:
                        ts_ns = _header_ts_ns(ts.header)
                        rows.append(_pose_row(bag_id, chunk_id, ts_ns, T, source="tf"))
            elif mt == "Odometry":
                # nav_msgs/Odometry — pose.pose has position + orientation.
                pose = msg.pose.pose
                T = make_se3(
                    np.array([pose.position.x, pose.position.y, pose.position.z]),
                    (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w),
                )
                ts_ns = _header_ts_ns(msg.header)
                rows.append(_pose_row(bag_id, chunk_id, ts_ns, T, source="odom"))

    rows = _dedupe_and_sort(rows)
    out_uri = poses_path(bag_id, chunk_id)
    ensure_local_dir(local_path(out_uri).rsplit("/", 1)[0])
    write_table(rows, POSES_SCHEMA, out_uri)
    return PoseExtractionResult(rows_written=len(rows), output_uri=out_uri)


def _resolve_chain(
    cache: dict[tuple[str, str], np.ndarray],
    world: str,
    ego: str,
    max_hops: int = 4,
) -> np.ndarray | None:
    """Naive forward-chain walk: world -> ... -> ego.  Returns None if not connected."""
    if (world, ego) in cache:
        return cache[(world, ego)]

    # BFS up to max_hops.
    visited = {world}
    frontier: list[tuple[str, np.ndarray]] = [(world, np.eye(4))]
    for _ in range(max_hops):
        next_frontier: list[tuple[str, np.ndarray]] = []
        for parent, T in frontier:
            for (p, c), Tpc in cache.items():
                if p == parent and c not in visited:
                    Tnext = T @ Tpc
                    if c == ego:
                        return Tnext
                    visited.add(c)
                    next_frontier.append((c, Tnext))
        frontier = next_frontier
        if not frontier:
            break
    return None


def _pose_row(bag_id: str, chunk_id: str, ts_ns: int, T: np.ndarray, *, source: str) -> dict:
    from wato_common.geometry.transforms import matrix_to_quat
    qx, qy, qz, qw = matrix_to_quat(T[:3, :3])
    return PoseRow(
        bag_id=bag_id,
        chunk_id=chunk_id,
        timestamp_ns=int(ts_ns),
        x=float(T[0, 3]),
        y=float(T[1, 3]),
        z=float(T[2, 3]),
        qx=float(qx),
        qy=float(qy),
        qz=float(qz),
        qw=float(qw),
        world_T_ego_flat=flatten_se3(T),
        source=source,
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
