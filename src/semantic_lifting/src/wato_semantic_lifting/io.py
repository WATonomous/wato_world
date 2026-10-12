"""Artifact I/O helpers for semantic_lifting.

Reads upstream artifacts from ingest + lidar_preprocessing + perception_2d
(calibration, processed sweeps, depth and masklets through
wato_common.artifacts).
Writes lifted_labels/ and lifted_stats.parquet through
wato_common.artifacts.lifted_labels.
"""

from __future__ import annotations

from wato_common.artifact_store import (
    chunks_index_path,
    frame_index_path,
)
from wato_common.io.parquet_io import read_rows
from wato_common.pose_lookup import PoseLookup
from wato_semantic_lifting.temporal_match import CameraFrameRef


def load_chunks(bag_id: str) -> list[dict]:
    return read_rows(chunks_index_path(bag_id))


def load_frame_refs(bag_id: str, chunk_id: str) -> list[CameraFrameRef]:
    """Return one reference per camera frame, with the ego pose at the time
    that frame was taken.

    frame_index has one row per (sweep, camera), so a camera frame appears once
    for every sweep it is nearest to; it is returned once.  Its pose is looked
    up at ``camera_timestamp_ns`` (wato_common.pose_lookup) — frame_index's own
    ``world_T_ego`` is the sweep's pose, and the car moves between the two.
    Frames whose time falls in an untrusted stretch of the pose stream are
    skipped.
    """
    rows = read_rows(frame_index_path(bag_id, chunk_id))
    poses = PoseLookup.load(bag_id, chunk_id)
    result = []
    seen: set[tuple[str, int]] = set()
    for r in rows:
        cam_ts = r.get("camera_timestamp_ns")
        if not r.get("valid_camera", False) or cam_ts is None:
            continue
        key = (str(r["cam_id"]), int(r["camera_seq"]))
        if key in seen:
            continue
        seen.add(key)
        pose = poses.at(int(cam_ts))
        if not pose.valid:
            continue
        result.append(
            CameraFrameRef(
                cam_id=key[0],
                camera_seq=key[1],
                timestamp_ns=int(cam_ts),
                world_T_ego=pose.world_T_ego,
            )
        )
    return result
