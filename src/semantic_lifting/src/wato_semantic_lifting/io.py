"""Artifact I/O helpers for semantic_lifting.

Reads upstream artifacts from ingest + lidar_preprocessing + perception_2d
(calibration, processed sweeps, depth and masklets through
wato_common.artifacts).
Writes lifted_labels/, lifted_instances.parquet and lifted_stats.parquet
through wato_common.artifacts.lifted_labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wato_common.artifact_store import (
    chunks_index_path,
    frame_index_path,
    local_path,
)
from wato_common.artifacts import ArtifactFormatError
from wato_common.io.parquet_io import read_rows
from wato_common.pose_lookup import PoseLookup


@dataclass(frozen=True)
class CameraFrameRef:
    """One camera frame, with the ego pose at the time it was taken."""

    cam_id: str
    camera_seq: int
    timestamp_ns: int
    world_T_ego: np.ndarray  # (4, 4) float64


def load_chunks(bag_id: str) -> list[dict]:
    return read_rows(chunks_index_path(bag_id))


def load_sweep_frames(bag_id: str, chunk_id: str) -> dict[int, list[CameraFrameRef]]:
    """The camera frames ingest paired with each sweep, by sweep_id.

    The pairing is frame_index's (one row per (sweep, camera), nearest frame
    within ingest's max_cam_offset_ms); rows ingest dropped (valid_camera
    False, see camera_drop_reason) are skipped. A frame paired with several
    sweeps is one shared CameraFrameRef. Its pose is looked up at
    ``camera_timestamp_ns`` (wato_common.pose_lookup) — frame_index's own
    ``world_T_ego`` is the sweep's pose, and the car moves between the two.
    Frames whose time falls in an untrusted stretch of the pose stream are
    skipped.
    """
    path = frame_index_path(bag_id, chunk_id)
    poses = PoseLookup.load(bag_id, chunk_id)
    frames: dict[tuple[str, int], CameraFrameRef | None] = {}
    by_sweep: dict[int, list[CameraFrameRef]] = {}
    for r in read_rows(path):
        if not r.get("valid_camera", False):
            continue
        if r.get("camera_seq") is None or r.get("camera_timestamp_ns") is None:
            raise ArtifactFormatError(
                local_path(path),
                f"sweep {r['sweep_id']} / {r['cam_id']}: valid_camera row "
                "without camera_seq or camera_timestamp_ns",
            )
        key = (str(r["cam_id"]), int(r["camera_seq"]))
        if key not in frames:
            t = int(r["camera_timestamp_ns"])
            pose = poses.at(t)
            frames[key] = (
                CameraFrameRef(key[0], key[1], t, pose.world_T_ego)
                if pose.valid
                else None
            )
        ref = frames[key]
        if ref is not None:
            by_sweep.setdefault(int(r["sweep_id"]), []).append(ref)
    return by_sweep
