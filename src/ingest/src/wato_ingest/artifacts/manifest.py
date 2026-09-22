"""Write an ingest manifest per chunk for traceability.

Thin adapter over ``wato_common.manifest``, which all components share. This
module only knows ingest's own input (the rosbag) and output set; the
provenance capture, input hashing and manifest schema live in wato_common so
that nine components cannot drift into nine manifest formats.
"""

from __future__ import annotations

import os
from typing import Any

from wato_common import manifest as common_manifest
from wato_common.artifact_store import (
    bag_meta_path,
    calibration_path,
    camera_frames_path,
    chunks_index_path,
    frame_index_path,
    lidar_sweeps_path,
    poses_path,
    quality_path,
)


def write(
    *,
    bag_id: str,
    chunk_id: str,
    bag_path: str,
    config_path: str,
    extra: dict[str, Any] | None = None,
) -> str:
    """Write ``manifest.json`` for one ingested chunk and return its URI.

    The rosbag is declared as an input so the manifest carries its hash (for a
    bag under the 64 MB threshold) or its size+mtime. That is what lets a
    downstream stage tell "ingest already ran for this chunk" apart from
    "ingest ran, but against a bag that has since been replaced".
    """
    return common_manifest.write(
        component="ingest",
        bag_id=bag_id,
        chunk_id=chunk_id,
        inputs={"rosbag": f"file://{os.path.abspath(bag_path)}"},
        outputs={
            "bag_meta": bag_meta_path(bag_id),
            "calibration": calibration_path(bag_id),
            "chunks_index": chunks_index_path(bag_id),
            "camera_frames": camera_frames_path(bag_id, chunk_id),
            "lidar_sweeps": lidar_sweeps_path(bag_id, chunk_id),
            "poses": poses_path(bag_id, chunk_id),
            "frame_index": frame_index_path(bag_id, chunk_id),
            "quality": quality_path(bag_id, chunk_id),
        },
        config_path=config_path,
        extra=extra,
    )
