"""Write an ingest manifest per chunk for traceability."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
from typing import Any

import yaml

from wato_common.artifact_store import (
    bag_meta_path,
    calibration_path,
    camera_frames_path,
    chunks_index_path,
    ensure_local_dir,
    frame_index_path,
    lidar_sweeps_path,
    local_path,
    manifest_path,
    poses_path,
    quality_path,
)


def _config_hash(config_path: str) -> str:
    """Stable hash of pipeline.yaml's ingest section."""
    if not os.path.exists(config_path):
        return ""
    with open(config_path, "rb") as fh:
        h = hashlib.sha256(fh.read()).hexdigest()[:12]
    return h


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return out
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def write(
    *,
    bag_id: str,
    chunk_id: str,
    bag_path: str,
    config_path: str,
    extra: dict[str, Any] | None = None,
) -> str:
    manifest = {
        "component": "ingest",
        "bag_id": bag_id,
        "chunk_id": chunk_id,
        "inputs": {"rosbag": os.path.abspath(bag_path)},
        "outputs": {
            "bag_meta": bag_meta_path(bag_id),
            "calibration": calibration_path(bag_id),
            "chunks_index": chunks_index_path(bag_id),
            "camera_frames": camera_frames_path(bag_id, chunk_id),
            "lidar_sweeps": lidar_sweeps_path(bag_id, chunk_id),
            "poses": poses_path(bag_id, chunk_id),
            "frame_index": frame_index_path(bag_id, chunk_id),
            "quality": quality_path(bag_id, chunk_id),
        },
        "config_hash": _config_hash(config_path),
        "git_commit": _git_commit(),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if extra:
        manifest.update(extra)

    out_uri = manifest_path(bag_id, chunk_id)
    ensure_local_dir(os.path.dirname(local_path(out_uri)))
    with open(local_path(out_uri), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return out_uri
