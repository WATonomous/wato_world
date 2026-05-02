"""Pydantic schema for the ingest YAML config.

The actual parameter values live in `src/ingest/config/ingest.yaml` (or
whatever path you pass via --config).  This file only defines the *shape* of
the config — the YAML is the source of truth for values.
"""

from __future__ import annotations

import yaml
from pydantic import BaseModel, ConfigDict, Field


class CameraTopics(BaseModel):
    """Per-camera topic pair: image stream + CameraInfo."""

    model_config = ConfigDict(extra="forbid")

    image: str
    info: str


class TopicMap(BaseModel):
    """Maps logical sensor names (used in artifact paths) to bag topic names."""

    model_config = ConfigDict(extra="forbid")

    cameras: dict[str, CameraTopics]
    lidars: dict[str, str]
    # Pose: a single nav_msgs/Odometry topic.  Canonical source is eidos's
    # slam/odometry, stamped at LiDAR keyframe sensor time and expressed in
    # the SLAM-optimized map frame.  See ingest.yaml for details.
    pose: str
    # /tf_static carries rigid extrinsics; read once at bag scope.
    tf_static: str


class IngestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_seconds: float
    chunk_overlap_seconds: float
    reference_clock: str
    max_cam_offset_ms: float
    storage_id: str
    topics: TopicMap
    ego_frame: str
    quality_thresholds: dict[str, float]
    upstream_versions: dict[str, str] = Field(default_factory=dict)


def load_config(path: str) -> IngestConfig:
    """Load and validate an ingest YAML config from `path`."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return IngestConfig.model_validate(data)
