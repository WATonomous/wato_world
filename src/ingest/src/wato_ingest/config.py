"""Pydantic schema for the ingest YAML config.

The actual parameter values live in `src/ingest/config/ingest.yaml` (or
whatever path you pass via --config).  This file only defines the *shape* of
the config — the YAML is the source of truth for values.  Ingest has no
per-dataset code: a new recording needs only a new YAML (README "Ingesting a
new bag").
"""

from __future__ import annotations

from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field


class CameraTopics(BaseModel):
    """Per-camera topic pair: image stream + CameraInfo."""

    model_config = ConfigDict(extra="forbid")

    # sensor_msgs/CompressedImage (JPEG/PNG) or sensor_msgs/Image.
    image: str
    # sensor_msgs/CameraInfo.  Only needed when calibration comes from the bag;
    # omit it when the bag has none and pass --calibration instead.
    info: Optional[str] = None


class TopicMap(BaseModel):
    """Maps logical sensor names (used in artifact paths) to bag topic names."""

    model_config = ConfigDict(extra="forbid")

    cameras: dict[str, CameraTopics]
    # sensor_msgs/PointCloud2, one topic per physical LiDAR.
    lidars: dict[str, str]
    # Pose: one topic of nav_msgs/Odometry, geometry_msgs/PoseStamped or
    # geometry_msgs/PoseWithCovarianceStamped giving the world pose of
    # `ego_frame` (an Odometry's child_frame_id must equal it).  It must be
    # DENSE and SMOOTH — see PoseRequirements and the README's "Pose
    # requirements" section for which topics qualify.
    pose: str
    # tf2_msgs/TFMessage topic carrying the rigid sensor extrinsics (usually
    # /tf_static); read once at bag scope.  Like `info`, only needed when
    # calibration comes from the bag.
    tf_static: Optional[str] = None


class PoseRequirements(BaseModel):
    """What ingest demands of the `topics.pose` stream.

    Every pose downstream (a sweep's, a LiDAR point's, a camera frame's) is
    interpolated between the two pose samples around it, which assumes
    constant velocity in between.  That only holds when the samples are close
    together, so ingest enforces it instead of silently interpolating across
    seconds.  The two stretch-level rules are written into poses.parquet
    (`interval_drop_reason`) and applied by wato_common.pose_lookup wherever a
    pose is looked up.  See the README's "Pose requirements" section for the
    measured rationale.
    """

    model_config = ConfigDict(extra="forbid")

    # Chunk level, hard failure: the fraction of the chunk's pose span that
    # lies between samples at most `max_bracket_ms` apart.  Below this, ingest
    # aborts the bag.  Coverage, not median spacing, so a bursty stream (many
    # samples at once, then seconds of nothing) can't pass on paper.
    min_dense_fraction: float = Field(gt=0, le=1)
    # Stretch level: two consecutive samples further apart than this can't be
    # interpolated between (interval_drop_reason pose_gap_*); a sweep or camera
    # frame inside that stretch gets no valid pose.
    max_bracket_ms: float = Field(gt=0)
    # Stretch level: an implied ego speed above this between two consecutive
    # samples is a pose jump (loop closure, INS reset), not motion
    # (interval_drop_reason pose_jump_*).
    max_speed_mps: float = Field(gt=0)


class IngestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_seconds: float
    chunk_overlap_seconds: float
    reference_clock: str
    max_cam_offset_ms: float
    pose_requirements: PoseRequirements
    # rosbag2 storage plugin ("mcap", "sqlite3").  Empty = detect from the bag,
    # which is what every shipped profile does.
    storage_id: str = ""
    topics: TopicMap
    ego_frame: str
    quality_thresholds: dict[str, float]
    upstream_versions: dict[str, str] = Field(default_factory=dict)


def load_config(path: str) -> IngestConfig:
    """Load and validate an ingest YAML config from `path`."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return IngestConfig.model_validate(data)
