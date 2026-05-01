"""Pydantic + pyarrow schemas for the artifacts the pipeline produces.

Components serialize to Parquet using the field names defined here.  The artifact
tree is the source of truth for pipeline metadata.
"""

from __future__ import annotations

from typing import Optional

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Ingest — bag / chunk / sensor tables.
# ---------------------------------------------------------------------------

class BagMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bag_id: str
    source_path: str
    duration_s: float
    storage_type: str = "sqlite3"  # rosbag2 storage backend
    topics: dict[str, int] = Field(description="topic_name -> message count")
    vehicle: Optional[str] = None
    calibration_version: Optional[str] = None
    recording_date: Optional[str] = None


class ChunkRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bag_id: str
    chunk_id: str
    t_start_ns: int
    t_end_ns: int
    t_overlap_start_ns: int
    t_overlap_end_ns: int
    num_lidar_sweeps: int = 0
    num_camera_frames: int = 0


CHUNK_SCHEMA = pa.schema([
    pa.field("bag_id", pa.string()),
    pa.field("chunk_id", pa.string()),
    pa.field("t_start_ns", pa.int64()),
    pa.field("t_end_ns", pa.int64()),
    pa.field("t_overlap_start_ns", pa.int64()),
    pa.field("t_overlap_end_ns", pa.int64()),
    pa.field("num_lidar_sweeps", pa.int64()),
    pa.field("num_camera_frames", pa.int64()),
])


class CameraFrameRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bag_id: str
    chunk_id: str
    cam_id: str
    camera_seq: int
    image_path: str
    header_timestamp_ns: int
    record_timestamp_ns: int
    width: int
    height: int
    encoding: str
    is_compressed: bool
    valid: bool = True
    drop_reason: Optional[str] = None


CAMERA_FRAMES_SCHEMA = pa.schema([
    pa.field("bag_id", pa.string()),
    pa.field("chunk_id", pa.string()),
    pa.field("cam_id", pa.string()),
    pa.field("camera_seq", pa.int64()),
    pa.field("image_path", pa.string()),
    pa.field("header_timestamp_ns", pa.int64()),
    pa.field("record_timestamp_ns", pa.int64()),
    pa.field("width", pa.int64()),
    pa.field("height", pa.int64()),
    pa.field("encoding", pa.string()),
    pa.field("is_compressed", pa.bool_()),
    pa.field("valid", pa.bool_()),
    pa.field("drop_reason", pa.string()),
])


class LidarSweepRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bag_id: str
    chunk_id: str
    lidar_id: str
    sweep_id: int
    lidar_path: str
    header_timestamp_ns: int
    record_timestamp_ns: int
    num_points: int
    has_ring: bool
    has_intensity: bool
    has_point_time: bool
    min_range_m: float
    max_range_m: float
    valid: bool = True
    drop_reason: Optional[str] = None


LIDAR_SWEEPS_SCHEMA = pa.schema([
    pa.field("bag_id", pa.string()),
    pa.field("chunk_id", pa.string()),
    pa.field("lidar_id", pa.string()),
    pa.field("sweep_id", pa.int64()),
    pa.field("lidar_path", pa.string()),
    pa.field("header_timestamp_ns", pa.int64()),
    pa.field("record_timestamp_ns", pa.int64()),
    pa.field("num_points", pa.int64()),
    pa.field("has_ring", pa.bool_()),
    pa.field("has_intensity", pa.bool_()),
    pa.field("has_point_time", pa.bool_()),
    pa.field("min_range_m", pa.float64()),
    pa.field("max_range_m", pa.float64()),
    pa.field("valid", pa.bool_()),
    pa.field("drop_reason", pa.string()),
])


class PoseRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bag_id: str
    chunk_id: str
    timestamp_ns: int
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    world_T_ego_flat: list[float] = Field(description="row-major 4x4")
    source: str
    valid: bool = True


POSES_SCHEMA = pa.schema([
    pa.field("bag_id", pa.string()),
    pa.field("chunk_id", pa.string()),
    pa.field("timestamp_ns", pa.int64()),
    pa.field("x", pa.float64()),
    pa.field("y", pa.float64()),
    pa.field("z", pa.float64()),
    pa.field("qx", pa.float64()),
    pa.field("qy", pa.float64()),
    pa.field("qz", pa.float64()),
    pa.field("qw", pa.float64()),
    pa.field("world_T_ego_flat", pa.list_(pa.float64(), 16)),
    pa.field("source", pa.string()),
    pa.field("valid", pa.bool_()),
])


class FrameIndexRow(BaseModel):
    """One row per (sweep_id, cam_id) — the contract for downstream components."""
    model_config = ConfigDict(extra="forbid")

    frame_id: str
    bag_id: str
    chunk_id: str
    sweep_id: int
    lidar_id: str
    lidar_path: str
    reference_timestamp_ns: int

    cam_id: str
    image_path: Optional[str] = None
    camera_seq: Optional[int] = None
    camera_timestamp_ns: Optional[int] = None
    camera_offset_ms: Optional[float] = None
    valid_camera: bool = False
    camera_drop_reason: Optional[str] = None

    pose_timestamp_ns: Optional[int] = None
    world_T_ego_flat: Optional[list[float]] = None
    pose_interp_error: Optional[float] = None
    valid_pose: bool = False

    calibration_path: Optional[str] = None


FRAME_INDEX_SCHEMA = pa.schema([
    pa.field("frame_id", pa.string()),
    pa.field("bag_id", pa.string()),
    pa.field("chunk_id", pa.string()),
    pa.field("sweep_id", pa.int64()),
    pa.field("lidar_id", pa.string()),
    pa.field("lidar_path", pa.string()),
    pa.field("reference_timestamp_ns", pa.int64()),

    pa.field("cam_id", pa.string()),
    pa.field("image_path", pa.string()),
    pa.field("camera_seq", pa.int64()),
    pa.field("camera_timestamp_ns", pa.int64()),
    pa.field("camera_offset_ms", pa.float64()),
    pa.field("valid_camera", pa.bool_()),
    pa.field("camera_drop_reason", pa.string()),

    pa.field("pose_timestamp_ns", pa.int64()),
    pa.field("world_T_ego_flat", pa.list_(pa.float64(), 16)),
    pa.field("pose_interp_error", pa.float64()),
    pa.field("valid_pose", pa.bool_()),

    pa.field("calibration_path", pa.string()),
])


# ---------------------------------------------------------------------------
# Downstream component artifacts (unchanged from the original skeleton).
# ---------------------------------------------------------------------------

class Box3D(BaseModel):
    model_config = ConfigDict(extra="forbid")

    xyz: list[float]
    lwh: list[float]
    yaw: float


class MaskletRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    masklet_id: str
    bag_id: str
    chunk_id: str
    cam_id: int
    cls: str = Field(alias="class")
    score: float
    frames_present: list[int]
    mask_path: str
    dino_feature_path: Optional[str] = None
    global_object_id: Optional[str] = None


class ProposalRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    frame_id: str
    box: Box3D
    cls: str = Field(alias="class")
    score: float
    provenance: str
    lidar_point_count: Optional[int] = None
    supporting_cameras: list[int] = Field(default_factory=list)
    supporting_masklet_ids: list[str] = Field(default_factory=list)


class TrackRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    track_id: str
    frame_id: str
    box: Box3D
    cls: str = Field(alias="class")
    score: float
    supporting_cameras: list[int] = Field(default_factory=list)
    supporting_masklet_ids: list[str] = Field(default_factory=list)
    dino_feature_path: Optional[str] = None


class RefinedTrackRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    track_id: str
    frame_id: str
    box: Box3D
    cls: str = Field(alias="class")
    confidence: float
    residual_silhouette: Optional[float] = None
    residual_lidar_fit: Optional[float] = None
    residual_smoothness: Optional[float] = None
