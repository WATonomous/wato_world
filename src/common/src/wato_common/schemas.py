"""Pydantic + pyarrow schemas for the artifacts the pipeline produces.

Components serialize to Parquet using the field names defined here.  The artifact
tree is the source of truth for pipeline metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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


CHUNK_SCHEMA = pa.schema(
    [
        pa.field("bag_id", pa.string()),
        pa.field("chunk_id", pa.string()),
        pa.field("t_start_ns", pa.int64()),
        pa.field("t_end_ns", pa.int64()),
        pa.field("t_overlap_start_ns", pa.int64()),
        pa.field("t_overlap_end_ns", pa.int64()),
    ]
)


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


CAMERA_FRAMES_SCHEMA = pa.schema(
    [
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
    ]
)


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


LIDAR_SWEEPS_SCHEMA = pa.schema(
    [
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
    ]
)


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


POSES_SCHEMA = pa.schema(
    [
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
    ]
)


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


FRAME_INDEX_SCHEMA = pa.schema(
    [
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
    ]
)


# ---------------------------------------------------------------------------
# lidar_preprocessing artifacts.
# ---------------------------------------------------------------------------


class ProcessedSweepMeta(BaseModel):
    """One row per processed sweep written by lidar_preprocessing.

    `valid=False` rows record sweeps that ingest delivered but
    lidar_preprocessing failed to process — they let downstream stages
    distinguish "sweep dropped due to error" from "sweep never existed".

    `n_points_ground` and the `world_*` bbox columns are populated by Step A
    (deskew) so Steps B (classify) and C (ground) can plan their work
    without re-scanning every NPZ.  They're 0 / NaN for `valid=False` rows.
    """

    model_config = ConfigDict(extra="forbid")

    bag_id: str
    chunk_id: str
    sweep_id: int
    lidar_id: str
    reference_timestamp_ns: int
    n_points_total: int
    n_points_static: int
    n_points_dynamic: int
    n_points_ground: int = 0
    world_path: str
    dynamic_mask_path: str
    has_intensity: bool
    deskewed: bool
    valid: bool = True
    drop_reason: Optional[str] = None
    world_xmin: Optional[float] = None
    world_xmax: Optional[float] = None
    world_ymin: Optional[float] = None
    world_ymax: Optional[float] = None
    world_zmin: Optional[float] = None
    world_zmax: Optional[float] = None
    # SAM4D-style canonical-frame grouping: sweeps from different lidars whose
    # reference_timestamp_ns falls within the configured tolerance of a
    # canonical-lidar sweep share that sweep's frame_id.  Single-lidar bags
    # (canonical_lidar=None in config) just get frame_id = sweep ordinal.
    # Nullable: orphan sweeps outside any window are None.
    frame_id: Optional[int] = None


PROCESSED_SWEEPS_SCHEMA = pa.schema(
    [
        pa.field("bag_id", pa.string()),
        pa.field("chunk_id", pa.string()),
        pa.field("sweep_id", pa.int64()),
        pa.field("lidar_id", pa.string()),
        pa.field("reference_timestamp_ns", pa.int64()),
        pa.field("n_points_total", pa.int64()),
        pa.field("n_points_static", pa.int64()),
        pa.field("n_points_dynamic", pa.int64()),
        pa.field("n_points_ground", pa.int64()),
        pa.field("world_path", pa.string()),
        pa.field("dynamic_mask_path", pa.string()),
        pa.field("has_intensity", pa.bool_()),
        pa.field("deskewed", pa.bool_()),
        pa.field("valid", pa.bool_()),
        pa.field("drop_reason", pa.string()),
        pa.field("world_xmin", pa.float64()),
        pa.field("world_xmax", pa.float64()),
        pa.field("world_ymin", pa.float64()),
        pa.field("world_ymax", pa.float64()),
        pa.field("world_zmin", pa.float64()),
        pa.field("world_zmax", pa.float64()),
        pa.field("frame_id", pa.int64()),
    ]
)


class ChunkSummaryRow(BaseModel):
    """One-row chunk-level rollup of lidar_proc_index plus runtime stats.

    Lets downstream stages spot pathological chunks (high invalid rate,
    cache auto-disabled, no ground returns) without iterating every sweep
    row in lidar_proc_index.parquet.
    """

    model_config = ConfigDict(extra="forbid")

    bag_id: str
    chunk_id: str
    n_sweeps_total: int
    n_sweeps_valid: int
    n_sweeps_invalid: int
    n_points_total: int
    n_points_static: int
    n_points_dynamic: int
    n_points_ground: int
    n_dropped_dynamic_ground: int
    cache_auto_disabled: bool
    estimated_cache_bytes: int
    ground_status: str  # "ok" | "skipped_no_ground_mask" | "empty"


CHUNK_SUMMARY_SCHEMA = pa.schema(
    [
        pa.field("bag_id", pa.string()),
        pa.field("chunk_id", pa.string()),
        pa.field("n_sweeps_total", pa.int64()),
        pa.field("n_sweeps_valid", pa.int64()),
        pa.field("n_sweeps_invalid", pa.int64()),
        pa.field("n_points_total", pa.int64()),
        pa.field("n_points_static", pa.int64()),
        pa.field("n_points_dynamic", pa.int64()),
        pa.field("n_points_ground", pa.int64()),
        pa.field("n_dropped_dynamic_ground", pa.int64()),
        pa.field("cache_auto_disabled", pa.bool_()),
        pa.field("estimated_cache_bytes", pa.int64()),
        pa.field("ground_status", pa.string()),
    ]
)


# ---------------------------------------------------------------------------
# Downstream component artifacts.
# ---------------------------------------------------------------------------


@dataclass
class Box3D:
    """In-memory 3D box helper.  Not serialized to Parquet directly — downstream
    rows use flat cx/cy/cz/w/l/h/heading columns instead."""

    cx: float
    cy: float
    cz: float
    w: float
    l: float
    h: float
    heading: float  # radians, world frame


# ---------------------------------------------------------------------------
# perception_2d — masklets (per-camera instance, temporally associated).
# ---------------------------------------------------------------------------


class MaskletRow(BaseModel):
    """One row per tracklet-camera instance produced by perception_2d.

    frames_present and supporting_masklet_ids are JSON-encoded lists so they
    round-trip cleanly through Parquet without native list-column handling.
    """

    model_config = ConfigDict(extra="forbid")

    masklet_id: str
    bag_id: str
    chunk_id: str
    cam_id: str            # camera name, e.g. "CAM_FRONT"
    cls: str               # "vehicle" | "pedestrian" | "cyclist"
    score: float
    frames_present: str    # JSON list[int] — camera_seq values where mask is present
    mask_path: str         # path to directory of per-frame mask PNG files
    dino_feature_path: Optional[str] = None   # DINOv2 embedding NPZ
    global_object_id: Optional[str] = None    # cross-camera identity


MASKLET_SCHEMA = pa.schema(
    [
        pa.field("masklet_id", pa.string()),
        pa.field("bag_id", pa.string()),
        pa.field("chunk_id", pa.string()),
        pa.field("cam_id", pa.string()),
        pa.field("cls", pa.string()),
        pa.field("score", pa.float64()),
        pa.field("frames_present", pa.string()),
        pa.field("mask_path", pa.string()),
        pa.field("dino_feature_path", pa.string()),
        pa.field("global_object_id", pa.string()),
    ]
)


# ---------------------------------------------------------------------------
# perception_2d — per-camera-frame monocular depth (Depth Anything V2).
# ---------------------------------------------------------------------------


class DepthFrameRow(BaseModel):
    """One row per (cam_id, camera_seq) pointing at a saved depth map NPY.

    Depth Anything V2 emits *relative* depth; (scale, shift) align it to
    metric scale via overlapping LiDAR returns (see
    docs/research/depth_anything_guidance.md).  Downstream consumers in
    proposal_generation apply `z_metric = scale * depth + shift`.

    scale_method records how (scale, shift) was determined:
      "lidar_aligned" — RANSAC fit against overlapping LiDAR (the usual case)
      "metric_native" — model already emits metric depth, scale=1, shift=0
      "uncalibrated"  — insufficient overlap, downstream should skip depth term
    """

    model_config = ConfigDict(extra="forbid")

    bag_id: str
    chunk_id: str
    cam_id: str
    camera_seq: int
    depth_path: str
    model_name: str
    height: int
    width: int
    scale: float = 1.0
    shift: float = 0.0
    scale_method: str = "lidar_aligned"
    residual_rmse: Optional[float] = None
    n_overlap_pts: Optional[int] = None
    valid: bool = True
    drop_reason: Optional[str] = None


DEPTH_INDEX_SCHEMA = pa.schema(
    [
        pa.field("bag_id", pa.string()),
        pa.field("chunk_id", pa.string()),
        pa.field("cam_id", pa.string()),
        pa.field("camera_seq", pa.int64()),
        pa.field("depth_path", pa.string()),
        pa.field("model_name", pa.string()),
        pa.field("height", pa.int64()),
        pa.field("width", pa.int64()),
        pa.field("scale", pa.float64()),
        pa.field("shift", pa.float64()),
        pa.field("scale_method", pa.string()),
        pa.field("residual_rmse", pa.float64()),
        pa.field("n_overlap_pts", pa.int64()),
        pa.field("valid", pa.bool_()),
        pa.field("drop_reason", pa.string()),
    ]
)


# ---------------------------------------------------------------------------
# proposal_generation — 3D box proposals (one per LiDAR sweep × object).
# ---------------------------------------------------------------------------


class ProposalRow(BaseModel):
    """One row per 3D box proposal for a single sweep.

    supporting_cam_ids and supporting_masklet_ids are JSON-encoded lists.
    provenance identifies the source: "lidar_detector", "slf", or "fused".

    The diagnostic columns (n_detectors_agreeing, slf_*_loss/chamfer,
    lidar_density_in_box, da_pixels_in_mask, uncertainty) are the
    cross-modal uncertainty signals defined in
    docs/research/cross_modal_uncertainty_guidance.md.  They are populated
    by proposal_generation and consumed as soft weights by label_refinement.
    Optional so legacy producers can still write rows.
    """

    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    bag_id: str
    chunk_id: str
    sweep_id: int
    cx: float
    cy: float
    cz: float
    w: float
    l: float
    h: float
    heading: float   # radians, world frame
    cls: str         # "vehicle" | "pedestrian" | "cyclist"
    score: float
    provenance: str  # "lidar_detector" | "slf" | "fused"
    lidar_point_count: Optional[int] = None
    supporting_cam_ids: str = "[]"        # JSON list[str] of cam_id
    supporting_masklet_ids: str = "[]"    # JSON list[str] of masklet_id

    # MS3D++ ensemble bookkeeping.
    n_detectors_agreeing: Optional[int] = None
    ensemble_score_var: Optional[float] = None

    # SLF fitter convergence diagnostics.
    slf_dice_loss: Optional[float] = None
    slf_lidar_chamfer: Optional[float] = None
    slf_depth_chamfer: Optional[float] = None

    # Raw evidence signals.
    lidar_density_in_box: Optional[float] = None  # points / m³
    da_pixels_in_mask: Optional[int] = None       # DA pseudo-LiDAR pixel count

    # Combined uncertainty in [0, 1] (0 = high confidence, 1 = low).
    uncertainty: Optional[float] = None


PROPOSAL_SCHEMA = pa.schema(
    [
        pa.field("proposal_id", pa.string()),
        pa.field("bag_id", pa.string()),
        pa.field("chunk_id", pa.string()),
        pa.field("sweep_id", pa.int64()),
        pa.field("cx", pa.float64()),
        pa.field("cy", pa.float64()),
        pa.field("cz", pa.float64()),
        pa.field("w", pa.float64()),
        pa.field("l", pa.float64()),
        pa.field("h", pa.float64()),
        pa.field("heading", pa.float64()),
        pa.field("cls", pa.string()),
        pa.field("score", pa.float64()),
        pa.field("provenance", pa.string()),
        pa.field("lidar_point_count", pa.int64()),
        pa.field("supporting_cam_ids", pa.string()),
        pa.field("supporting_masklet_ids", pa.string()),
        pa.field("n_detectors_agreeing", pa.int64()),
        pa.field("ensemble_score_var", pa.float64()),
        pa.field("slf_dice_loss", pa.float64()),
        pa.field("slf_lidar_chamfer", pa.float64()),
        pa.field("slf_depth_chamfer", pa.float64()),
        pa.field("lidar_density_in_box", pa.float64()),
        pa.field("da_pixels_in_mask", pa.int64()),
        pa.field("uncertainty", pa.float64()),
    ]
)


# Helpers for JSON-encoded list fields.
def encode_str_list(values: list[str]) -> str:
    return json.dumps(values)


def decode_str_list(s: str) -> list[str]:
    return json.loads(s) if s else []


def encode_int_list(values: list[int]) -> str:
    return json.dumps(values)


def decode_int_list(s: str) -> list[int]:
    return json.loads(s) if s else []


# ---------------------------------------------------------------------------
# tracking — per-sweep track state (bag-level, spans all chunks).
# ---------------------------------------------------------------------------


class TrackRow(BaseModel):
    """One row per (track_id, sweep_id) in tracks.parquet.

    supporting_cam_ids, supporting_masklet_ids, and merged_from are
    JSON-encoded lists.

    DetZero-style bidirectional tracking writes both `direction` (which pass
    produced the row) and `merged_from` (list of upstream track_ids that
    were joined at an endpoint, if any).
    """

    model_config = ConfigDict(extra="forbid")

    track_id: str
    bag_id: str
    chunk_id: str
    sweep_id: int
    cx: float
    cy: float
    cz: float
    w: float
    l: float
    h: float
    heading: float  # radians, world frame
    cls: str        # "vehicle" | "pedestrian" | "cyclist"
    score: float
    supporting_cam_ids: str = "[]"      # JSON list[str]
    supporting_masklet_ids: str = "[]"  # JSON list[str]
    dino_feature_path: Optional[str] = None
    direction: str = "forward"          # "forward" | "backward" | "merged"
    merged_from: str = "[]"             # JSON list[str] of predecessor track_ids


TRACK_SCHEMA = pa.schema(
    [
        pa.field("track_id", pa.string()),
        pa.field("bag_id", pa.string()),
        pa.field("chunk_id", pa.string()),
        pa.field("sweep_id", pa.int64()),
        pa.field("cx", pa.float64()),
        pa.field("cy", pa.float64()),
        pa.field("cz", pa.float64()),
        pa.field("w", pa.float64()),
        pa.field("l", pa.float64()),
        pa.field("h", pa.float64()),
        pa.field("heading", pa.float64()),
        pa.field("cls", pa.string()),
        pa.field("score", pa.float64()),
        pa.field("supporting_cam_ids", pa.string()),
        pa.field("supporting_masklet_ids", pa.string()),
        pa.field("dino_feature_path", pa.string()),
        pa.field("direction", pa.string()),
        pa.field("merged_from", pa.string()),
    ]
)


# ---------------------------------------------------------------------------
# label_refinement — per-sweep refined track state.
# ---------------------------------------------------------------------------


class RefinedTrackRow(BaseModel):
    """One row per (track_id, sweep_id) in refined_labels.parquet.

    The LabelFormer produces a single (w, l, h) shared across all frames of a
    track, so w/l/h will be identical for every row of the same track_id.
    Per-frame pose corrections are applied to cx/cy/cz/heading.
    """

    model_config = ConfigDict(extra="forbid")

    track_id: str
    bag_id: str
    chunk_id: str
    sweep_id: int
    cx: float
    cy: float
    cz: float
    w: float
    l: float
    h: float
    heading: float  # radians, world frame
    cls: str
    confidence: float
    residual_silhouette: Optional[float] = None
    residual_lidar_fit: Optional[float] = None
    residual_smoothness: Optional[float] = None


REFINED_TRACK_SCHEMA = pa.schema(
    [
        pa.field("track_id", pa.string()),
        pa.field("bag_id", pa.string()),
        pa.field("chunk_id", pa.string()),
        pa.field("sweep_id", pa.int64()),
        pa.field("cx", pa.float64()),
        pa.field("cy", pa.float64()),
        pa.field("cz", pa.float64()),
        pa.field("w", pa.float64()),
        pa.field("l", pa.float64()),
        pa.field("h", pa.float64()),
        pa.field("heading", pa.float64()),
        pa.field("cls", pa.string()),
        pa.field("confidence", pa.float64()),
        pa.field("residual_silhouette", pa.float64()),
        pa.field("residual_lidar_fit", pa.float64()),
        pa.field("residual_smoothness", pa.float64()),
    ]
)


# ---------------------------------------------------------------------------
# label_refinement — per-track body-frame aggregated cloud (3DAL / DetZero).
# ---------------------------------------------------------------------------


class AggregatedTrackRow(BaseModel):
    """One row per track summarising the body-frame aggregated cloud written
    by label_refinement's aggregation step.

    The actual point cloud lives in the NPZ at `points_path`; this row is
    the index entry consumed by LabelFormer's batch loader.

    aggregation_method records the path taken:
      "naive"          — concat-only, no ICP correction
      "icp_corrected"  — DetZero geometric refining applied
      "empty"          — no dynamic points inside the track's enlarged box
      "high_pose_jitter" — upstream tracker confidence was too low to trust
    """

    model_config = ConfigDict(extra="forbid")

    track_id: str
    bag_id: str
    cls: str
    n_frames: int
    n_points_aggregated: int
    points_path: str            # URI to aggregated_tracks/<track>.npz
    aggregation_method: str
    voxel_size_m: float
    icp_corrected: bool = False
    pose_jitter_m: Optional[float] = None  # diagnostic: how much ICP moved frames


AGGREGATED_TRACK_SCHEMA = pa.schema(
    [
        pa.field("track_id", pa.string()),
        pa.field("bag_id", pa.string()),
        pa.field("cls", pa.string()),
        pa.field("n_frames", pa.int64()),
        pa.field("n_points_aggregated", pa.int64()),
        pa.field("points_path", pa.string()),
        pa.field("aggregation_method", pa.string()),
        pa.field("voxel_size_m", pa.float64()),
        pa.field("icp_corrected", pa.bool_()),
        pa.field("pose_jitter_m", pa.float64()),
    ]
)
