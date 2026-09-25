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
    storage_type: str = "sqlite3"  # rosbag2 storage backend (detected from the bag)
    topics: dict[str, int] = Field(description="topic_name -> message count")
    topic_types: dict[str, str] = Field(
        default_factory=dict, description="topic_name -> message type"
    )
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
    # Unique within the chunk across ALL LiDARs (numbered in record order), so
    # (bag_id, chunk_id, sweep_id) names one sweep on a multi-LiDAR rig too.
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
    # Why the stretch from this sample to the next can't be interpolated
    # across: pose_gap_<ms>ms | pose_jump_<mps>mps; None = trusted (and always
    # None on the last sample).  Set by ingest from pose_requirements; read by
    # wato_common.pose_lookup, which every component uses to look poses up.
    interval_drop_reason: Optional[str] = None


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
        pa.field("interval_drop_reason", pa.string()),
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

    # Ego pose at the SWEEP's time (reference_timestamp_ns), not the camera's.
    # A component that projects into the image looks the pose up at
    # camera_timestamp_ns with wato_common.pose_lookup instead.
    pose_timestamp_ns: Optional[int] = None
    world_T_ego_flat: Optional[list[float]] = None
    pose_interp_error: Optional[float] = None
    valid_pose: bool = False
    # Why valid_pose is False: no_pose_samples | outside_pose_span |
    # pose_gap_<ms>ms | pose_jump_<mps>mps.  Null when valid_pose is True.
    pose_drop_reason: Optional[str] = None

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
        pa.field("world_T_ego_flat", pa.list_(pa.float64())),
        pa.field("pose_interp_error", pa.float64()),
        pa.field("valid_pose", pa.bool_()),
        pa.field("pose_drop_reason", pa.string()),
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
    # Per-point times were applied: the sweep's own time field, or times
    # synthesized from azimuth.  False = every point got the header pose.
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
    # Step A.5 — MF-MOS mask path.  Populated by mf_mos.process_chunk on the
    # `--seg mos` / `--seg union` paths; None otherwise (and on `--seg aw`).  Points to a (n_raw,) bool
    # NPY file aligned to the raw sweep (same length as the raw lidar NPZ).
    mf_mos_mask_path: Optional[str] = None


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
        pa.field("mf_mos_mask_path", pa.string()),
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
    # Which Step-B method produced this chunk's artifacts ("aw"|"mos"|"union").
    # None on summaries written before the column existed. The orchestrator's
    # skip check compares it against the current run's cfg.segmentation so
    # switching --seg never silently serves another method's artifacts.
    segmentation_method: Optional[str] = None
    # Sweeps Step B had no usable MF-MOS mask for (mos/union only; None for
    # aw). Their movers are left static (mos) or dropped (union) — chunks
    # where this is high (e.g. residual warm-up at chunk starts) lose recall.
    seg_n_sweeps_no_mask: Optional[int] = None
    # Dynamic candidates removed by the AW-static veto (union only).
    union_n_points_vetoed: Optional[int] = None
    # Dynamic candidates removed by the ground-height veto (union only).
    union_n_points_ground_vetoed: Optional[int] = None
    # Dynamic points removed by the post-veto motion filter (union only; None
    # when the filter is disabled). persistence = voxel occupied across too
    # many sweeps; coherence = not part of a multi-sweep cluster track.
    motion_filter_n_persistence_dropped: Optional[int] = None
    motion_filter_n_coherence_dropped: Optional[int] = None
    # MF-MOS step stats — None when MF-MOS didn't run (`--seg aw`).
    mf_mos_n_processed: Optional[int] = None
    mf_mos_n_skipped: Optional[int] = None  # failures (pose gap, empty, infer error)
    mf_mos_n_unsupported: Optional[int] = None  # scanner below MIN_BEAMS; by design
    mf_mos_n_points_moving: Optional[int] = None
    # Step F motion-proposal stats — None until `proposals` has run on the
    # chunk (it rewrites this row in place after Step B/C wrote it).
    # n_points_proposal: points with any source bit set (the recall union).
    # n_clusters_moving: clusters with motion_score > 1 (Chen's criterion).
    n_points_proposal: Optional[int] = None
    n_clusters: Optional[int] = None
    n_clusters_moving: Optional[int] = None


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
        pa.field("segmentation_method", pa.string()),
        pa.field("seg_n_sweeps_no_mask", pa.int64()),
        pa.field("union_n_points_vetoed", pa.int64()),
        pa.field("union_n_points_ground_vetoed", pa.int64()),
        pa.field("motion_filter_n_persistence_dropped", pa.int64()),
        pa.field("motion_filter_n_coherence_dropped", pa.int64()),
        pa.field("mf_mos_n_processed", pa.int64()),
        pa.field("mf_mos_n_skipped", pa.int64()),
        pa.field("mf_mos_n_unsupported", pa.int64()),
        pa.field("mf_mos_n_points_moving", pa.int64()),
        pa.field("n_points_proposal", pa.int64()),
        pa.field("n_clusters", pa.int64()),
        pa.field("n_clusters_moving", pa.int64()),
    ]
)


# ---------------------------------------------------------------------------
# lidar_preprocessing Step F — motion proposals (per-frame clusters).
# ---------------------------------------------------------------------------


class MotionClusterRow(BaseModel):
    """One row per moving-object proposal cluster in motion_clusters.parquet.

    Recall-oriented and false-positive tolerant: nothing is dropped on motion
    evidence. Every row carries soft features (motion_score, track_life,
    n_sources, frac_seg_dynamic, frac_persistent) for downstream stages to
    threshold. Box columns follow ProposalRow's cx/cy/cz/w/l/h/heading so a
    cluster maps 1:1 onto a proposal (provenance "lidar_mos").

    track_hint_id is chunk-local and NOT an identity — it is the geometry-only
    track the motion score was computed on. The tracking component re-tracks
    proposals and must not reuse it.
    """

    model_config = ConfigDict(extra="forbid")

    bag_id: str
    chunk_id: str
    frame_id: int
    reference_timestamp_ns: int
    sweep_ids: str  # JSON list[int] — the sweeps fused into this frame
    cluster_id: int  # unique within the chunk; matches motion_proposals.npz
    track_hint_id: int
    cx: float
    cy: float
    cz: float
    w: float
    l: float  # noqa: E741 — domain term: length, matches parquet w/l/h column triplet
    h: float
    heading: float  # radians, world frame, undirected (mod pi)
    n_points: int  # seed + attached points
    n_seed_points: int
    source_bits: int  # OR of member points' source bits
    n_sources: int  # popcount of source_bits, excluding BOX_FILL
    frac_seg_dynamic: float  # fraction of members the seg method called dynamic
    frac_persistent: float  # fraction of members in long-occupied voxels
    track_life: int  # frames in this cluster's track
    net_displacement_m: float  # BEV, smoothed first → last of the track
    motion_score: float  # net_displacement_m / max side (Chen: moving if > 1)
    box_filled: bool


MOTION_CLUSTER_SCHEMA = pa.schema(
    [
        pa.field("bag_id", pa.string()),
        pa.field("chunk_id", pa.string()),
        pa.field("frame_id", pa.int64()),
        pa.field("reference_timestamp_ns", pa.int64()),
        pa.field("sweep_ids", pa.string()),
        pa.field("cluster_id", pa.int64()),
        pa.field("track_hint_id", pa.int64()),
        pa.field("cx", pa.float64()),
        pa.field("cy", pa.float64()),
        pa.field("cz", pa.float64()),
        pa.field("w", pa.float64()),
        pa.field("l", pa.float64()),
        pa.field("h", pa.float64()),
        pa.field("heading", pa.float64()),
        pa.field("n_points", pa.int64()),
        pa.field("n_seed_points", pa.int64()),
        pa.field("source_bits", pa.int64()),
        pa.field("n_sources", pa.int64()),
        pa.field("frac_seg_dynamic", pa.float64()),
        pa.field("frac_persistent", pa.float64()),
        pa.field("track_life", pa.int64()),
        pa.field("net_displacement_m", pa.float64()),
        pa.field("motion_score", pa.float64()),
        pa.field("box_filled", pa.bool_()),
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
    l: float  # noqa: E741 — domain term: length, matches parquet w/l/h column triplet
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
    cam_id: str  # camera name, e.g. "CAM_FRONT"
    cls: str  # "vehicle" | "pedestrian" | "cyclist"
    score: float
    frames_present: str  # JSON list[int] — camera_seq values where mask is present
    mask_path: str  # path to directory of per-frame mask PNG files
    dino_feature_path: Optional[str] = None  # DINOv2 embedding NPZ
    global_object_id: Optional[str] = None  # cross-camera identity
    # detector + SAM2 fields
    raw_phrase: str = ""  # raw detector label before canonicalisation
    det_score: float = 0.0  # detector (× SAM2 mask) confidence
    discovery_score: float = 0.0  # detector / Florence-2 confidence
    centroid_depth_m: float = (
        0.0  # metric depth at mask centroid (used for cross-cam merge)
    )
    tracker_backend: str = "sam2"  # tracker that produced this masklet


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
        pa.field("raw_phrase", pa.string()),
        pa.field("det_score", pa.float64()),
        pa.field("discovery_score", pa.float64()),
        pa.field("centroid_depth_m", pa.float64()),
        pa.field("tracker_backend", pa.string()),
    ]
)


# ---------------------------------------------------------------------------
# perception_2d v2 — depth branch (Depth Anything V2 + LiDAR affine align).
# ---------------------------------------------------------------------------


class DepthFrameRow(BaseModel):
    """Per-frame metadata for the depth_2d artifact (the actual arrays live in npz).

    fit_status: 0=ok, 1=fell back to prior-frame affine, 2=fit failed entirely.
    """

    model_config = ConfigDict(extra="forbid")

    bag_id: str
    chunk_id: str
    cam_id: str
    frame_seq: int
    affine_a: float  # scale: d_lidar = a * d_da + b
    affine_b: float  # offset
    n_anchors: int  # (d_lidar, d_da) pairs before RANSAC
    n_inliers: int  # RANSAC inliers
    rmse_inliers_m: float
    fit_status: int


DEPTH_FRAME_SCHEMA = pa.schema(
    [
        pa.field("bag_id", pa.string()),
        pa.field("chunk_id", pa.string()),
        pa.field("cam_id", pa.string()),
        pa.field("frame_seq", pa.int64()),
        pa.field("affine_a", pa.float64()),
        pa.field("affine_b", pa.float64()),
        pa.field("n_anchors", pa.int64()),
        pa.field("n_inliers", pa.int64()),
        pa.field("rmse_inliers_m", pa.float64()),
        pa.field("fit_status", pa.int32()),
    ]
)


# ---------------------------------------------------------------------------
# semantic_lifting — per-sweep diagnostics (actual labels in npz).
# ---------------------------------------------------------------------------


class LiftedStatsRow(BaseModel):
    """One row per sweep in lifted_stats.parquet."""

    model_config = ConfigDict(extra="forbid")

    sweep_id: str
    bag_id: str
    chunk_id: str
    n_points_total: int
    n_points_labeled: int
    n_points_in_any_mask: int
    n_points_failed_visibility: int
    n_points_disagreement: int
    mean_confidence_labeled: float
    n_cameras_used: int


LIFTED_STATS_SCHEMA = pa.schema(
    [
        pa.field("sweep_id", pa.string()),
        pa.field("bag_id", pa.string()),
        pa.field("chunk_id", pa.string()),
        pa.field("n_points_total", pa.int64()),
        pa.field("n_points_labeled", pa.int64()),
        pa.field("n_points_in_any_mask", pa.int64()),
        pa.field("n_points_failed_visibility", pa.int64()),
        pa.field("n_points_disagreement", pa.int64()),
        pa.field("mean_confidence_labeled", pa.float64()),
        pa.field("n_cameras_used", pa.int64()),
    ]
)


# ---------------------------------------------------------------------------
# proposal_generation — 3D box proposals (one per LiDAR sweep × object).
# ---------------------------------------------------------------------------


class ProposalRow(BaseModel):
    """One row per 3D box proposal for a single sweep.

    supporting_cam_ids and supporting_masklet_ids are JSON-encoded lists.
    provenance identifies the source: "lidar_detector", "slf", "lidar_mos"
    (lidar_preprocessing Step F motion clusters — MotionClusterRow shares the
    box columns), or "fused".
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
    l: float  # noqa: E741 — domain term: length, matches parquet w/l/h column triplet
    h: float
    heading: float  # radians, world frame
    cls: str  # "vehicle" | "pedestrian" | "cyclist"
    score: float
    provenance: str  # "lidar_detector" | "slf" | "lidar_mos" | "fused"
    lidar_point_count: Optional[int] = None
    supporting_cam_ids: str = "[]"  # JSON list[str] of cam_id
    supporting_masklet_ids: str = "[]"  # JSON list[str] of masklet_id


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

    supporting_cam_ids and supporting_masklet_ids are JSON-encoded lists.
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
    l: float  # noqa: E741 — domain term: length, matches parquet w/l/h column triplet
    h: float
    heading: float  # radians, world frame
    cls: str  # "vehicle" | "pedestrian" | "cyclist"
    score: float
    supporting_cam_ids: str = "[]"  # JSON list[str]
    supporting_masklet_ids: str = "[]"  # JSON list[str]
    dino_feature_path: Optional[str] = None


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
    l: float  # noqa: E741 — domain term: length, matches parquet w/l/h column triplet
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
