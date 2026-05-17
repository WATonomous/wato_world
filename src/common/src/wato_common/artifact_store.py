"""Path conventions for the wato_world labeling pipeline.

Every component's I/O goes through here so paths live in one place. Returns
URIs (file:///..., s3://...) suitable for fsspec; resolve to concrete
filesystem paths via `local_path` when you must hand them to a non-fsspec
library (e.g. PIL, pyarrow's local writer).
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

from wato_common.storage import artifact_root


def _join(*parts: str) -> str:
    """URI-safe join: avoids `os.path.join` collapsing the `file://` scheme."""
    head = parts[0].rstrip("/")
    tail = "/".join(p.strip("/") for p in parts[1:] if p)
    return f"{head}/{tail}" if tail else head


# ---------------------------------------------------------------------------
# Bag-level artifacts.
# ---------------------------------------------------------------------------
def bag_root(bag_id: str) -> str:
    return _join(artifact_root(), "raw", bag_id)


def bag_meta_path(bag_id: str) -> str:
    return _join(bag_root(bag_id), "bag_meta.json")


def calibration_path(bag_id: str) -> str:
    return _join(bag_root(bag_id), "calibration.json")


def chunks_index_path(bag_id: str) -> str:
    return _join(bag_root(bag_id), "chunks", "index.parquet")


# ---------------------------------------------------------------------------
# Chunk-level artifacts.
# ---------------------------------------------------------------------------
def chunk_root(bag_id: str, chunk_id: str) -> str:
    return _join(bag_root(bag_id), "chunks", chunk_id)


def camera_dir(bag_id: str, chunk_id: str, cam_id: str) -> str:
    return _join(chunk_root(bag_id, chunk_id), f"cam_{cam_id}")


def camera_image_path(
    bag_id: str, chunk_id: str, cam_id: str, seq: int, ext: str
) -> str:
    return _join(camera_dir(bag_id, chunk_id, cam_id), f"{seq:06d}.{ext.lstrip('.')}")


def lidar_dir(bag_id: str, chunk_id: str, lidar_id: str) -> str:
    return _join(chunk_root(bag_id, chunk_id), "lidar", lidar_id)


def lidar_sweep_path(bag_id: str, chunk_id: str, lidar_id: str, sweep_id: int) -> str:
    return _join(lidar_dir(bag_id, chunk_id, lidar_id), f"{sweep_id:06d}.npz")


def poses_path(bag_id: str, chunk_id: str) -> str:
    return _join(chunk_root(bag_id, chunk_id), "poses.parquet")


def camera_frames_path(bag_id: str, chunk_id: str) -> str:
    return _join(chunk_root(bag_id, chunk_id), "camera_frames.parquet")


def lidar_sweeps_path(bag_id: str, chunk_id: str) -> str:
    return _join(chunk_root(bag_id, chunk_id), "lidar_sweeps.parquet")


def frame_index_path(bag_id: str, chunk_id: str) -> str:
    return _join(chunk_root(bag_id, chunk_id), "frame_index.parquet")


def quality_path(bag_id: str, chunk_id: str) -> str:
    return _join(chunk_root(bag_id, chunk_id), "quality.json")


def manifest_path(bag_id: str, chunk_id: str) -> str:
    return _join(chunk_root(bag_id, chunk_id), "manifest.json")


# ---------------------------------------------------------------------------
# lidar_preprocessing artifacts.
# ---------------------------------------------------------------------------
def lidar_proc_dir(bag_id: str, chunk_id: str) -> str:
    return _join(chunk_root(bag_id, chunk_id), "lidar_proc")


def lidar_world_path(bag_id: str, chunk_id: str, sweep_id: int) -> str:
    return _join(lidar_proc_dir(bag_id, chunk_id), f"{sweep_id:06d}_world.npz")


def dynamic_mask_path(bag_id: str, chunk_id: str, sweep_id: int) -> str:
    return _join(lidar_proc_dir(bag_id, chunk_id), f"{sweep_id:06d}_dynamic_mask.npy")


def lidar_proc_index_path(bag_id: str, chunk_id: str) -> str:
    return _join(chunk_root(bag_id, chunk_id), "lidar_proc_index.parquet")


def lidar_proc_summary_path(bag_id: str, chunk_id: str) -> str:
    return _join(chunk_root(bag_id, chunk_id), "lidar_proc_summary.parquet")


def static_map_path(bag_id: str, chunk_id: str) -> str:
    return _join(chunk_root(bag_id, chunk_id), "static_map.npz")


def dynamic_map_path(bag_id: str, chunk_id: str) -> str:
    return _join(chunk_root(bag_id, chunk_id), "dynamic_map.npz")


def ground_path(bag_id: str, chunk_id: str) -> str:
    return _join(chunk_root(bag_id, chunk_id), "ground.npz")


def global_static_map_path(bag_id: str) -> str:
    return _join(bag_root(bag_id), "global_static_map.npz")


def global_ground_path(bag_id: str) -> str:
    return _join(bag_root(bag_id), "global_ground.npz")


def voxel_occupancy_path(bag_id: str, chunk_id: str) -> str:
    return _join(chunk_root(bag_id, chunk_id), "voxel_occupancy.npz")


def voxel_occupancy_frame_path(bag_id: str, chunk_id: str, frame_id: int) -> str:
    return _join(chunk_root(bag_id, chunk_id), f"voxel_occupancy_frame_{frame_id:04d}.npz")


# ---------------------------------------------------------------------------
# perception_2d artifacts.
# ---------------------------------------------------------------------------
def detections_2d_path(bag_id: str, chunk_id: str) -> str:
    return _join(chunk_root(bag_id, chunk_id), "detections_2d.parquet")


def tracklets_2d_path(bag_id: str, chunk_id: str) -> str:
    return _join(chunk_root(bag_id, chunk_id), "tracklets_2d.parquet")


def masks_2d_dir(bag_id: str, chunk_id: str) -> str:
    """Directory containing per-masklet mask PNGs.

    Layout: masks_2d/<masklet_id>/<camera_seq:06d>.png
    """
    return _join(chunk_root(bag_id, chunk_id), "masks_2d")


def depth_2d_dir(bag_id: str, chunk_id: str) -> str:
    """Directory containing per-camera Depth Anything V2 depth maps.

    Layout: depth_2d/<cam_id>/<camera_seq:06d>.npy (fp16 raw relative depth).
    Metric scale/shift lives in depth_index.parquet alongside.
    """
    return _join(chunk_root(bag_id, chunk_id), "depth_2d")


def depth_2d_path(bag_id: str, chunk_id: str, cam_id: str, camera_seq: int) -> str:
    return _join(depth_2d_dir(bag_id, chunk_id), cam_id, f"{camera_seq:06d}.npy")


def depth_index_path(bag_id: str, chunk_id: str) -> str:
    """Per-frame DA V2 scale/shift index — see DEPTH_INDEX_SCHEMA."""
    return _join(chunk_root(bag_id, chunk_id), "depth_index.parquet")


# ---------------------------------------------------------------------------
# proposal_generation artifacts.
# ---------------------------------------------------------------------------
def proposals_path(bag_id: str, chunk_id: str) -> str:
    return _join(chunk_root(bag_id, chunk_id), "proposals.parquet")


def proposal_diagnostics_path(bag_id: str, chunk_id: str) -> str:
    """Optional per-chunk verbose diagnostics (per-detector scores, SLF
    convergence trajectories, etc.) for researcher inspection."""
    return _join(chunk_root(bag_id, chunk_id), "proposal_diagnostics.parquet")


def pseudo_lidar_dir(bag_id: str, chunk_id: str) -> str:
    """Optional per-detection lifted-depth points (debug only — not consumed
    downstream).  Layout: pseudo_lidar/<detection_id>.npz.  Disabled unless
    proposal_generation runs with --debug-pseudo-lidar."""
    return _join(chunk_root(bag_id, chunk_id), "pseudo_lidar")


# ---------------------------------------------------------------------------
# tracking artifacts (bag-level — spans all chunks).
# ---------------------------------------------------------------------------
def tracks_path(bag_id: str) -> str:
    return _join(bag_root(bag_id), "tracks.parquet")


def tracks_forward_path(bag_id: str) -> str:
    """Forward-pass output of bidirectional tracking (pre-merge)."""
    return _join(bag_root(bag_id), "tracks_forward.parquet")


def tracks_backward_path(bag_id: str) -> str:
    """Backward-pass output of bidirectional tracking (pre-merge)."""
    return _join(bag_root(bag_id), "tracks_backward.parquet")


# ---------------------------------------------------------------------------
# label_refinement artifacts (bag-level).
# ---------------------------------------------------------------------------
def refined_labels_path(bag_id: str) -> str:
    return _join(bag_root(bag_id), "refined_labels.parquet")


def aggregated_tracks_dir(bag_id: str) -> str:
    """Directory containing per-track body-frame aggregated NPZ clouds.

    Layout: aggregated_tracks/<track_id>.npz with fields
    {points_body, pose_history, n_frames, cls, voxel_size_m, icp_corrected}.
    """
    return _join(bag_root(bag_id), "aggregated_tracks")


def aggregated_track_path(bag_id: str, track_id: str) -> str:
    return _join(aggregated_tracks_dir(bag_id), f"{track_id}.npz")


def aggregated_tracks_index_path(bag_id: str) -> str:
    """Bag-level index of aggregated tracks — see AGGREGATED_TRACK_SCHEMA."""
    return _join(bag_root(bag_id), "aggregated_tracks_index.parquet")


# ---------------------------------------------------------------------------
# Model checkpoints (live under MODELS_ROOT, not under the artifact tree).
# These return concrete filesystem paths because the deep-learning libraries
# (torch.load, transformers, ultralytics) don't speak fsspec URIs.
# ---------------------------------------------------------------------------
def _models_root() -> str:
    """Resolve MODELS_ROOT env var; falls back to ./data/models for dev."""
    return os.environ.get("MODELS_ROOT", os.path.abspath("data/models"))


def detector_checkpoint_path(detector_name: str, filename: str) -> str:
    """LiDAR detector checkpoint, e.g. MODELS_ROOT/lidar_detectors/centerpoint.pth.

    `detector_name` is the subdirectory (e.g. "lidar_detectors", "depth_anything_v2");
    `filename` is the file inside it.
    """
    return os.path.join(_models_root(), detector_name, filename)


def shape_prior_path(class_name: str) -> str:
    """SLF shape prior NPZ for a class, e.g. MODELS_ROOT/shape_priors/shape_prior_vehicle.npz."""
    return os.path.join(_models_root(), "shape_priors", f"shape_prior_{class_name}.npz")


def models_root() -> str:
    """Public accessor for tests + downstream callers that need MODELS_ROOT."""
    return _models_root()


# ---------------------------------------------------------------------------
# URI ↔ local path helpers. Ingest needs concrete paths because rosbag2_py,
# PIL, and pyarrow's local writer don't speak fsspec URIs.
# ---------------------------------------------------------------------------
def local_path(uri: str) -> str:
    """Resolve a file:// URI to a real filesystem path.  Errors on remote URIs."""
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        return parsed.path or uri
    raise ValueError(f"local_path() called on non-local URI: {uri}")


def ensure_local_dir(uri: str) -> str:
    """Create the local directory for a file:// URI and return its path."""
    path = local_path(uri)
    os.makedirs(path, exist_ok=True)
    return path
