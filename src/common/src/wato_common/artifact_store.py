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
    return _join(
        chunk_root(bag_id, chunk_id), f"voxel_occupancy_frame_{frame_id:04d}.npz"
    )


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
    """Directory containing per-camera per-frame metric depth npz files.

    Layout: depth_2d/<cam_id>/<frame_seq:06d>.npz
    Arrays in each npz: depth_m (H,W float16), confidence (H,W float16),
    lidar_coverage (H,W bool), affine_a, affine_b, n_anchors, n_inliers,
    rmse_inliers_m, fit_status.
    """
    return _join(chunk_root(bag_id, chunk_id), "depth_2d")


def depth_2d_path(bag_id: str, chunk_id: str, cam_id: str, frame_seq: int) -> str:
    return _join(depth_2d_dir(bag_id, chunk_id), cam_id, f"{frame_seq:06d}.npz")


def depth_stats_path(bag_id: str, chunk_id: str) -> str:
    """Parquet of DepthFrameRow rows — one per (cam, frame) in this chunk."""
    return _join(chunk_root(bag_id, chunk_id), "depth_stats.parquet")


# ---------------------------------------------------------------------------
# proposal_generation artifacts.
# ---------------------------------------------------------------------------
def proposals_path(bag_id: str, chunk_id: str) -> str:
    return _join(chunk_root(bag_id, chunk_id), "proposals.parquet")


# ---------------------------------------------------------------------------
# semantic_lifting artifacts.
# ---------------------------------------------------------------------------
def semantic_lifting_dir(bag_id: str, chunk_id: str) -> str:
    return _join(chunk_root(bag_id, chunk_id), "semantic_lifting")


def lifted_labels_path(bag_id: str, chunk_id: str, sweep_id: str) -> str:
    """Per-sweep npz: point_idx, class_id, instance_id, confidence, etc."""
    return _join(semantic_lifting_dir(bag_id, chunk_id), "lifted_labels", f"{sweep_id}.npz")


def lifted_stats_path(bag_id: str, chunk_id: str) -> str:
    """Parquet of LiftedStatsRow rows — one per sweep in this chunk."""
    return _join(semantic_lifting_dir(bag_id, chunk_id), "lifted_stats.parquet")


# ---------------------------------------------------------------------------
# tracking artifacts (bag-level — spans all chunks).
# ---------------------------------------------------------------------------
def tracks_path(bag_id: str) -> str:
    return _join(bag_root(bag_id), "tracks.parquet")


# ---------------------------------------------------------------------------
# label_refinement artifacts (bag-level).
# ---------------------------------------------------------------------------
def refined_labels_path(bag_id: str) -> str:
    return _join(bag_root(bag_id), "refined_labels.parquet")


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
