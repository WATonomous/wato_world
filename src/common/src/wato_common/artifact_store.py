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


# ---------------------------------------------------------------------------
# proposal_generation artifacts.
# ---------------------------------------------------------------------------
def proposals_path(bag_id: str, chunk_id: str) -> str:
    return _join(chunk_root(bag_id, chunk_id), "proposals.parquet")


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
