"""Path conventions for the wato_world labeling pipeline — the one layout.

Everything lives under ARTIFACT_ROOT_URI as raw/<bag_id>/... (bag level) and
raw/<bag_id>/chunks/<chunk_id>/... (chunk level), whichever component wrote
it. Functions return file:// URIs (what index rows and manifests record);
`local_path` turns one into a filesystem path, and refuses remote URIs —
the pipeline only runs against a local or bind-mounted tree.

Layout only: an artifact read by another component is encoded, written and
read by its module in wato_common.artifacts. Use these functions directly only
for a component's own files, or to record a path (manifests, index rows).
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


def calibration_refinement_dir(bag_id: str) -> str:
    """Camera-only refinement; calibration.json remains ingest provenance."""
    return _join(bag_root(bag_id), "camera_calibration")


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


def motion_proposals_path(bag_id: str, chunk_id: str, sweep_id: int) -> str:
    """Per-sweep recall-oriented moving-object proposals (Step F).

    npz with `source_bits` uint8[N] and `cluster_id` int32[N], aligned to the
    sweep's world NPZ. Deterministic path, not an index column: every seg
    method rewrites lidar_proc_index.parquet, so a column would be dropped by
    whichever rewrite forgot to carry it.
    """
    return _join(
        lidar_proc_dir(bag_id, chunk_id), f"{sweep_id:06d}_motion_proposals.npz"
    )


def motion_clusters_path(bag_id: str, chunk_id: str) -> str:
    """Per-chunk proposal clusters with soft motion features (Step F)."""
    return _join(chunk_root(bag_id, chunk_id), "motion_clusters.parquet")


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


def global_iwu_path(bag_id: str) -> str:
    """Bag-level UniLiPs IWU result over global_static_map.npz (Step E).

    Separate from global_static_map.npz so the CLI's post-run re-reduce can
    never overwrite it.
    """
    return _join(bag_root(bag_id), "global_iwu.npz")


def voxel_occupancy_path(bag_id: str, chunk_id: str) -> str:
    return _join(chunk_root(bag_id, chunk_id), "voxel_occupancy.npz")


def voxel_occupancy_frame_path(bag_id: str, chunk_id: str, frame_id: int) -> str:
    return _join(
        chunk_root(bag_id, chunk_id), f"voxel_occupancy_frame_{frame_id:04d}.npz"
    )


def voxel_diag_path(bag_id: str, chunk_id: str) -> str:
    """Per-voxel diagnostics for classify (log_odds, n_obs, n_hits, classification).

    Unlike voxel_occupancy.npz (log_odds >= 0 only), this includes carved
    (log_odds < 0) voxels, so the viz can color dynamic points by their
    voxel's p_occ to tell threshold-edge leakage from deep-carving leakage.
    Only written when cfg.save_voxel_diagnostics is True.
    """
    return _join(chunk_root(bag_id, chunk_id), "voxel_diag.npz")


# ---------------------------------------------------------------------------
# perception_2d artifacts.
# ---------------------------------------------------------------------------
def detections_2d_path(bag_id: str, chunk_id: str) -> str:
    return _join(chunk_root(bag_id, chunk_id), "detections_2d.parquet")


def tracklets_2d_path(bag_id: str, chunk_id: str) -> str:
    return _join(chunk_root(bag_id, chunk_id), "tracklets_2d.parquet")


def masklet_embeddings_path(bag_id: str, chunk_id: str) -> str:
    return _join(chunk_root(bag_id, chunk_id), "masklet_embeddings.npz")


def masks_2d_dir(bag_id: str, chunk_id: str) -> str:
    """Directory containing per-masklet mask PNGs.

    Layout: masks_2d/<masklet_id>/<camera_seq:06d>.png
    """
    return _join(chunk_root(bag_id, chunk_id), "masks_2d")


def depth_2d_dir(bag_id: str, chunk_id: str) -> str:
    """Directory containing per-camera per-frame metric depth npz files.

    Layout: depth_2d/<cam_id>/<frame_seq:06d>.npz
    Arrays in each npz: depth_m (H,W float16), disp_scale, disp_shift, n_anchors,
    n_inliers, rmse_inliers_m, fit_status.
    """
    return _join(chunk_root(bag_id, chunk_id), "depth_2d")


def depth_2d_path(bag_id: str, chunk_id: str, cam_id: str, frame_seq: int) -> str:
    return _join(depth_2d_dir(bag_id, chunk_id), cam_id, f"{frame_seq:06d}.npz")


def compare_dir(bag_id: str) -> str:
    """perception_2d compare's review files; read by people, not components."""
    return _join(bag_root(bag_id), "compare")


def ego_mask_dir(bag_id: str) -> str:
    """perception_2d ego-mask's evidence and draft; read by people, not components."""
    return _join(bag_root(bag_id), "ego_mask")


def depth_stats_path(bag_id: str, chunk_id: str) -> str:
    """Parquet of DepthFrameRow rows — one per (cam, frame). Not written yet."""
    return _join(chunk_root(bag_id, chunk_id), "depth_stats.parquet")


# ---------------------------------------------------------------------------
# detection_3d artifacts (format: wato_common.artifacts.detections_3d).
# ---------------------------------------------------------------------------
def detections_3d_path(bag_id: str, chunk_id: str) -> str:
    return _join(chunk_root(bag_id, chunk_id), "detections_3d.parquet")


def detections_3d_frames_path(bag_id: str, chunk_id: str) -> str:
    return _join(chunk_root(bag_id, chunk_id), "detections_3d_frames.parquet")


# ---------------------------------------------------------------------------
# proposal_generation artifacts.
# ---------------------------------------------------------------------------
def proposals_path(bag_id: str, chunk_id: str) -> str:
    return _join(chunk_root(bag_id, chunk_id), "proposals.parquet")


def proposal_refinement_path(bag_id: str, chunk_id: str) -> str:
    """proposal_generation's diagnostic table of all lifted-mask matches."""
    return _join(chunk_root(bag_id, chunk_id), "proposal_refinement.parquet")


# ---------------------------------------------------------------------------
# semantic_lifting artifacts.
# ---------------------------------------------------------------------------
def semantic_lifting_dir(bag_id: str, chunk_id: str) -> str:
    return _join(chunk_root(bag_id, chunk_id), "semantic_lifting")


def lifted_labels_path(bag_id: str, chunk_id: str, sweep_id: int) -> str:
    """Per-sweep labels npz (format: wato_common.artifacts.lifted_labels)."""
    return _join(
        semantic_lifting_dir(bag_id, chunk_id), "lifted_labels", f"{sweep_id:06d}.npz"
    )


def lifted_instances_path(bag_id: str, chunk_id: str) -> str:
    """Parquet of LiftedInstanceRow rows — one per lifted instance."""
    return _join(semantic_lifting_dir(bag_id, chunk_id), "lifted_instances.parquet")


def lifted_candidates_path(bag_id: str, chunk_id: str, sweep_id: int) -> str:
    """Competing lifted point-instance claims; see artifacts.lifted_candidates."""
    return _join(
        semantic_lifting_dir(bag_id, chunk_id),
        "lifted_candidates",
        f"{sweep_id:06d}.npz",
    )


def lifted_stats_path(bag_id: str, chunk_id: str) -> str:
    """Parquet of LiftedStatsRow rows — one per sweep in this chunk."""
    return _join(semantic_lifting_dir(bag_id, chunk_id), "lifted_stats.parquet")


# ---------------------------------------------------------------------------
# tracking artifacts (bag-level — spans all chunks).
# ---------------------------------------------------------------------------
def tracks_path(bag_id: str) -> str:
    return _join(bag_root(bag_id), "tracks.parquet")


def track_summaries_path(bag_id: str) -> str:
    return _join(bag_root(bag_id), "track_summaries.parquet")


# ---------------------------------------------------------------------------
# label_refinement artifacts (bag-level).
# ---------------------------------------------------------------------------
def refined_labels_path(bag_id: str) -> str:
    return _join(bag_root(bag_id), "refined_labels.parquet")


# ---------------------------------------------------------------------------
# evaluation artifacts (bag-level) — imported ground truth, not pipeline output.
# ---------------------------------------------------------------------------
def gt_boxes_path(bag_id: str) -> str:
    return _join(bag_root(bag_id), "gt_boxes.parquet")


# ---------------------------------------------------------------------------
# URI ↔ local path helpers. rosbag2_py, numpy, PIL and pyarrow's local writer
# all need concrete paths.
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
