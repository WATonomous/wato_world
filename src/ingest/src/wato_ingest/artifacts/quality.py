"""Compute per-chunk quality metrics + tags from ingest artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from wato_common.artifact_store import (
    camera_frames_path,
    ensure_local_dir,
    frame_index_path,
    lidar_sweeps_path,
    local_path,
    poses_path,
    quality_path,
)
from wato_common.io.parquet_io import read_rows
from wato_ingest.config import IngestConfig


@dataclass
class QualityReport:
    bag_id: str
    chunk_id: str
    metrics: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


def compute(bag_id: str, chunk_id: str, cfg: IngestConfig) -> QualityReport:
    sweeps = read_rows(lidar_sweeps_path(bag_id, chunk_id))
    cameras = read_rows(camera_frames_path(bag_id, chunk_id))
    poses = read_rows(poses_path(bag_id, chunk_id))
    frames = read_rows(frame_index_path(bag_id, chunk_id))

    metrics: dict[str, float] = {}
    tags: list[str] = []

    # ---- camera drop fraction (rows where valid_camera = False / total). ----
    if frames:
        invalid = sum(1 for f in frames if not f["valid_camera"])
        metrics["camera_drop_fraction"] = invalid / len(frames)
        if (
            metrics["camera_drop_fraction"]
            > cfg.quality_thresholds["max_camera_drop_fraction"]
        ):
            tags.append("CAMERA_DROPS")

    # ---- camera-LiDAR offset stats (only over valid rows). ------------------
    # Stats are intentionally over valid frames only (those within the
    # max_cam_offset_ms threshold). Dropped frames appear in frame_index with
    # valid_camera=False and will show a wider signed range there — that's
    # expected, not a discrepancy.
    offsets = [
        f["camera_offset_ms"]
        for f in frames
        if f["valid_camera"] and f["camera_offset_ms"] is not None
    ]
    if offsets:
        metrics["mean_camera_lidar_offset_ms"] = float(np.mean(np.abs(offsets)))
        metrics["max_camera_lidar_offset_ms"] = float(np.max(np.abs(offsets)))

    # ---- LiDAR point count distribution. ------------------------------------
    if sweeps:
        counts = np.array([s["num_points"] for s in sweeps])
        metrics["lidar_point_count_mean"] = float(counts.mean())
        metrics["lidar_point_count_min"] = float(counts.min())
        if counts.min() < cfg.quality_thresholds["min_lidar_points"]:
            tags.append("LOW_LIDAR_POINTS")

    # ---- Ego speed.  Differences between consecutive (x, y, z, t). ----------
    if len(poses) >= 2:
        sorted_poses = sorted(poses, key=lambda r: r["timestamp_ns"])
        ts = np.array([p["timestamp_ns"] for p in sorted_poses], dtype=np.int64)
        xy = np.array([[p["x"], p["y"], p["z"]] for p in sorted_poses])
        dt_s = np.diff(ts) / 1e9
        dxy = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        speeds = dxy / np.clip(dt_s, 1e-3, None)
        metrics["ego_speed_mean"] = float(speeds.mean())
        metrics["ego_speed_max"] = float(speeds.max())
        if metrics["ego_speed_mean"] < cfg.quality_thresholds["stationary_speed_mps"]:
            tags.append("STATIONARY")

    # ---- Pose availability over the chunk window. ---------------------------
    if frames:
        with_pose = sum(1 for f in frames if f["valid_pose"])
        metrics["pose_availability_fraction"] = with_pose / len(frames)
        if (
            metrics["pose_availability_fraction"]
            < cfg.quality_thresholds["min_pose_availability"]
        ):
            tags.append("POSE_MISSING")

    # ---- Lighting (mean V channel) — sample a few images to keep this cheap.
    metrics["lighting_mean_v"] = _sample_v_channel(cameras, max_samples=12)
    if (
        metrics["lighting_mean_v"] > 0
        and metrics["lighting_mean_v"] < cfg.quality_thresholds["low_light_v_max"]
    ):
        tags.append("LOW_LIGHT")

    if not tags:
        tags = ["OK"]

    report = QualityReport(bag_id=bag_id, chunk_id=chunk_id, metrics=metrics, tags=tags)
    _write(report)
    return report


def _sample_v_channel(camera_rows: list[dict], *, max_samples: int) -> float:
    """Return mean V (HSV) over up to `max_samples` evenly spaced images.

    Returns 0.0 if no images can be opened (e.g. compressed bytes only and
    Pillow is missing).
    """
    if not camera_rows:
        return 0.0
    try:
        import numpy as _np
        from PIL import Image
    except ImportError:
        return 0.0

    samples = camera_rows[:: max(len(camera_rows) // max_samples, 1)][:max_samples]
    vs: list[float] = []
    for r in samples:
        path = local_path(r["image_path"])
        if not os.path.exists(path):
            continue
        try:
            img = Image.open(path).convert("HSV")
            arr = _np.asarray(img)
            vs.append(float(arr[..., 2].mean()))
        except Exception:
            continue
    return float(_np.mean(vs)) if vs else 0.0


def _write(report: QualityReport) -> str:
    out_uri = quality_path(report.bag_id, report.chunk_id)
    ensure_local_dir(os.path.dirname(local_path(out_uri)))
    with open(local_path(out_uri), "w", encoding="utf-8") as fh:
        json.dump(asdict(report), fh, indent=2)
    return out_uri
