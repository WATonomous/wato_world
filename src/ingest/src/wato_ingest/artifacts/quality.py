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
from wato_ingest.decoders.poses import dense_fraction


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

    # ---- Pose density (the chunk-level requirement already passed in
    # decoders/poses.py; recorded so a bag's margin to it is visible). --------
    if len(poses) >= 2:
        pose_ts = np.sort(np.array([p["timestamp_ns"] for p in poses], dtype=np.int64))
        spacing_ms = np.diff(pose_ts) / 1e6
        metrics["pose_median_spacing_ms"] = float(np.median(spacing_ms))
        metrics["pose_max_spacing_ms"] = float(spacing_ms.max())
        metrics["pose_dense_fraction"] = dense_fraction(
            pose_ts.tolist(), cfg.pose_requirements.max_bracket_ms
        )

    # ---- Pose availability over the chunk window. ---------------------------
    if frames:
        with_pose = sum(1 for f in frames if f["valid_pose"])
        metrics["pose_availability_fraction"] = with_pose / len(frames)
        if (
            metrics["pose_availability_fraction"]
            < cfg.quality_thresholds["min_pose_availability"]
        ):
            tags.append("POSE_MISSING")

        # Sweeps (not frame_index rows — those repeat per camera) that lost
        # their pose, by pose_drop_reason family.
        dropped: dict[str, set[tuple[str, int]]] = {
            "gap": set(),
            "jump": set(),
            "span": set(),
        }
        for f in frames:
            family = _pose_drop_family(f.get("pose_drop_reason"))
            if family is not None:
                dropped[family].add((f["lidar_id"], int(f["sweep_id"])))
        metrics["pose_gap_sweeps"] = float(len(dropped["gap"]))
        metrics["pose_jump_sweeps"] = float(len(dropped["jump"]))
        metrics["pose_outside_span_sweeps"] = float(len(dropped["span"]))
        if dropped["jump"]:
            tags.append("POSE_JUMPS")

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


def _pose_drop_family(reason: str | None) -> str | None:
    """Map a frame_index pose_drop_reason to gap / jump / span (or None)."""
    if not reason:
        return None
    if reason.startswith("pose_gap_"):
        return "gap"
    if reason.startswith("pose_jump_"):
        return "jump"
    if reason == "outside_pose_span":
        return "span"
    return None


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
