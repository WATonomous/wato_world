"""Deterministic sampling across chunks, with guarded scene-block splits."""

from __future__ import annotations

import bisect
from collections import Counter
from pathlib import Path

import numpy as np

from wato_common.artifact_store import (
    chunk_root,
    chunks_index_path,
    frame_index_path,
    lidar_proc_index_path,
    local_path,
    poses_path,
)
from wato_common.artifacts.calibration import CameraCalibration
from wato_common.artifacts.frame_index import FrameIndex
from wato_common.artifacts.lidar_sweeps import ProcessedSweeps
from wato_common.io.parquet_io import read_rows

from wato_camera_calibration.config import SamplingConfig
from wato_camera_calibration.inputs import build_sample


def scene_split(timestamp_ns: int, origin_ns: int, cfg: SamplingConfig):
    elapsed = (timestamp_ns - origin_ns) / 1e9
    block = int(elapsed // cfg.block_s)
    within = elapsed - block * cfg.block_s
    if within < cfg.guard_s or within >= cfg.block_s - cfg.guard_s:
        return None
    split = ("train", "train", "train", "validation", "test")[block % 5]
    return block, split


def observations(bag_id: str, cam_id: str, lidar_id: str):
    """Each posed image and each deskewed reference sweep once, in time order.

    Overlapping chunks decode the same image/sweep; they are deduplicated by
    timestamp and LiDAR ID. Also returns the index files they were read from.
    """
    sources = {Path(local_path(chunks_index_path(bag_id)))}
    images, sweeps = {}, {}
    for chunk in read_rows(chunks_index_path(bag_id)):
        cid = chunk["chunk_id"]
        for uri in (
            frame_index_path(bag_id, cid),
            poses_path(bag_id, cid),
            lidar_proc_index_path(bag_id, cid),
        ):
            sources.add(Path(local_path(uri)))
        manifest = (
            Path(local_path(chunk_root(bag_id, cid)))
            / "manifest_lidar_preprocessing.json"
        )
        if manifest.exists():
            sources.add(manifest)
        for frame in FrameIndex.load(bag_id, cid).by_camera().get(cam_id, []):
            if frame.world_T_ego is not None:
                images.setdefault(frame.timestamp_ns, frame)
        for row in ProcessedSweeps.load(bag_id, cid).rows:
            if row.lidar_id == lidar_id and row.deskewed:
                sweeps.setdefault((row.lidar_id, row.reference_timestamp_ns), row)
    frames = sorted(images.values(), key=lambda f: f.timestamp_ns)
    rows = sorted(sweeps.values(), key=lambda r: r.reference_timestamp_ns)
    return frames, rows, sources


def sweeps_near(rows: list, timestamp_ns: int, seconds: float) -> list:
    """Time-ordered sweeps whose reference time is within ±seconds."""
    times = [r.reference_timestamp_ns for r in rows]
    start = bisect.bisect_left(times, timestamp_ns - int(seconds * 1e9))
    end = bisect.bisect_right(times, timestamp_ns + int(seconds * 1e9))
    return rows[start:end]


def collect(bag_id: str, cam: CameraCalibration, cfg: SamplingConfig):
    frames, rows, sources = observations(bag_id, cam.cam_id, cfg.lidar_id)
    rejections = Counter()
    if not frames or not rows:
        return [], sources, {"no_images_or_deskewed_reference_sweeps": 1}
    origin = frames[0].timestamp_ns
    candidates = []
    last = -(10**30)
    for frame in frames:
        split = scene_split(frame.timestamp_ns, origin, cfg)
        if split is not None and frame.timestamp_ns - last >= cfg.interval_s * 1e9:
            candidates.append((frame, *split))
            last = frame.timestamp_ns
    # Uniformly span the whole bag; do not exhaust the budget in the first chunks.
    if len(candidates) > cfg.max_views_per_camera:
        candidates = [
            candidates[i]
            for i in np.linspace(
                0, len(candidates) - 1, cfg.max_views_per_camera
            ).astype(int)
        ]
    samples = []
    for frame, block, split in candidates:
        selected = sweeps_near(rows, frame.timestamp_ns, cfg.accumulation_s)
        try:
            sample, used = build_sample(frame, selected, cam, cfg, block, split)
        except ValueError as exc:
            rejections[str(exc)] += 1
            # Changes to a failed sample's input can make it usable next time.
            from wato_camera_calibration.inputs import relocated

            sources.add(relocated(frame.image_path))
            for row in selected:
                sources.update(
                    (relocated(row.world_path), relocated(row.dynamic_mask_path))
                )
            continue
        samples.append(sample)
        sources.update(used)
    return samples, sources, dict(rejections)
