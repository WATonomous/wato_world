"""Static intensity clouds in ego coordinates at each image's own time."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from wato_common.artifact_store import local_path
from wato_common.artifacts.calibration import CameraCalibration
from wato_common.artifacts.frame_index import CameraFrame
from wato_common.artifacts.lidar_sweeps import read_dynamic_mask, read_world
from wato_common.storage import artifact_root

from wato_camera_calibration.config import SamplingConfig


def relocated(path: str) -> Path:
    path = path.removeprefix("file://")
    prefix = "/data/artifacts/"
    if path.startswith(prefix):
        return Path(local_path(artifact_root())) / path[len(prefix) :]
    return Path(path)


@dataclass
class Sample:
    timestamp_ns: int
    block: int
    split: str
    image_path: str
    world_T_ego: np.ndarray
    xyz: np.ndarray
    intensity: np.ndarray
    gray: np.ndarray


def world_to_ego(xyz: np.ndarray, world_T_ego: np.ndarray) -> np.ndarray:
    return (xyz - world_T_ego[:3, 3]) @ world_T_ego[:3, :3]


def voxel_mean(xyz: np.ndarray, intensity: np.ndarray, size: float):
    _, inverse = np.unique(
        np.floor(xyz / size).astype(np.int64), axis=0, return_inverse=True
    )
    counts = np.bincount(inverse)
    points = np.column_stack(
        [np.bincount(inverse, weights=xyz[:, i]) / counts for i in range(3)]
    )
    values = np.bincount(inverse, weights=intensity) / counts
    return points, values


def build_sample(
    frame: CameraFrame,
    rows: list,
    cam: CameraCalibration,
    cfg: SamplingConfig,
    block: int,
    split: str,
) -> tuple[Sample, set[Path]]:
    if frame.world_T_ego is None:
        raise ValueError("image has no trusted pose")
    image_path = relocated(frame.image_path)
    with Image.open(image_path) as image:
        model = cam.model_for_image(image.size)
        gray = np.asarray(image.convert("L")).copy()
    sources = {image_path}
    xyzs, values = [], []
    for row in rows:
        world_path = relocated(row.world_path)
        mask_path = relocated(row.dynamic_mask_path)
        world = read_world(str(world_path))
        mask = read_dynamic_mask(str(mask_path), n_points=world.n_points)
        if world.intensity is None:
            raise ValueError(f"{world_path}: intensity is missing")
        sources.update((world_path, mask_path))
        xyz = world_to_ego(world.xyz, frame.world_T_ego)
        distance = np.linalg.norm(xyz, axis=1)
        keep = (
            ~mask
            & np.isfinite(xyz).all(axis=1)
            & np.isfinite(world.intensity)
            & (distance >= cfg.min_range_m)
            & (distance <= cfg.max_range_m)
        )
        xyzs.append(xyz[keep])
        values.append(world.intensity[keep])
    if not xyzs:
        raise ValueError("no reference LiDAR sweeps in the accumulation window")
    xyz, intensity = voxel_mean(
        np.concatenate(xyzs), np.concatenate(values), cfg.voxel_m
    )
    # Keep a margin substantially larger than the expected mounting correction.
    # Culling here is only a workload bound; native visibility is updated during fit.
    uv, depth, _ = model.project(xyz, np.linalg.inv(cam.ego_T_cam))
    height, width = gray.shape
    margin = 0.2 * max(width, height)
    keep = (
        (depth > 0.5)
        & np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= -margin)
        & (uv[:, 0] < width + margin)
        & (uv[:, 1] >= -margin)
        & (uv[:, 1] < height + margin)
    )
    xyz, intensity = xyz[keep], intensity[keep]
    if len(xyz) < cfg.min_points_per_view:
        raise ValueError(f"too few static overlapping returns: {len(xyz)}")
    low, high = np.percentile(intensity, [1, 99])
    if high - low < cfg.min_intensity_span:
        raise ValueError("LiDAR intensity has insufficient variation")
    intensity = np.clip((intensity - low) / (high - low), 0.0, 1.0)
    if len(xyz) > cfg.max_points_per_view:
        take = np.sort(
            np.random.default_rng(frame.timestamp_ns).choice(
                len(xyz), cfg.max_points_per_view, replace=False
            )
        )
        xyz, intensity = xyz[take], intensity[take]
    return Sample(
        frame.timestamp_ns,
        block,
        split,
        str(image_path),
        frame.world_T_ego,
        xyz,
        intensity,
        gray,
    ), sources
