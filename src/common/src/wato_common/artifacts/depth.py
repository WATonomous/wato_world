"""perception_2d metric depth: depth_2d/<cam_id>/<camera_seq:06d>.npz.

Written by perception_2d (Depth Anything V2 relative depth, affine-aligned to
static LiDAR points); read by semantic_lifting's visibility test and
perception_2d's depth viewer.

Each npz holds:

  depth_m         (H, W)  metric depth in metres, at the image's resolution
                          (stored float16 by default — config depth.output_dtype)
  affine_a        float32 d_metric = affine_a * d_relative + affine_b
  affine_b        float32
  n_anchors       int32   LiDAR / relative-depth pairs before RANSAC
  n_inliers       int32
  rmse_inliers_m  float32
  fit_status      int32   FitStatus

A frame whose fit failed entirely (FitStatus.FAILED) is never written, so a
missing file means "no usable depth" whether the frame was processed or not;
read_frame returns None for both.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np

from wato_common.artifact_store import depth_2d_path, local_path
from wato_common.artifacts import ArtifactFormatError


class FitStatus(IntEnum):
    OK = 0  # RANSAC fit on this frame's own LiDAR anchors
    FALLBACK = 1  # too few anchors; a recent frame's affine reused
    FAILED = 2  # no usable affine — never written


@dataclass(frozen=True)
class DepthFrame:
    depth_m: np.ndarray  # (H, W) metres
    affine_a: float
    affine_b: float
    n_anchors: int
    n_inliers: int
    rmse_inliers_m: float
    fit_status: FitStatus


_SCALARS = (
    "affine_a",
    "affine_b",
    "n_anchors",
    "n_inliers",
    "rmse_inliers_m",
    "fit_status",
)


def _path(bag_id: str, chunk_id: str, cam_id: str, camera_seq: int) -> str:
    return local_path(depth_2d_path(bag_id, chunk_id, cam_id, camera_seq))


def write_frame(
    bag_id: str, chunk_id: str, cam_id: str, camera_seq: int, frame: DepthFrame
) -> None:
    if frame.fit_status == FitStatus.FAILED:
        raise ValueError(
            "a FAILED depth fit is not written; readers treat absence as failure"
        )
    path = _path(bag_id, chunk_id, cam_id, camera_seq)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        depth_m=frame.depth_m,
        affine_a=np.float32(frame.affine_a),
        affine_b=np.float32(frame.affine_b),
        n_anchors=np.int32(frame.n_anchors),
        n_inliers=np.int32(frame.n_inliers),
        rmse_inliers_m=np.float32(frame.rmse_inliers_m),
        fit_status=np.int32(frame.fit_status),
    )


def exists(bag_id: str, chunk_id: str, cam_id: str, camera_seq: int) -> bool:
    return os.path.exists(_path(bag_id, chunk_id, cam_id, camera_seq))


def read_frame(
    bag_id: str, chunk_id: str, cam_id: str, camera_seq: int
) -> Optional[DepthFrame]:
    """The frame's depth (depth_m as float32), or None when it has none."""
    path = _path(bag_id, chunk_id, cam_id, camera_seq)
    if not os.path.exists(path):
        return None
    with np.load(path) as d:
        missing = [k for k in ("depth_m", *_SCALARS) if k not in d]
        if missing:
            raise ArtifactFormatError(path, f"missing arrays {missing}")
        depth_m = np.asarray(d["depth_m"], dtype=np.float32)
        s = {k: d[k].item() for k in _SCALARS}
    if depth_m.ndim != 2:
        raise ArtifactFormatError(
            path, f"depth_m has shape {depth_m.shape}, not (H, W)"
        )
    try:
        fit_status = FitStatus(int(s["fit_status"]))
    except ValueError:
        raise ArtifactFormatError(
            path, f"unknown fit_status {s['fit_status']}"
        ) from None
    return DepthFrame(
        depth_m=depth_m,
        affine_a=float(s["affine_a"]),
        affine_b=float(s["affine_b"]),
        n_anchors=int(s["n_anchors"]),
        n_inliers=int(s["n_inliers"]),
        rmse_inliers_m=float(s["rmse_inliers_m"]),
        fit_status=fit_status,
    )
