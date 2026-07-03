"""Artifact I/O helpers for semantic_lifting.

Reads upstream artifacts from ingest + lidar_preprocessing + perception_2d.
Writes: lifted_labels/<sweep_id>.npz, lifted_stats.parquet.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from wato_common.artifact_store import (
    calibration_path,
    chunks_index_path,
    depth_2d_path,
    frame_index_path,
    lidar_proc_index_path,
    lifted_labels_path,
    lifted_stats_path,
    local_path,
    masks_2d_dir,
    tracklets_2d_path,
)
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import LIFTED_STATS_SCHEMA, LiftedStatsRow
from wato_semantic_lifting.temporal_match import CameraFrameRef


@dataclass
class LidarSweepInfo:
    sweep_id: int
    timestamp_ns: int
    world_path: str
    dynamic_mask_path: str


@dataclass
class CalibrationInfo:
    K: np.ndarray        # (3, 3) float64 intrinsic
    ego_T_cam: np.ndarray   # (4, 4) float64 SE3
    ego_T_lidar: np.ndarray  # (4, 4) float64 SE3


def load_chunks(bag_id: str) -> list[dict]:
    return read_rows(chunks_index_path(bag_id))


def load_sweeps(bag_id: str, chunk_id: str) -> list[LidarSweepInfo]:
    """Return LiDAR sweep metadata for one chunk."""
    rows = read_rows(lidar_proc_index_path(bag_id, chunk_id))
    result = []
    for r in rows:
        if r.get("valid") is False:
            continue
        result.append(
            LidarSweepInfo(
                sweep_id=int(r["sweep_id"]),
                timestamp_ns=int(r.get("timestamp_ns", 0)),
                world_path=local_path(str(r["world_path"])),
                dynamic_mask_path=local_path(str(r["dynamic_mask_path"])),
            )
        )
    return result


def load_frame_refs(bag_id: str, chunk_id: str) -> list[CameraFrameRef]:
    """Return camera frame references with poses for temporal matching."""
    rows = read_rows(frame_index_path(bag_id, chunk_id))
    result = []
    for r in rows:
        if not r.get("valid_camera", False) or not r.get("valid_pose", False):
            continue
        flat = r.get("world_T_ego_flat")
        if flat is None:
            continue
        world_T_ego = np.asarray(list(flat), dtype=np.float64).reshape(4, 4)
        result.append(
            CameraFrameRef(
                cam_id=str(r["cam_id"]),
                camera_seq=int(r["camera_seq"]),
                timestamp_ns=int(r.get("timestamp_ns", 0)),
                world_T_ego=world_T_ego,
            )
        )
    return result


def load_calibration(bag_id: str) -> dict[str, CalibrationInfo]:
    """Load per-camera K, ego_T_cam, and shared ego_T_lidar from calibration.json."""
    calib_file = local_path(calibration_path(bag_id))
    with open(calib_file, "r", encoding="utf-8") as fh:
        calib = json.load(fh)

    ego_T_lidar = np.asarray(
        calib.get("lidar", {}).get("ego_T_lidar", np.eye(4).tolist()),
        dtype=np.float64,
    )

    result: dict[str, CalibrationInfo] = {}
    for cam_id, entry in calib.get("cameras", {}).items():
        K = np.asarray(entry["K"], dtype=np.float64)
        ego_T_cam = np.asarray(entry["ego_T_cam"], dtype=np.float64)
        result[cam_id] = CalibrationInfo(K=K, ego_T_cam=ego_T_cam, ego_T_lidar=ego_T_lidar)
    return result


def load_all_lidar_points(sweep: LidarSweepInfo) -> Optional[np.ndarray]:
    """Return (N, 3) float64 world-frame points for this sweep (static + dynamic)."""
    if not os.path.exists(sweep.world_path):
        return None
    data = np.load(sweep.world_path)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1)
    return xyz.astype(np.float64)


def load_depth_map(
    bag_id: str, chunk_id: str, cam_id: str, camera_seq: int
) -> Optional[dict]:
    """Load the depth_2d artifact for one camera frame.

    Returns dict with keys: depth_m (H,W float32), fit_status (int).
    Returns None if the artifact is missing.
    """
    path = local_path(depth_2d_path(bag_id, chunk_id, cam_id, camera_seq))
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return {
        "depth_m": data["depth_m"].astype(np.float32),
        "fit_status": int(data.get("fit_status", 2)),
    }


def load_masks_for_frame(
    bag_id: str, chunk_id: str, cam_id: str, camera_seq: int
) -> tuple[list[np.ndarray], list[str], list[str], list[float]]:
    """Load per-detection masks for one camera frame.

    Returns:
        masks: list of (H, W) bool arrays.
        instance_ids: list of global_object_id strings.
        classes: list of cls strings.
        scores: list of detector confidences (``det_score``, falling back to the
            masklet ``score``) used to weight this mask's vote.
    """
    rows = read_rows(tracklets_2d_path(bag_id, chunk_id))
    cam_rows = [
        r for r in rows
        if r.get("cam_id") == cam_id
        and camera_seq in _decode_frames_present(r.get("frames_present", []))
    ]

    masks_base = local_path(masks_2d_dir(bag_id, chunk_id))
    masks, instance_ids, classes, scores = [], [], [], []
    for r in cam_rows:
        masklet_id = str(r.get("masklet_id", ""))
        mask_path = os.path.join(masks_base, masklet_id, f"{camera_seq:06d}.png")
        if not os.path.exists(mask_path):
            continue
        try:
            from PIL import Image as PILImage

            mask = np.array(PILImage.open(mask_path)).astype(bool)
        except Exception:  # noqa: BLE001
            continue
        masks.append(mask)
        instance_ids.append(str(r.get("global_object_id") or r.get("masklet_id", "")))
        classes.append(str(r.get("cls", "")))
        det_score = r.get("det_score")
        if det_score is None:
            det_score = r.get("score", 0.0)
        scores.append(float(det_score or 0.0))

    return masks, instance_ids, classes, scores


def _decode_frames_present(encoded) -> list[int]:
    """Decode the frames_present column (bytes or list) back to ints."""
    if isinstance(encoded, (list, np.ndarray)):
        return [int(x) for x in encoded]
    if isinstance(encoded, (bytes, bytearray)):
        return list(np.frombuffer(encoded, dtype=np.int32).tolist())
    return []


def write_lifted_labels(
    bag_id: str,
    chunk_id: str,
    sweep_id: int,
    point_indices: np.ndarray,
    instance_ids: list[str],
    classes: list[str],
    confidences: np.ndarray,
    n_supporting: np.ndarray,
    n_disagreeing: np.ndarray,
) -> None:
    """Write the per-point label assignment for one sweep."""
    path = local_path(lifted_labels_path(bag_id, chunk_id, str(sweep_id)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        point_indices=point_indices.astype(np.int32),
        instance_ids=np.array(instance_ids, dtype=object),
        classes=np.array(classes, dtype=object),
        confidences=confidences.astype(np.float32),
        n_supporting_cameras=n_supporting.astype(np.int32),
        n_disagreeing_cameras=n_disagreeing.astype(np.int32),
    )


def write_lifted_stats(
    bag_id: str, chunk_id: str, rows: list[dict]
) -> None:
    write_table(rows, LIFTED_STATS_SCHEMA, lifted_stats_path(bag_id, chunk_id))
