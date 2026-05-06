"""Freeze a per-bag calibration JSON.

Two paths:

1. `freeze_from_bag` — read /camera_*/camera_info + /tf_static + first
   PointCloud2 per LiDAR and synthesize calibration.json.  This is the default
   the pipeline uses; the bag is the source of truth.
2. `freeze_from_file` — copy an authored calibration JSON in.  Use this only
   when the bag's CameraInfo or /tf_static is missing or wrong.

Either way the output is `raw/<bag_id>/calibration.json` with this schema:
    {
      "calibration_version": str,
      "ego_frame": str,                       # parent frame for ego_T_*
      "cameras": {
        "<cam_id>": {
          "frame_id": str,                    # CameraInfo.header.frame_id
          "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
          "distortion": [...],
          "distortion_model": str,
          "width": int, "height": int,
          "ego_T_cam": [[...]]                # 4x4 row-major, or null if unresolved
        }
      },
      "lidars": {
        "<lidar_id>": {
          "frame_id": str,                    # PointCloud2.header.frame_id
          "ego_T_lidar": [[...]]              # 4x4 row-major, or null if unresolved
        }
      },
      "static_transforms": {                  # raw dump of /tf_static for debugging
        "<parent>__<child>": [[...]]
      },
      "checks": {"sanity": str, "notes": str}
    }
"""

from __future__ import annotations

import json
import os

import numpy as np

from wato_common.artifact_store import (
    calibration_path,
    ensure_local_dir,
    local_path,
)
from wato_common.geometry import make_se3
from wato_common.io.rosbag_reader import messages

from wato_ingest.config import IngestConfig


# ---------------------------------------------------------------------------
# Pure-logic builder — exposed for unit tests.
# ---------------------------------------------------------------------------


def build_calibration_dict(
    *,
    camera_infos: dict[str, dict],
    lidar_frame_ids: dict[str, str],
    static_transforms: dict[tuple[str, str], np.ndarray],
    ego_frame: str,
    bag_id: str,
) -> dict:
    """Assemble a calibration dict from already-decoded inputs.

    `camera_infos` is `{cam_id: {"frame_id", "K", "D", "distortion_model",
                                  "width", "height"}}`.
    `lidar_frame_ids` is `{lidar_id: frame_id}`.
    `static_transforms` is `{(parent, child): SE3 4x4}`.
    """
    calib: dict = {
        "calibration_version": f"auto_from_{bag_id}",
        "ego_frame": ego_frame,
        "cameras": {},
        "lidars": {},
        "static_transforms": {},
        "checks": {"sanity": "auto", "notes": ""},
    }

    notes: list[str] = []

    for cam_id, info in camera_infos.items():
        frame_id = info["frame_id"]
        ego_T_cam = _resolve_chain(static_transforms, ego_frame, frame_id)
        if ego_T_cam is None:
            notes.append(
                f"camera {cam_id}: no /tf_static path {ego_frame} -> {frame_id}"
            )
        K_3x3 = np.asarray(info["K"], dtype=np.float64).reshape(3, 3).tolist()
        calib["cameras"][cam_id] = {
            "frame_id": frame_id,
            "K": K_3x3,
            "distortion": list(info["D"]),
            "distortion_model": info["distortion_model"],
            "width": int(info["width"]),
            "height": int(info["height"]),
            "ego_T_cam": ego_T_cam.tolist() if ego_T_cam is not None else None,
        }

    for lidar_id, frame_id in lidar_frame_ids.items():
        ego_T_lidar = _resolve_chain(static_transforms, ego_frame, frame_id)
        if ego_T_lidar is None:
            notes.append(
                f"lidar {lidar_id}: no /tf_static path {ego_frame} -> {frame_id}"
            )
        calib["lidars"][lidar_id] = {
            "frame_id": frame_id,
            "ego_T_lidar": ego_T_lidar.tolist() if ego_T_lidar is not None else None,
        }

    for (parent, child), T in static_transforms.items():
        calib["static_transforms"][f"{parent}__{child}"] = T.tolist()

    if notes:
        calib["checks"]["sanity"] = "warn"
        calib["checks"]["notes"] = "; ".join(notes)
    else:
        calib["checks"]["notes"] = "all extrinsics resolved from /tf_static"

    return calib


def _resolve_chain(
    transforms: dict[tuple[str, str], np.ndarray],
    parent: str,
    child: str,
    max_hops: int = 4,
) -> np.ndarray | None:
    """Forward-chain BFS for parent -> ... -> child in /tf_static.

    Most racks have direct base->sensor transforms, but rigs with intermediate
    frames (e.g. base -> roof_rack -> camera) need a hop or two.
    """
    if (parent, child) in transforms:
        return transforms[(parent, child)]
    visited = {parent}
    frontier: list[tuple[str, np.ndarray]] = [(parent, np.eye(4))]
    for _ in range(max_hops):
        nxt: list[tuple[str, np.ndarray]] = []
        for cur, T in frontier:
            for (p, c), Tpc in transforms.items():
                if p == cur and c not in visited:
                    Tnext = T @ Tpc
                    if c == child:
                        return Tnext
                    visited.add(c)
                    nxt.append((c, Tnext))
        frontier = nxt
        if not frontier:
            break
    return None


# ---------------------------------------------------------------------------
# Bag-reading wrapper.  Uses the rosbag2 reader; not unit-testable directly.
# ---------------------------------------------------------------------------


def freeze_from_bag(bag_path: str, bag_id: str, cfg: IngestConfig) -> str:
    """Auto-build calibration.json from the bag's CameraInfo + tf_static.

    Reads:
      - first sensor_msgs/CameraInfo per configured camera
      - first sensor_msgs/PointCloud2 per configured lidar (for frame_id)
      - all tf2_msgs/TFMessage on /tf_static

    Writes raw/<bag_id>/calibration.json and returns its URI.
    """
    info_topic_to_cam = {c.info: cam_id for cam_id, c in cfg.topics.cameras.items()}
    lidar_topic_to_id = {
        topic: lidar_id for lidar_id, topic in cfg.topics.lidars.items()
    }
    needed_camera_ids = set(info_topic_to_cam.values())
    needed_lidar_ids = set(lidar_topic_to_id.values())

    camera_infos: dict[str, dict] = {}
    lidar_frame_ids: dict[str, str] = {}
    static_transforms: dict[tuple[str, str], np.ndarray] = {}
    saw_tf_static = False

    topics_to_read = (
        list(info_topic_to_cam.keys())
        + list(lidar_topic_to_id.keys())
        + [cfg.topics.tf_static]
    )

    with messages(
        bag_path, storage_id=cfg.storage_id, topics=topics_to_read
    ) as iterator:
        for topic, msg, _ts in iterator:
            mt = type(msg).__name__
            if mt == "CameraInfo":
                cam_id = info_topic_to_cam[topic]
                if cam_id not in camera_infos:
                    camera_infos[cam_id] = {
                        "frame_id": msg.header.frame_id,
                        "K": list(msg.k),  # length-9 array
                        "D": list(msg.d),
                        "distortion_model": msg.distortion_model,
                        "width": int(msg.width),
                        "height": int(msg.height),
                    }
            elif mt == "PointCloud2":
                lidar_id = lidar_topic_to_id[topic]
                if lidar_id not in lidar_frame_ids:
                    lidar_frame_ids[lidar_id] = msg.header.frame_id
            elif mt == "TFMessage":
                saw_tf_static = True
                for ts_msg in msg.transforms:
                    key = (ts_msg.header.frame_id, ts_msg.child_frame_id)
                    if key not in static_transforms:
                        static_transforms[key] = _transform_se3(ts_msg.transform)

            # Early exit once we have everything; tf_static is typically
            # published in one batch at bag start.
            if (
                saw_tf_static
                and set(camera_infos.keys()) >= needed_camera_ids
                and set(lidar_frame_ids.keys()) >= needed_lidar_ids
            ):
                break

    calib = build_calibration_dict(
        camera_infos=camera_infos,
        lidar_frame_ids=lidar_frame_ids,
        static_transforms=static_transforms,
        ego_frame=cfg.ego_frame,
        bag_id=bag_id,
    )

    out_uri = calibration_path(bag_id)
    ensure_local_dir(os.path.dirname(local_path(out_uri)))
    with open(local_path(out_uri), "w", encoding="utf-8") as fh:
        json.dump(calib, fh, indent=2)
    return out_uri


def _transform_se3(transform) -> np.ndarray:
    t = transform.translation
    q = transform.rotation
    return make_se3(
        np.array([t.x, t.y, t.z], dtype=np.float64),
        (q.x, q.y, q.z, q.w),
    )


# ---------------------------------------------------------------------------
# File-based override (used when the bag's intrinsics/extrinsics are missing
# or wrong and you want to substitute an authored calibration).
# ---------------------------------------------------------------------------


def freeze_from_file(
    bag_id: str, source_path: str, *, version: str | None = None
) -> str:
    """Copy an authored calibration JSON to the bag's artifact root."""
    with open(source_path, "r", encoding="utf-8") as fh:
        calib = json.load(fh)

    if "cameras" not in calib or "lidars" not in calib:
        raise ValueError(
            f"calibration {source_path} missing required keys 'cameras'/'lidars'"
        )
    if version:
        calib["calibration_version"] = version
    calib.setdefault("checks", {"sanity": "skipped", "notes": "imported from file"})

    out_uri = calibration_path(bag_id)
    ensure_local_dir(os.path.dirname(local_path(out_uri)))
    with open(local_path(out_uri), "w", encoding="utf-8") as fh:
        json.dump(calib, fh, indent=2)
    return out_uri


def load(bag_id: str) -> dict:
    with open(local_path(calibration_path(bag_id)), "r", encoding="utf-8") as fh:
        return json.load(fh)


def sanity_check(bag_id: str) -> tuple[bool, str]:
    """Structural check.  Numeric (LiDAR-to-image projection) check is TODO."""
    try:
        calib = load(bag_id)
    except FileNotFoundError:
        return False, "calibration.json missing"
    if "cameras" not in calib:
        return False, "no cameras in calibration"
    return True, "structural check only; numeric sanity not yet implemented"
