"""Freeze a per-bag calibration JSON.

Two paths:

1. `freeze_from_bag` (default) — each camera's CameraInfo topic, the TF topic,
   and the first PointCloud2 per LiDAR.  A camera configured `rectified: true`
   gets CameraInfo.P[:3, :3] and no distortion (K / D describe the raw image).
2. `freeze_from_file` — copy an authored calibration JSON in, for a bag whose
   CameraInfo or TF is missing or wrong.

Either way `require_complete` then raises CalibrationError unless every
configured sensor has intrinsics and an extrinsic.  The output,
`raw/<bag_id>/calibration.json`, is written through
wato_common.artifacts.calibration.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from wato_common.artifacts import calibration as calibration_artifact
from wato_common.artifacts.calibration import (
    Calibration,
    CameraCalibration,
    LidarCalibration,
)
from wato_common.geometry import make_se3
from wato_common.io.rosbag_reader import messages

from wato_ingest.config import IngestConfig


class CalibrationError(RuntimeError):
    """A configured sensor has no usable intrinsics or extrinsics."""


# ---------------------------------------------------------------------------
# Pure-logic builder — exposed for unit tests.
# ---------------------------------------------------------------------------


def build_calibration(
    *,
    camera_infos: dict[str, dict],
    lidar_frame_ids: dict[str, str],
    static_transforms: dict[tuple[str, str], np.ndarray],
    ego_frame: str,
    bag_id: str,
    rectified_cameras: frozenset[str] = frozenset(),
) -> Calibration:
    """Assemble a bag's calibration from already-decoded inputs.

    `camera_infos` is `{cam_id: {"frame_id", "K", "D", "distortion_model",
                                  "width", "height", "P", "R"}}` ("P"/"R"
    optional, needed only for a rectified camera).
    `lidar_frame_ids` is `{lidar_id: frame_id}`.
    `static_transforms` is `{(parent, child): SE3 4x4}`.
    `rectified_cameras` are the cameras whose image topic is rectified: their
    images are described by CameraInfo.P[:3, :3] with no distortion.
    """
    notes: list[str] = []

    cameras: dict[str, CameraCalibration] = {}
    for cam_id, info in camera_infos.items():
        frame_id = info["frame_id"]
        ego_T_cam = _resolve_chain(static_transforms, ego_frame, frame_id)
        if ego_T_cam is None:
            notes.append(
                f"camera {cam_id}: no /tf_static path {ego_frame} -> {frame_id}"
            )
        K_raw = np.asarray(info["K"], dtype=np.float64).reshape(3, 3)
        D_raw = tuple(float(x) for x in info["D"])
        P = (
            np.asarray(info["P"], dtype=np.float64).reshape(3, 4)
            if info.get("P") is not None
            else None
        )
        rectified = cam_id in rectified_cameras
        if rectified:
            K, distortion = _rectified_intrinsics(cam_id, P, info.get("R")), ()
        else:
            K, distortion = K_raw, D_raw
            if not D_raw:
                notes.append(
                    f"camera {cam_id}: empty distortion — bag camera_info has no D"
                    " (pre-undistorted images or missing calibration)"
                )
        cameras[cam_id] = CameraCalibration(
            cam_id=cam_id,
            K=K,
            ego_T_cam=ego_T_cam,
            frame_id=frame_id,
            distortion=distortion,
            distortion_model=info["distortion_model"],
            width=int(info["width"]),
            height=int(info["height"]),
            rectified=rectified,
            camera_info_K=K_raw,
            camera_info_D=D_raw,
            camera_info_P=P,
        )

    lidars: dict[str, LidarCalibration] = {}
    for lidar_id, frame_id in lidar_frame_ids.items():
        ego_T_lidar = _resolve_chain(static_transforms, ego_frame, frame_id)
        if ego_T_lidar is None:
            notes.append(
                f"lidar {lidar_id}: no /tf_static path {ego_frame} -> {frame_id}"
            )
        lidars[lidar_id] = LidarCalibration(
            lidar_id=lidar_id, ego_T_lidar=ego_T_lidar, frame_id=frame_id
        )

    return Calibration(
        cameras=cameras,
        lidars=lidars,
        version=f"auto_from_{bag_id}",
        ego_frame=ego_frame,
        static_transforms={
            f"{parent}__{child}": T for (parent, child), T in static_transforms.items()
        },
        checks={
            "sanity": "warn" if notes else "auto",
            "notes": "; ".join(notes)
            if notes
            else "all extrinsics resolved from /tf_static",
        },
    )


def _rectified_intrinsics(cam_id: str, P, R) -> np.ndarray:
    """Intrinsics of a rectified image: CameraInfo.P[:3, :3].

    Only a monocular rectification is representable as (K, ego_T_cam): a
    non-identity R rotates the rectified frame away from the camera's TF frame,
    and a non-zero P[:, 3] is a stereo baseline. Both are refused rather than
    projected wrongly.
    """
    if P is None or not np.any(P[:, :3]):
        raise CalibrationError(
            f"camera {cam_id}: rectified image but CameraInfo.P is unset; set "
            "rectified: false if the image topic is raw, or pass --calibration"
        )
    if np.any(np.abs(P[:, 3]) > 1e-9):
        raise CalibrationError(
            f"camera {cam_id}: CameraInfo.P has a stereo baseline "
            f"(P[:, 3] = {P[:, 3].tolist()}); only monocular rectification is supported"
        )
    if R is not None and not np.allclose(
        np.asarray(R, dtype=np.float64).reshape(3, 3), np.eye(3)
    ):
        raise CalibrationError(
            f"camera {cam_id}: CameraInfo.R is not identity; the rectified frame "
            "differs from the camera's TF frame, which ingest can't represent"
        )
    return P[:, :3].copy()


def _resolve_chain(
    transforms: dict[tuple[str, str], np.ndarray],
    parent: str,
    child: str,
    max_hops: int = 8,
) -> np.ndarray | None:
    """BFS for parent -> ... -> child through /tf_static, following both
    forward edges (p→c = Tpc) and inverse edges (c→p = Tpc^-1).

    ROS bags don't guarantee all transforms are published in the ego→sensor
    direction; some rigs publish sensor→ego and this handles both.
    """
    if (parent, child) in transforms:
        return transforms[(parent, child)]
    visited = {parent}
    frontier: list[tuple[str, np.ndarray]] = [(parent, np.eye(4))]
    for _ in range(max_hops):
        nxt: list[tuple[str, np.ndarray]] = []
        for cur, T in frontier:
            for (p, c), Tpc in transforms.items():
                # Forward edge: p→c
                if p == cur and c not in visited:
                    Tnext = T @ Tpc
                    if c == child:
                        return Tnext
                    visited.add(c)
                    nxt.append((c, Tnext))
                # Inverse edge: c→p (use T^-1 since Tpc is the forward transform)
                elif c == cur and p not in visited:
                    Tnext = T @ np.linalg.inv(Tpc)
                    if p == child:
                        return Tnext
                    visited.add(p)
                    nxt.append((p, Tnext))
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
    unset = [
        f"topics.cameras.{cam_id}.info"
        for cam_id, c in cfg.topics.cameras.items()
        if c.info is None
    ]
    if cfg.topics.tf_static is None:
        unset.append("topics.tf_static")
    if unset:
        raise CalibrationError(
            f"calibration from the bag needs {unset} in the ingest config; "
            "set them, or pass --calibration <file> for a bag without them"
        )
    info_topic_to_cam = {c.info: cam_id for cam_id, c in cfg.topics.cameras.items()}
    lidar_topic_to_id = {
        topic: lidar_id for lidar_id, topic in cfg.topics.lidars.items()
    }
    needed_camera_ids = set(info_topic_to_cam.values())
    needed_lidar_ids = set(lidar_topic_to_id.values())

    camera_infos: dict[str, dict] = {}
    lidar_frame_ids: dict[str, str] = {}
    static_transforms: dict[tuple[str, str], np.ndarray] = {}

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
                        "P": list(msg.p),
                        "R": list(msg.r),
                    }
            elif mt == "PointCloud2":
                lidar_id = lidar_topic_to_id[topic]
                if lidar_id not in lidar_frame_ids:
                    lidar_frame_ids[lidar_id] = msg.header.frame_id
            elif mt == "TFMessage":
                for ts_msg in msg.transforms:
                    key = (ts_msg.header.frame_id, ts_msg.child_frame_id)
                    if key not in static_transforms:
                        static_transforms[key] = _transform_se3(ts_msg.transform)

            # Stop once every sensor's info is in AND every ego→sensor chain
            # resolves (TF links can arrive spread over several messages).
            if (
                set(camera_infos.keys()) >= needed_camera_ids
                and set(lidar_frame_ids.keys()) >= needed_lidar_ids
            ):
                needed_frames = {
                    info["frame_id"] for info in camera_infos.values()
                } | set(lidar_frame_ids.values())
                if all(
                    _resolve_chain(static_transforms, cfg.ego_frame, fid) is not None
                    for fid in needed_frames
                ):
                    break

    calib = build_calibration(
        camera_infos=camera_infos,
        lidar_frame_ids=lidar_frame_ids,
        static_transforms=static_transforms,
        ego_frame=cfg.ego_frame,
        bag_id=bag_id,
        rectified_cameras=frozenset(
            cam_id for cam_id, c in cfg.topics.cameras.items() if c.rectified
        ),
    )
    return calibration_artifact.write(bag_id, calib)


def _transform_se3(transform) -> np.ndarray:
    t = transform.translation
    q = transform.rotation
    return make_se3(
        np.array([t.x, t.y, t.z], dtype=np.float64),
        (q.x, q.y, q.z, q.w),
    )


# ---------------------------------------------------------------------------
# File-based override.
# ---------------------------------------------------------------------------


def freeze_from_file(
    bag_id: str, source_path: str, *, version: str | None = None
) -> str:
    """Copy an authored calibration JSON to the bag's artifact root.

    The file is parsed with the same reader every component uses, so a file
    missing `cameras`, `lidars`, a camera's `K`/`ego_T_cam` or a LiDAR's
    `ego_T_lidar` is rejected here, naming the field.
    """
    with open(source_path, "r", encoding="utf-8") as fh:
        calib = Calibration.from_dict(json.load(fh), source=source_path)
    if version:
        calib = replace(calib, version=version)
    if not calib.checks:
        calib = replace(
            calib, checks={"sanity": "skipped", "notes": "imported from file"}
        )
    return calibration_artifact.write(bag_id, calib)


def require_complete(calib: Calibration, cfg: IngestConfig) -> None:
    """Raise CalibrationError unless every configured sensor is calibrated.

    Each camera needs ego_T_cam (K is required to read the file at all); each
    LiDAR needs ego_T_lidar.  Without them nothing downstream can place that
    sensor's data in the ego or world frame.
    """
    problems: list[str] = []
    for cam_id in cfg.topics.cameras:
        c = calib.cameras.get(cam_id)
        if c is None:
            problems.append(f"camera {cam_id}: no entry (no CameraInfo received?)")
        elif c.ego_T_cam is None:
            problems.append(
                f"camera {cam_id}: no transform {calib.ego_frame} -> {c.frame_id}"
            )
    for lidar_id in cfg.topics.lidars:
        lid = calib.lidars.get(lidar_id)
        if lid is None:
            problems.append(f"lidar {lidar_id}: no entry (no PointCloud2 received?)")
        elif lid.ego_T_lidar is None:
            problems.append(
                f"lidar {lidar_id}: no transform {calib.ego_frame} -> {lid.frame_id}"
            )
    if problems:
        raise CalibrationError(
            "calibration is incomplete: "
            + "; ".join(problems)
            + ". Check ego_frame and the TF topic in the ingest config, or pass "
            "--calibration <file>."
        )
