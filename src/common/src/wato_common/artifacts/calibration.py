"""Per-bag sensor calibration: raw/<bag_id>/calibration.json.

Written by ingest (from the bag's CameraInfo + TF, or from an authored file
passed with --calibration); read by perception_2d, semantic_lifting and
lidar_preprocessing.

On disk (JSON):

  {
    "calibration_version": str,
    "ego_frame": str,                     # parent frame of every ego_T_*
    "cameras": {
      "<cam_id>": {
        "frame_id": str,                  # CameraInfo.header.frame_id
        "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],   # of the ingested images
        "distortion": [...],              # of the ingested images
        "distortion_model": str,
        "width": int, "height": int,
        "ego_T_cam": [[...]] | null,      # 4x4 row-major; null = TF unresolved
        "rectified": bool,                # images are rectified: K = P[:3, :3],
                                          # distortion = []
        "camera_info": {"K": [[...]], "D": [...], "P": [[...]]}   # as recorded
      }
    },
    "lidars": {
      "<lidar_id>": {"frame_id": str, "ego_T_lidar": [[...]] | null}
    },
    "static_transforms": {"<parent>__<child>": [[...]]},   # TF dump, debugging
    "checks": {"sanity": str, "notes": str}
  }

`K` and `distortion` always describe the images ingest wrote, so projecting
into them needs nothing else: for a rectified image topic ingest takes them
from CameraInfo.P (CameraInfo.K / D describe the raw sensor image, which was
never recorded), and keeps the raw CameraInfo under `camera_info`.

Reading requires `cameras` and `lidars`, each camera's `K` and `ego_T_cam`
key, and each LiDAR's `ego_T_lidar` key. The other fields are metadata no
projection uses; an authored file may leave them out and they read as empty.
An extrinsic is null when ingest couldn't resolve the TF chain; ingest's
require_complete refuses such a bag, and the accessors below raise rather
than hand a consumer None.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from wato_common.artifact_store import calibration_path, ensure_local_dir, local_path
from wato_common.artifacts import ArtifactFormatError


@dataclass(frozen=True)
class CameraCalibration:
    cam_id: str
    K: np.ndarray  # (3, 3) float64
    ego_T_cam: Optional[np.ndarray]  # (4, 4) float64; None = TF unresolved
    frame_id: str = ""
    distortion: tuple[float, ...] = ()
    distortion_model: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    rectified: bool = False
    # CameraInfo as recorded (None for an authored file): K (3, 3), D, P (3, 4).
    camera_info_K: Optional[np.ndarray] = None
    camera_info_D: tuple[float, ...] = ()
    camera_info_P: Optional[np.ndarray] = None


@dataclass(frozen=True)
class LidarCalibration:
    lidar_id: str
    ego_T_lidar: Optional[np.ndarray]  # (4, 4) float64; None = TF unresolved
    frame_id: str = ""


@dataclass(frozen=True)
class Calibration:
    cameras: dict[str, CameraCalibration]
    lidars: dict[str, LidarCalibration]
    version: str = ""
    ego_frame: str = ""
    # "<parent>__<child>" -> (4, 4); every TF link the bag carried.
    static_transforms: dict[str, np.ndarray] = field(default_factory=dict)
    checks: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, bag_id: str) -> "Calibration":
        path = local_path(calibration_path(bag_id))
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh), source=path)

    def camera(self, cam_id: str) -> CameraCalibration:
        """The camera's calibration, with a resolved ego_T_cam."""
        cam = self.cameras.get(cam_id)
        if cam is None:
            raise KeyError(f"camera {cam_id!r} not found in calibration.json")
        if cam.ego_T_cam is None:
            raise ValueError(
                f"camera {cam_id!r} has null ego_T_cam in calibration.json "
                "(extrinsic unresolved — check /tf_static coverage)"
            )
        return cam

    def ego_T_lidar(self, lidar_id: str) -> np.ndarray:
        lid = self.lidars.get(lidar_id)
        if lid is None:
            raise KeyError(f"lidar {lidar_id!r} not found in calibration.json")
        if lid.ego_T_lidar is None:
            raise ValueError(
                f"lidar {lidar_id!r} has null ego_T_lidar in calibration.json "
                "(extrinsic unresolved — check /tf_static coverage)"
            )
        return lid.ego_T_lidar

    @classmethod
    def from_dict(cls, d: dict, *, source: str = "<dict>") -> "Calibration":
        """Parse calibration.json content; raises ArtifactFormatError."""

        def need(obj: dict, key: str, where: str):
            if not isinstance(obj, dict) or key not in obj:
                raise ArtifactFormatError(source, f"{where} has no {key!r}")
            return obj[key]

        def matrix(value, shape: tuple[int, int], where: str) -> Optional[np.ndarray]:
            if value is None:
                return None
            m = np.asarray(value, dtype=np.float64)
            if m.shape != shape:
                raise ArtifactFormatError(
                    source, f"{where} has shape {m.shape}, not {shape}"
                )
            return m

        cameras = {}
        for cam_id, c in need(d, "cameras", "calibration").items():
            where = f"camera {cam_id}"
            K = matrix(need(c, "K", where), (3, 3), f"{where} K")
            if K is None:
                raise ArtifactFormatError(source, f"{where} has null K")
            info = c.get("camera_info") or {}
            cameras[cam_id] = CameraCalibration(
                cam_id=cam_id,
                K=K,
                ego_T_cam=matrix(
                    need(c, "ego_T_cam", where), (4, 4), f"{where} ego_T_cam"
                ),
                frame_id=c.get("frame_id") or "",
                distortion=tuple(float(x) for x in c.get("distortion") or ()),
                distortion_model=c.get("distortion_model") or "",
                width=int(c["width"]) if c.get("width") is not None else None,
                height=int(c["height"]) if c.get("height") is not None else None,
                rectified=bool(c.get("rectified", False)),
                camera_info_K=matrix(info.get("K"), (3, 3), f"{where} camera_info K"),
                camera_info_D=tuple(float(x) for x in info.get("D") or ()),
                camera_info_P=matrix(info.get("P"), (3, 4), f"{where} camera_info P"),
            )

        lidars = {}
        for lidar_id, entry in need(d, "lidars", "calibration").items():
            where = f"lidar {lidar_id}"
            lidars[lidar_id] = LidarCalibration(
                lidar_id=lidar_id,
                ego_T_lidar=matrix(
                    need(entry, "ego_T_lidar", where), (4, 4), f"{where} ego_T_lidar"
                ),
                frame_id=entry.get("frame_id") or "",
            )

        return cls(
            cameras=cameras,
            lidars=lidars,
            version=d.get("calibration_version") or "",
            ego_frame=d.get("ego_frame") or "",
            static_transforms={
                k: np.asarray(v, dtype=np.float64)
                for k, v in (d.get("static_transforms") or {}).items()
            },
            checks=dict(d.get("checks") or {}),
        )

    def to_dict(self) -> dict:
        def listed(m: Optional[np.ndarray]):
            return m.tolist() if m is not None else None

        return {
            "calibration_version": self.version,
            "ego_frame": self.ego_frame,
            "cameras": {
                cam_id: {
                    "frame_id": c.frame_id,
                    "K": c.K.tolist(),
                    "distortion": list(c.distortion),
                    "distortion_model": c.distortion_model,
                    "width": c.width,
                    "height": c.height,
                    "ego_T_cam": listed(c.ego_T_cam),
                    "rectified": c.rectified,
                    "camera_info": {
                        "K": listed(c.camera_info_K),
                        "D": list(c.camera_info_D),
                        "P": listed(c.camera_info_P),
                    },
                }
                for cam_id, c in self.cameras.items()
            },
            "lidars": {
                lidar_id: {
                    "frame_id": lid.frame_id,
                    "ego_T_lidar": listed(lid.ego_T_lidar),
                }
                for lidar_id, lid in self.lidars.items()
            },
            "static_transforms": {
                k: v.tolist() for k, v in self.static_transforms.items()
            },
            "checks": dict(self.checks),
        }


def write(bag_id: str, calib: Calibration) -> str:
    """Freeze the bag's calibration; returns its URI."""
    uri = calibration_path(bag_id)
    ensure_local_dir(os.path.dirname(local_path(uri)))
    with open(local_path(uri), "w", encoding="utf-8") as fh:
        json.dump(calib.to_dict(), fh, indent=2)
    return uri
