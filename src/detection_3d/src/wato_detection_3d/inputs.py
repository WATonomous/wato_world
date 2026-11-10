"""One frame's model inputs: points, six images and their calibration.

Everything is expressed in the model's input frame at the frame's reference
time (geometry.ego_T_model), and every pose is looked up at its datum's own
time: each sweep's points are already deskewed into the world by
lidar_preprocessing, and each image's camera-to-LiDAR matrix composes the ego
pose at the *image's* time with the one at the reference time, as nuScenes'
own info converter (obtain_sensor2top) does for BEVFusion's training data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from wato_common.artifacts.calibration import Calibration
from wato_common.artifacts.frame_index import CameraFrame
from wato_common.artifacts.lidar_sweeps import WorldSweep
from wato_detection_3d.config import ComponentConfig, ImageConfig, PointsConfig
from wato_detection_3d.geometry import transform_points

# The model's fixed input contract (the fork's libpybev load(), the car's
# retrain-dependent parameters): six 1600x900 RGB images, resized by 0.48 and
# cropped from the bottom-centre to 704x256; points voxelized within this box.
IMAGE_WIDTH = 1600
IMAGE_HEIGHT = 900
RESIZE = 0.48
NET_WIDTH = 704
NET_HEIGHT = 256
POINT_RANGE_MIN = np.array([-54.0, -54.0, -5.0])
POINT_RANGE_MAX = np.array([54.0, 54.0, 3.0])
MAX_POINTS = 300_000


@dataclass(frozen=True)
class SweepPoints:
    """One sweep's world points, as an earlier or current input sweep."""

    sweep_id: int
    timestamp_ns: int
    sweep: WorldSweep
    # The sensor's frame at the sweep's time, from the world; None for the
    # current sweeps, whose close points are kept (LoadPointsFromMultiSweeps
    # drops them from earlier sweeps only).
    sensor_T_world: Optional[np.ndarray]


def assemble_points(
    sweeps: Sequence[SweepPoints],
    model_T_world: np.ndarray,
    reference_timestamp_ns: int,
    cfg: PointsConfig,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """(N, 5) float32 x, y, z, intensity, age in seconds, in the model frame,
    and the sweeps that contributed points.

    Points outside the model's voxel range are dropped first (it ignores them
    anyway); if more than MAX_POINTS remain, the oldest go first.
    """
    parts = []
    used = []
    for s in sweeps:
        if s.sweep.intensity is None:
            raise ValueError(f"sweep {s.sweep_id} has no intensity; the model needs it")
        xyz = s.sweep.xyz
        keep = np.ones(xyz.shape[0], dtype=bool)
        if s.sensor_T_world is not None and cfg.remove_close_m > 0:
            local = transform_points(s.sensor_T_world, xyz)
            keep &= ~(
                (np.abs(local[:, 0]) < cfg.remove_close_m)
                & (np.abs(local[:, 1]) < cfg.remove_close_m)
            )
        pts = transform_points(model_T_world, xyz[keep])
        inside = np.all((pts >= POINT_RANGE_MIN) & (pts <= POINT_RANGE_MAX), axis=1)
        pts = pts[inside]
        if pts.shape[0] == 0:
            continue
        age = max(0.0, (reference_timestamp_ns - s.timestamp_ns) * 1e-9)
        part = np.empty((pts.shape[0], 5), dtype=np.float32)
        part[:, :3] = pts
        part[:, 3] = np.asarray(s.sweep.intensity)[keep][inside]
        part[:, 4] = age
        parts.append(part)
        used.append(s.sweep_id)
    if not parts:
        return np.zeros((0, 5), dtype=np.float32), ()
    points = np.concatenate(parts)
    if points.shape[0] > MAX_POINTS:
        order = np.argsort(points[:, 4], kind="stable")
        points = points[np.sort(order[:MAX_POINTS])]
    return points, tuple(sorted(used))


def pick_sweeps(
    reference_timestamp_ns: int,
    current: dict[str, tuple[int, int]],
    history: dict[str, list[tuple[int, int]]],
    cfg: PointsConfig,
) -> list[tuple[str, int, int, bool]]:
    """(lidar_id, sweep_id, timestamp_ns, is_current) of the sweeps to feed:
    each LiDAR's sweep in the frame, then up to cfg.past_sweeps of its
    earlier sweeps, newest first, no older than cfg.max_time_lag_s.

    current: lidar_id -> (sweep_id, timestamp_ns) for the frame's sweeps.
    history: lidar_id -> every (timestamp_ns, sweep_id) of that LiDAR, sorted.
    """
    max_age_ns = int(round(cfg.max_time_lag_s * 1e9))
    out = []
    for lidar_id, (sweep_id, t) in current.items():
        out.append((lidar_id, sweep_id, t, True))
        earlier = [
            (ts, sid)
            for ts, sid in history.get(lidar_id, [])
            if ts < t and reference_timestamp_ns - ts <= max_age_ns
        ]
        for ts, sid in sorted(earlier, reverse=True)[: cfg.past_sweeps]:
            out.append((lidar_id, sid, ts, False))
    return out


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #


def img_aug_matrix() -> np.ndarray:
    """(4, 4) the vendor's resize + bottom-centre crop of a 1600x900 image
    (camera-normalization.cu; BEVFusion's test-time ImageAug3D)."""
    resized_w = int(IMAGE_WIDTH * RESIZE)
    resized_h = int(IMAGE_HEIGHT * RESIZE)
    m = np.eye(4)
    m[0, 0] = m[1, 1] = RESIZE
    m[0, 3] = -float((resized_w - NET_WIDTH) // 2)
    m[1, 3] = -float(resized_h - NET_HEIGHT)
    return m


@dataclass(frozen=True)
class Band:
    """Where a camera's image is cut and scaled to the model's 1600x900."""

    scale: float
    top: float  # first source row (fractional), after which band_h rows
    band_h: float


def band_for(width: int, height: int, cy: float, cfg: ImageConfig) -> Band:
    """Scale to 1600 wide; cut a 900-high band whose principal point lands on
    cfg.principal_row_px (as near as the image allows). A 1600x900 image is
    passed through unchanged."""
    scale = IMAGE_WIDTH / width
    band_h = IMAGE_HEIGHT / scale
    if band_h > height + 1e-6:
        raise ValueError(
            f"a {width}x{height} image is wider than 16:9; padding it to the "
            "model's 1600x900 is not supported"
        )
    # Pixel centres: row v maps to (v - top + 0.5) * scale - 0.5.
    top = cy + 0.5 - (cfg.principal_row_px + 0.5) / scale
    top = min(max(top, 0.0), height - band_h)
    return Band(scale=scale, top=top, band_h=band_h)


def apply_band(
    image: Image.Image, K: np.ndarray, band: Band
) -> tuple[np.ndarray, np.ndarray]:
    """The (900, 1600, 3) uint8 RGB model image and its 3x3 intrinsics."""
    width = image.width
    out = image.convert("RGB").resize(
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        resample=Image.BILINEAR,
        box=(0.0, band.top, float(width), band.top + band.band_h),
    )
    s = band.scale
    K2 = np.asarray(K, dtype=np.float64).copy()
    K2[0, 0] *= s
    K2[0, 1] *= s
    K2[1, 1] *= s
    K2[0, 2] = (K[0, 2] + 0.5) * s - 0.5
    K2[1, 2] = (K[1, 2] - band.top + 0.5) * s - 0.5
    return np.ascontiguousarray(np.asarray(out, dtype=np.uint8)), K2


@dataclass(frozen=True)
class CameraInputs:
    images: np.ndarray  # (6, 900, 1600, 3) uint8 RGB
    camera2lidar: np.ndarray  # (6, 4, 4) model_T_cam
    intrinsics: np.ndarray  # (6, 4, 4) K padded
    lidar2image: np.ndarray  # (6, 4, 4) K @ cam_T_model
    img_aug: np.ndarray  # (6, 4, 4)
    timestamps_ns: tuple[int, ...]


def camera_inputs(
    frames: Sequence[CameraFrame],
    calib: Calibration,
    model_T_world: np.ndarray,
    cfg: ComponentConfig,
) -> CameraInputs:
    """The six cameras' images and matrices, in cfg.cameras order. Every
    frame must carry its pose at the image's time (world_T_ego)."""
    images, c2l, intr, l2i = [], [], [], []
    aug = img_aug_matrix()
    for f in frames:
        cam = calib.camera(f.cam_id)
        if cam.distortion and any(abs(d) > 0.0 for d in cam.distortion):
            raise ValueError(
                f"camera {f.cam_id}: the model takes pinhole images; this one "
                f"declares {cam.distortion_model} distortion"
            )
        if cam.ego_T_cam is None:
            raise ValueError(f"camera {f.cam_id}: no ego_T_cam")
        if f.world_T_ego is None:
            raise ValueError(f"camera {f.cam_id} frame {f.camera_seq}: no pose")
        with Image.open(f.image_path) as im:
            im.load()
            cam.model_for_image((im.width, im.height))
            band = band_for(im.width, im.height, float(cam.K[1, 2]), cfg.image)
            rgb, K = apply_band(im, cam.K, band)
        model_T_cam = model_T_world @ f.world_T_ego @ cam.ego_T_cam
        K4 = np.eye(4)
        K4[:3, :3] = K
        images.append(rgb)
        c2l.append(model_T_cam)
        intr.append(K4)
        l2i.append(K4 @ np.linalg.inv(model_T_cam))
    stack = lambda ms: np.ascontiguousarray(np.stack(ms).astype(np.float32))  # noqa: E731
    return CameraInputs(
        images=np.ascontiguousarray(np.stack(images)),
        camera2lidar=stack(c2l),
        intrinsics=stack(intr),
        lidar2image=stack(l2i),
        img_aug=stack([aug] * len(frames)),
        timestamps_ns=tuple(f.timestamp_ns for f in frames),
    )
