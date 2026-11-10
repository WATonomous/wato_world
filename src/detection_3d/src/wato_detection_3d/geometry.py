"""The model's input frame, and its boxes in the pipeline's convention.

**Input frame.** BEVFusion was trained on nuScenes' LIDAR_TOP frame: x right,
y forward, z up, its origin 1.84 m above the ground (nuScenes calibration).
`nuscenes_lidar` builds that frame on any rig: the canonical LiDAR's ego x/y,
the training height, ego-aligned axes turned to x right / y forward. `sensor`
uses the LiDAR's own frame, as the car does (x forward on WATO: the scene
the model sees is turned 90° from training).

**Boxes.** The vendor decoder (CUDA-BEVFusion head-transbbox.cu) returns
position (x, y, z), size (w, l, h) and rot, in mmdetection3d v0.x's LiDAR box
convention, which BEVFusion's nuScenes converter writes: w along x and l along
y at rot 0, and nuScenes yaw = -rot - pi/2 (mmdetection3d's v1.0
coordinate-refactor note). So the front points along -rot - pi/2, and l is the
extent along it. z is the box's *centre*: the decoder subtracts h/2 from the
head's height to get a bottom, but MIT BEVFusion's nuScenes loader builds its
targets reading each annotation's centre as the bottom (origin=(0.5, 0.5, 0),
"haotian: this is an important change"), so what comes out is the centre.
Measured on nuScenes scene-0061: taken as the bottom, boxes sat a median
0.77 m (h/2) above the ground truth. Velocity (vx, vy) is the object's own,
expressed in the input frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ego_R_model for nuscenes_lidar: model x = ego right (-y), model y = ego
# forward (+x), model z = ego up.
_EGO_R_NUSCENES_LIDAR = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def ego_T_model(kind: str, ego_T_lidar: np.ndarray, height_m: float) -> np.ndarray:
    """(4, 4) the model's input frame in the ego frame (module docstring)."""
    if kind == "sensor":
        return np.asarray(ego_T_lidar, dtype=np.float64).copy()
    if kind != "nuscenes_lidar":
        raise ValueError(f"unknown input frame {kind!r}")
    m = np.eye(4)
    m[:3, :3] = _EGO_R_NUSCENES_LIDAR
    m[:2, 3] = np.asarray(ego_T_lidar)[:2, 3]
    m[2, 3] = height_m
    return m


def transform_points(T: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """(N, 3) points through a (4, 4) transform."""
    return xyz @ T[:3, :3].T + T[:3, 3]


@dataclass(frozen=True)
class WorldBox:
    cx: float
    cy: float
    cz: float
    w: float
    l: float  # noqa: E741 — domain term: length along heading
    h: float
    heading: float  # directed, (-pi, pi]
    vx: float
    vy: float


def heading_in_model(rot: float) -> float:
    """The direction the box faces in the model frame, from the vendor rot."""
    return -rot - math.pi / 2.0


def rot_from_heading(heading: float) -> float:
    """Inverse of heading_in_model."""
    return -heading - math.pi / 2.0


def _wrap(theta: float) -> float:
    t = math.remainder(theta, 2.0 * math.pi)
    return math.pi if t == -math.pi else t


def to_world(
    world_T_model: np.ndarray,
    x: float,
    y: float,
    z: float,
    w: float,
    l: float,  # noqa: E741
    h: float,
    rot: float,
    vx: float,
    vy: float,
) -> WorldBox:
    """One vendor box (model frame) as a world box: centre, l along the
    directed heading. The heading is the world-frame yaw of the box's front
    direction; on a tilted input frame its vertical component is dropped."""
    R = world_T_model[:3, :3]
    centre = world_T_model @ np.array([x, y, z, 1.0])
    theta = heading_in_model(rot)
    front = R @ np.array([math.cos(theta), math.sin(theta), 0.0])
    v = R @ np.array([vx, vy, 0.0])
    return WorldBox(
        cx=float(centre[0]),
        cy=float(centre[1]),
        cz=float(centre[2]),
        w=float(w),
        l=float(l),
        h=float(h),
        heading=_wrap(math.atan2(front[1], front[0])),
        vx=float(v[0]),
        vy=float(v[1]),
    )


def from_world(world_T_model: np.ndarray, box: WorldBox) -> tuple[float, ...]:
    """Inverse of to_world on a level frame: (x, y, z, w, l, h, rot, vx, vy). For tests and for checking a conversion against labels."""
    model_T_world = np.linalg.inv(world_T_model)
    R = model_T_world[:3, :3]
    c = model_T_world @ np.array([box.cx, box.cy, box.cz, 1.0])
    front = R @ np.array([math.cos(box.heading), math.sin(box.heading), 0.0])
    v = R @ np.array([box.vx, box.vy, 0.0])
    rot = rot_from_heading(math.atan2(front[1], front[0]))
    return (
        float(c[0]),
        float(c[1]),
        float(c[2]),
        box.w,
        box.l,
        box.h,
        _wrap(rot),
        float(v[0]),
        float(v[1]),
    )
