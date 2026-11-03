"""Refining one track's boxes, training-free.

**Parked** (3D Auto Labeling's static path): every frame sees the car from a
different side, and it doesn't move, so the points of all its frames inside
the rectangle enclosing its boxes are one cloud. Its largest dense group is the
car; one box is fit to it and lowered to the ground, and every frame gets it.

**Moving** (and unknown): one size for the whole track, a high percentile over
its densest frames (far frames see one face, so their boxes are small; LISO,
OYSTER). Each frame's box is placed on the observed box's corner nearest the
ego and extends away from it (OYSTER, CPD): the near face is where the points
are. An axis the observed box already spans at full size keeps its observed
centre. Heading comes from the smoothed velocity where the track moves fast
enough (VESPA), which also gives the box its front; centres are smoothed by a
local linear fit (Offline-Poly's sliding window). Moving points are not
stacked: 3DAL found aligning a mover's points by its noisy boxes hurt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from wato_common.artifacts.ground import GroundGrid
from wato_common.geometry.objects import largest_group, to_ground
from wato_common.schemas import Box3D
from wato_common.tracking import bev_corners, fit_bev_box, wrap_heading_half_pi
from wato_label_refinement.config import MovingConfig, ParkedConfig


@dataclass(frozen=True)
class Refined:
    boxes: list[Box3D]  # one per input frame, in input order
    heading_directed: list[bool]
    residual_smoothness: Optional[float]


def enclosing_region(
    boxes: list[Box3D], margin_m: float, max_centre_offset_m: float
) -> Box3D:
    """The rectangle (and z span) enclosing the boxes centred within
    max_centre_offset_m of their median centre, grown by margin_m."""
    xy = np.array([[b.cx, b.cy] for b in boxes])
    near = np.linalg.norm(xy - np.median(xy, axis=0), axis=1) <= max_centre_offset_m
    boxes = [b for b, keep in zip(boxes, near) if keep] or boxes
    corners = np.vstack([bev_corners(b) for b in boxes])
    lo = min(b.cz - b.h / 2 for b in boxes)
    hi = max(b.cz + b.h / 2 for b in boxes)
    rect = fit_bev_box(np.column_stack([corners, np.full(len(corners), lo)]))
    return Box3D(
        cx=rect.cx,
        cy=rect.cy,
        cz=(lo + hi) / 2,
        w=rect.w + 2 * margin_m,
        l=rect.l + 2 * margin_m,
        h=(hi - lo) + 2 * margin_m,
        heading=rect.heading,
    )


def refine_parked(
    points: np.ndarray, ground: Optional[GroundGrid], cfg: ParkedConfig
) -> Optional[Box3D]:
    """One box for a parked track from its accumulated points; None when too
    few points for it."""
    if points.shape[0] < cfg.min_points:
        return None
    q = np.floor(points / cfg.voxel_m).astype(np.int64)
    _, first = np.unique(q, axis=0, return_index=True)
    p = points[np.sort(first)]
    group = p[largest_group(p, cfg.cluster_radius_m)]
    if group.shape[0] < cfg.min_points:
        return None
    return to_ground(fit_bev_box(group), ground)


def local_linear(
    t_s: np.ndarray, xy: np.ndarray, window_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """(fitted centres, velocities) at each time, by a least-squares line over
    the samples within ±window_s / 2. A sample with no neighbour keeps itself
    and gets zero velocity."""
    fitted = xy.copy()
    vel = np.zeros_like(xy)
    for i, t in enumerate(t_s):
        near = np.abs(t_s - t) <= window_s / 2
        dt = t_s[near] - t
        if near.sum() < 2 or np.ptp(dt) <= 0:
            continue
        a = np.column_stack([np.ones(dt.size), dt])
        coef, *_ = np.linalg.lstsq(a, xy[near], rcond=None)
        fitted[i], vel[i] = coef[0], coef[1]
    return fitted, vel


def _undirected_mean(headings: np.ndarray) -> float:
    """Mean of headings defined mod pi (averaged as doubled angles)."""
    return wrap_heading_half_pi(
        0.5 * math.atan2(np.sin(2 * headings).mean(), np.cos(2 * headings).mean())
    )


def anchored_centre(
    observed: Box3D, length: float, width: float, heading: float, ego_xy
) -> np.ndarray:
    """Where a length x width box at `heading` sits when its corner nearest the
    ego is the observed box's. Along an axis the observed box spans at least
    the refined size, the observed centre is kept."""
    c, s = math.cos(heading), math.sin(heading)
    u_axis, v_axis = np.array([c, s]), np.array([-s, c])
    centre = np.array([observed.cx, observed.cy])
    rel = bev_corners(observed) - centre
    u, v = rel @ u_axis, rel @ v_axis
    ego = np.asarray(ego_xy, float) - centre
    out = []
    for ext, size, e in ((u, length, ego @ u_axis), (v, width, ego @ v_axis)):
        lo, hi = float(ext.min()), float(ext.max())
        if hi - lo >= size:
            out.append((lo + hi) / 2)
        elif e < (lo + hi) / 2:
            out.append(lo + size / 2)
        else:
            out.append(hi - size / 2)
    return centre + out[0] * u_axis + out[1] * v_axis


def refine_moving(
    t_ns: np.ndarray,
    boxes: list[Box3D],
    n_points: np.ndarray,
    ego_xy: np.ndarray,
    ground: Optional[GroundGrid],
    cfg: MovingConfig,
    cls: Optional[str] = None,
) -> Refined:
    """Boxes for a moving (or unknown) track, one per frame in input order.

    ego_xy: (N, 2) ego position at each frame's time; cls picks the size
    percentile.
    """
    n = len(boxes)
    order = np.argsort(t_ns, kind="stable")
    k = max(1, math.ceil(cfg.densest_fraction * n))
    dense = np.argsort(n_points, kind="stable")[-k:]
    ls = np.array([boxes[i].l for i in dense])
    ws = np.array([boxes[i].w for i in dense])
    hs = np.array([boxes[i].h for i in dense])
    q = cfg.size_percentile_by_class.get(cls, cfg.size_percentile)
    length = float(np.percentile(ls, q))
    width = float(np.percentile(ws, q))
    height = float(np.percentile(hs, q))
    length, width = max(length, width), min(length, width)
    still_heading = _undirected_mean(np.array([boxes[i].heading for i in dense]))

    t = (t_ns[order] - t_ns[order][0]) * 1e-9
    measured = np.array([[boxes[i].cx, boxes[i].cy] for i in order])
    _, vel = local_linear(t, measured, cfg.smooth_window_s)
    speed = np.linalg.norm(vel, axis=1)
    directed = speed >= cfg.heading_min_speed_mps
    heading = np.where(directed, np.arctan2(vel[:, 1], vel[:, 0]), still_heading)

    anchored = np.array(
        [
            anchored_centre(boxes[i], length, width, float(heading[j]), ego_xy[i])
            for j, i in enumerate(order)
        ]
    )
    smooth, _ = local_linear(t, anchored, cfg.smooth_window_s)
    residual = (
        float(np.sqrt(np.mean(np.sum((smooth - anchored) ** 2, axis=1))))
        if n > 1
        else None
    )

    bottoms = np.array([boxes[i].cz - boxes[i].h / 2 for i in order])
    out: list[Optional[Box3D]] = [None] * n
    flags = [False] * n
    for j, i in enumerate(order):
        cx, cy = smooth[j]
        bottom = (
            float(ground.height_at(np.array([[cx, cy]]))[0])
            if ground is not None
            else float(np.median(bottoms))
        )
        out[i] = Box3D(
            cx=float(cx),
            cy=float(cy),
            cz=bottom + height / 2,
            w=width,
            l=length,
            h=height,
            heading=float(heading[j])
            if directed[j]
            else wrap_heading_half_pi(float(heading[j])),
        )
        flags[i] = bool(directed[j])
    return Refined(boxes=out, heading_directed=flags, residual_smoothness=residual)
