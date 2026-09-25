"""Oriented 3D box helpers on wato_common.schemas.Box3D.

Boxes are yaw-only (gravity-aligned): a BEV rectangle (cx, cy, l, w, heading)
extruded over [cz - h/2, cz + h/2]. `l` is always the longer BEV side and
`heading` is the direction of `l`, wrapped to (-pi/2, pi/2] — a cluster fit
has no front/back, so the heading is only defined modulo pi.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from wato_common.schemas import Box3D

# Floor on every fitted extent [m]. A single-beam or coplanar cluster has a
# zero-width side; without a floor its volume is 0 and the volume-ratio and
# IoU terms of the association cost degenerate.
MIN_EXTENT_M = 0.1


def wrap_heading_half_pi(theta: float) -> float:
    """Wrap an undirected heading into (-pi/2, pi/2]."""
    t = (theta + math.pi / 2.0) % math.pi - math.pi / 2.0
    return math.pi / 2.0 if t == -math.pi / 2.0 else t


def fit_bev_box(xyz: np.ndarray) -> Box3D:
    """Minimum-area oriented BEV rectangle + z extent around a point cluster.

    Rotating calipers over the 2D convex hull: the minimum-area rectangle has
    one side collinear with a hull edge, so testing each edge direction is
    exact. Degenerate clusters (< 3 points or collinear) fall back to the
    axis of largest spread.
    """
    if xyz.shape[0] == 0:
        raise ValueError("fit_bev_box needs at least one point")
    xy = xyz[:, :2].astype(np.float64)
    z_lo, z_hi = float(xyz[:, 2].min()), float(xyz[:, 2].max())

    angles: np.ndarray
    try:
        hull = ConvexHull(xy)
        pts = xy[hull.vertices]
        edges = np.roll(pts, -1, axis=0) - pts
        angles = np.unique(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), math.pi / 2))
    except (QhullError, ValueError):
        # < 3 points or all collinear: orient along the principal axis.
        centred = xy - xy.mean(axis=0)
        if xy.shape[0] >= 2 and np.any(centred):
            _, _, vt = np.linalg.svd(centred, full_matrices=False)
            angles = np.array([math.atan2(vt[0, 1], vt[0, 0])])
        else:
            angles = np.array([0.0])
        pts = xy

    best = None
    for a in angles:
        c, s = math.cos(a), math.sin(a)
        # Project onto the rotated axes (u along angle a, v perpendicular).
        u = pts[:, 0] * c + pts[:, 1] * s
        v = -pts[:, 0] * s + pts[:, 1] * c
        area = (u.max() - u.min()) * (v.max() - v.min())
        if best is None or area < best[0]:
            best = (area, a, u.min(), u.max(), v.min(), v.max())
    _, a, u0, u1, v0, v1 = best
    du, dv = u1 - u0, v1 - v0
    uc, vc = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
    c, s = math.cos(a), math.sin(a)
    cx, cy = uc * c - vc * s, uc * s + vc * c
    if du >= dv:
        length, width, heading = du, dv, a
    else:
        length, width, heading = dv, du, a + math.pi / 2.0
    return Box3D(
        cx=float(cx),
        cy=float(cy),
        cz=0.5 * (z_lo + z_hi),
        w=max(float(width), MIN_EXTENT_M),
        l=max(float(length), MIN_EXTENT_M),
        h=max(z_hi - z_lo, MIN_EXTENT_M),
        heading=wrap_heading_half_pi(heading),
    )


def bev_corners(box: Box3D) -> np.ndarray:
    """(4, 2) BEV corners, counter-clockwise."""
    c, s = math.cos(box.heading), math.sin(box.heading)
    hl, hw = box.l / 2.0, box.w / 2.0
    local = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]])
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([box.cx, box.cy])


def _clip_polygon(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland–Hodgman: subject ∩ clip for convex CCW polygons."""
    out = subject
    n = clip.shape[0]
    for i in range(n):
        if out.shape[0] == 0:
            break
        a, b = clip[i], clip[(i + 1) % n]
        edge = b - a

        def inside(p: np.ndarray) -> bool:
            return edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0]) >= -1e-12

        inp = out
        res: list[np.ndarray] = []
        for j in range(inp.shape[0]):
            cur, prev = inp[j], inp[j - 1]
            cur_in, prev_in = inside(cur), inside(prev)
            if cur_in != prev_in:
                d = cur - prev
                denom = edge[0] * d[1] - edge[1] * d[0]
                if abs(denom) > 1e-15:
                    t = (
                        edge[0] * (a[1] - prev[1]) - edge[1] * (a[0] - prev[0])
                    ) / denom
                    res.append(prev + t * d)
            if cur_in:
                res.append(cur)
        out = np.array(res) if res else np.empty((0, 2))
    return out


def _polygon_area(poly: np.ndarray) -> float:
    if poly.shape[0] < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def box_volume(box: Box3D) -> float:
    return (
        max(box.l, MIN_EXTENT_M) * max(box.w, MIN_EXTENT_M) * max(box.h, MIN_EXTENT_M)
    )


def iou_3d(a: Box3D, b: Box3D) -> float:
    """Volumetric IoU of two yaw-only boxes (BEV polygon ∩ × z overlap)."""
    # Cheap reject: centres farther apart than the sum of half-diagonals.
    ra = 0.5 * math.hypot(a.l, a.w)
    rb = 0.5 * math.hypot(b.l, b.w)
    if math.hypot(a.cx - b.cx, a.cy - b.cy) > ra + rb:
        return 0.0
    z_overlap = min(a.cz + a.h / 2, b.cz + b.h / 2) - max(
        a.cz - a.h / 2, b.cz - b.h / 2
    )
    if z_overlap <= 0.0:
        return 0.0
    inter_bev = _polygon_area(_clip_polygon(bev_corners(a), bev_corners(b)))
    inter = inter_bev * z_overlap
    union = box_volume(a) + box_volume(b) - inter
    return float(inter / union) if union > 0 else 0.0


def points_in_box(xyz: np.ndarray, box: Box3D, margin_m: float = 0.0) -> np.ndarray:
    """Boolean mask of points inside `box` grown by `margin_m` on every side."""
    c, s = math.cos(box.heading), math.sin(box.heading)
    dx = xyz[:, 0] - box.cx
    dy = xyz[:, 1] - box.cy
    u = dx * c + dy * s
    v = -dx * s + dy * c
    return (
        (np.abs(u) <= box.l / 2.0 + margin_m)
        & (np.abs(v) <= box.w / 2.0 + margin_m)
        & (np.abs(xyz[:, 2] - box.cz) <= box.h / 2.0 + margin_m)
    )
