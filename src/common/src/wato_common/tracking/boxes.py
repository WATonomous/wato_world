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


def box_corners_3d(box: Box3D) -> np.ndarray:
    """(8, 3) corners: the bottom face's 4 (bev_corners order), then the top's."""
    bev = bev_corners(box)
    z0, z1 = box.cz - box.h / 2.0, box.cz + box.h / 2.0
    return np.vstack(
        [np.column_stack([bev, np.full(4, z0)]), np.column_stack([bev, np.full(4, z1)])]
    )


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


def bev_inside_fraction(a: Box3D, b: Box3D) -> float:
    """Share of a's BEV footprint that lies inside b's (0 to 1; not symmetric)."""
    if math.hypot(a.cx - b.cx, a.cy - b.cy) > 0.5 * (
        math.hypot(a.l, a.w) + math.hypot(b.l, b.w)
    ):
        return 0.0
    inter = _polygon_area(_clip_polygon(bev_corners(a), bev_corners(b)))
    return inter / max(max(a.l, MIN_EXTENT_M) * max(a.w, MIN_EXTENT_M), 1e-12)


def box_volume(box: Box3D) -> float:
    return (
        max(box.l, MIN_EXTENT_M) * max(box.w, MIN_EXTENT_M) * max(box.h, MIN_EXTENT_M)
    )


def iou_bev(a: Box3D, b: Box3D) -> float:
    """IoU of two boxes' BEV footprints."""
    if math.hypot(a.cx - b.cx, a.cy - b.cy) > 0.5 * (
        math.hypot(a.l, a.w) + math.hypot(b.l, b.w)
    ):
        return 0.0
    inter = _polygon_area(_clip_polygon(bev_corners(a), bev_corners(b)))
    union = a.l * a.w + b.l * b.w - inter
    return float(inter / union) if union > 0 else 0.0


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


def _convex_hull(pts: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain: the hull of (n, 2) points, counter-clockwise."""
    p = sorted(map(tuple, pts))
    if len(p) < 3:
        return np.array(p)

    def half(seq):
        out: list[tuple[float, float]] = []
        for q in seq:
            while (
                len(out) >= 2
                and (
                    (out[-1][0] - out[-2][0]) * (q[1] - out[-2][1])
                    - (out[-1][1] - out[-2][1]) * (q[0] - out[-2][0])
                )
                <= 0
            ):
                out.pop()
            out.append(q)
        return out

    lower, upper = half(p), half(reversed(p))
    return np.array(lower[:-1] + upper[:-1])


def giou_3d(a: Box3D, b: Box3D) -> float:
    """Generalized IoU (Rezatofighi et al., CVPR 2019) of two yaw-only boxes,
    in [-1, 1]: IoU − (C − U) / C, where C, as in SimpleTrack's 3D GIoU, is
    the convex hull of the two BEV footprints extruded over both boxes' joint
    z span. Unlike IoU it keeps falling as disjoint boxes move apart."""
    ca, cb = bev_corners(a), bev_corners(b)
    a_lo, a_hi = a.cz - a.h / 2.0, a.cz + a.h / 2.0
    b_lo, b_hi = b.cz - b.h / 2.0, b.cz + b.h / 2.0
    inter = _polygon_area(_clip_polygon(ca, cb)) * max(
        0.0, min(a_hi, b_hi) - max(a_lo, b_lo)
    )
    union = box_volume(a) + box_volume(b) - inter
    enclose = _polygon_area(_convex_hull(np.vstack([ca, cb]))) * (
        max(a_hi, b_hi) - min(a_lo, b_lo)
    )
    if union <= 0.0 or enclose <= 0.0:
        return 0.0
    return float(inter / union - (enclose - union) / enclose)


def _chord(l: float, w: float, angle: np.ndarray) -> np.ndarray:  # noqa: E741
    """Length of an l × w rectangle's chord through its centre at `angle`
    from its long axis."""
    c, s = np.abs(np.cos(angle)), np.abs(np.sin(angle))
    with np.errstate(divide="ignore"):
        return np.minimum(
            np.where(c > 0, l / np.maximum(c, 1e-300), np.inf),
            np.where(s > 0, w / np.maximum(s, 1e-300), np.inf),
        )


def giou_3d_cannot_exceed(
    dets: list[Box3D], preds: list[Box3D], g: float
) -> np.ndarray:
    """(n_dets, n_preds) bool: pairs whose giou_3d is provably below g.

    Exact pruning, no approximation: a pair whose BEV footprints are disjoint
    has IoU 0 and U = V_a + V_b, and its hull holds the trapezoid spanned by
    each box's chord through its centre perpendicular to the centre line, of
    area d · (W_a + W_b) / 2, extruded over at least max(h_a, h_b). So
    GIoU ≤ (V_a + V_b) / (d · (W_a + W_b) / 2 · max(h_a, h_b)) − 1.
    """
    out = np.zeros((len(dets), len(preds)), dtype=bool)
    if not dets or not preds:
        return out
    D = np.array([[x.cx, x.cy, x.l, x.w, x.h, x.heading, box_volume(x)] for x in dets])
    P = np.array([[x.cx, x.cy, x.l, x.w, x.h, x.heading, box_volume(x)] for x in preds])
    dx = P[None, :, 0] - D[:, None, 0]
    dy = P[None, :, 1] - D[:, None, 1]
    d = np.hypot(dx, dy)
    r = (
        0.5 * np.hypot(D[:, 2], D[:, 3])[:, None]
        + 0.5 * np.hypot(P[:, 2], P[:, 3])[None, :]
    )
    perp = np.arctan2(dy, dx) + np.pi / 2.0
    w_d = _chord(D[:, None, 2], D[:, None, 3], perp - D[:, None, 5])
    w_p = _chord(P[None, :, 2], P[None, :, 3], perp - P[None, :, 5])
    lower_c = d * (w_d + w_p) / 2.0 * np.maximum(D[:, None, 4], P[None, :, 4])
    with np.errstate(divide="ignore", invalid="ignore"):
        bound = (D[:, None, 6] + P[None, :, 6]) / lower_c - 1.0
    return (d > r) & (lower_c > 0) & (bound < g)


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
