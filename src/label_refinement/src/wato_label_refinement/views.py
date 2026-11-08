"""What SLF fits a track to (slf.Problem), from the artifacts.

A view is one camera image in which one of the track's masklets has a mask
(perception_2d), with the camera's pose at the image's own time (frame_index's
CameraFrame): a grid of rays through a crop around the track's box and the
mask. Pixels a nearer tracked object's mask covers are left out (the object
may continue behind it). A moving track's box is moved by its velocity from
its frame's time to the image's, up to 50 ms apart.

Views are spread over the track's life: the candidates are split into
max_views stretches of time and the one whose box projects largest is kept
from each.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np

from wato_common.artifacts.calibration import Calibration
from wato_common.artifacts.frame_index import CameraFrame, FrameIndex
from wato_common.artifacts.masklets import MaskletIndex, MaskletRecord
from wato_common.artifacts.tracks import TrackFrame
from wato_common.geometry import CameraModel, invert_se3, project_with_depth as project
from wato_common.schemas import Box3D
from wato_common.tracking import bev_corners
from wato_label_refinement import refine, slf
from wato_label_refinement.config import SlfConfig

# A box corner nearer the camera than this is behind it for projection.
_MIN_DEPTH_M = 0.3


@dataclass(frozen=True)
class Chunk:
    """A chunk's camera frames and masklets, and every track box per sweep
    (occluders)."""

    frame_index: FrameIndex
    masklets: MaskletIndex
    by_id: dict[str, MaskletRecord]
    boxes_at: dict[int, list[tuple[str, tuple[str, ...], Box3D]]]

    @classmethod
    def load(cls, bag_id: str, chunk_id: str, rows: list[TrackFrame]) -> "Chunk":
        masklets = MaskletIndex.load(bag_id, chunk_id)
        boxes_at: dict[int, list] = defaultdict(list)
        for f in rows:
            if f.chunk_id == chunk_id:
                boxes_at[f.sweep_id].append(
                    (f.track_id, f.supporting_masklet_ids, box_of(f))
                )
        return cls(
            FrameIndex.load(bag_id, chunk_id),
            masklets,
            {r.masklet_id: r for r in masklets.records},
            dict(boxes_at),
        )


@dataclass(frozen=True)
class View:
    frame: int  # index into the track's frames
    camera: CameraFrame
    masklets: tuple[MaskletRecord, ...]  # the track's, with a mask in this image
    world_T_cam: np.ndarray
    area_px: float  # the observed box's projected area, clipped to the image


def box_of(f) -> Box3D:
    return Box3D(cx=f.cx, cy=f.cy, cz=f.cz, w=f.w, l=f.l, h=f.h, heading=f.heading)


def corners(box: Box3D) -> np.ndarray:
    """(8, 3) world corners."""
    xy = bev_corners(box)
    z0, z1 = box.cz - box.h / 2, box.cz + box.h / 2
    return np.vstack([np.column_stack([xy, np.full(4, z)]) for z in (z0, z1)])


def _rect(uv: np.ndarray, width: int, height: int) -> Optional[tuple]:
    """The projected points' bounding rectangle clipped to the image, or None
    if it misses the image."""
    if len(uv) == 0 or not np.isfinite(uv).all():
        return None
    u0, v0 = np.maximum(uv.min(0), 0)
    u1, v1 = np.minimum(uv.max(0), [width, height])
    if u1 <= u0 or v1 <= v0:
        return None
    return float(u0), float(v0), float(u1), float(v1)


def candidates(
    frames: list[TrackFrame],
    chunks: dict[str, Chunk],
    calib: Calibration,
    cfg: SlfConfig,
) -> list[View]:
    """Every image where one of a frame's masklets has a mask and its
    observed box projects onto the image, in time order."""
    out = []
    for k, f in enumerate(frames):
        ch = chunks[f.chunk_id]
        mine = [ch.by_id[m] for m in f.supporting_masklet_ids if m in ch.by_id]
        for cf in ch.frame_index.for_sweep(f.sweep_id):
            if cf.world_T_ego is None:
                continue
            recs = tuple(
                r
                for r in mine
                if r.cam_id == cf.cam_id and cf.camera_seq in r.frames_present
            )
            if not recs:
                continue
            cam = calib.camera(cf.cam_id)
            world_T_cam = cf.world_T_ego @ cam.ego_T_cam
            uv, z = project(corners(box_of(f)), cam.model, invert_se3(world_T_cam))
            if (z < _MIN_DEPTH_M).any() or cam.width is None or cam.height is None:
                continue
            rect = _rect(uv, cam.width, cam.height)
            if rect is None:
                continue
            area = (rect[2] - rect[0]) * (rect[3] - rect[1])
            if area >= cfg.min_mask_px:
                out.append(View(k, cf, recs, world_T_cam, area))
    return sorted(out, key=lambda v: (v.camera.timestamp_ns, v.camera.cam_id))


def select(views: list[View], max_views: int) -> list[View]:
    """At most max_views, the largest from each of max_views stretches of
    the candidates' time order."""
    if len(views) <= max_views:
        return views
    edges = np.linspace(0, len(views), max_views + 1).round().astype(int)
    return [max(views[a:b], key=lambda v: v.area_px) for a, b in zip(edges, edges[1:])]


def _occluders(
    view: View,
    track_id: str,
    ch: Chunk,
    sweep_id: int,
    camera_model: CameraModel,
    cam_T_world: np.ndarray,
    depth: float,
    rect: tuple,
    margin_m: float,
    size: tuple[int, int],
) -> np.ndarray:
    """(H, W) bool: pixels of nearer tracked objects' masks in the image."""
    mask = np.zeros(size[::-1], bool)
    for tid, masklet_ids, box in ch.boxes_at.get(sweep_id, []):
        if tid == track_id:
            continue
        uv, z = project(corners(box), camera_model, cam_T_world)
        if (z < _MIN_DEPTH_M).any():
            continue
        if project(np.array([[box.cx, box.cy, box.cz]]), camera_model, cam_T_world)[1][
            0
        ] > (depth - margin_m):
            continue
        r = _rect(uv, *size)
        if (
            r is None
            or r[2] < rect[0]
            or r[0] > rect[2]
            or r[3] < rect[1]
            or r[1] > rect[3]
        ):
            continue
        for m in masklet_ids:
            rec = ch.by_id.get(m)
            if (
                rec is not None
                and rec.cam_id == view.camera.cam_id
                and view.camera.camera_seq in rec.frames_present
            ):
                mask |= ch.masklets.mask(rec, view.camera.camera_seq)
    return mask


def problem(
    track_id: str,
    cls: Optional[str],
    frames: list[TrackFrame],
    boxes: list[Box3D],
    velocity: np.ndarray,
    ego_xy: np.ndarray,
    anchored: bool,
    views: list[View],
    points: dict[int, np.ndarray],
    chunks: dict[str, Chunk],
    calib: Calibration,
    cfg: SlfConfig,
    yaw_mode: str = "fixed",
) -> slf.Problem:
    """The fit for one track. frames, boxes (Phase 3's), velocity (N, 2) and
    ego_xy (N, 2) are the track's, one per instant in time order; points maps
    a frame index to the returns on its observed box. anchored: size changes
    extend from the near corner relative to Phase 3's smoothed positions; else
    each keeps its centre (a parked box, a pedestrian: refine.refine_moving's
    centred_classes). yaw_mode: slf.Problem's."""
    # The accepted fit writes every frame, including those with no selected
    # image. Evaluate one consistent placement for that whole output track.
    used = list(range(len(frames)))
    index = {k: i for i, k in enumerate(used)}
    heading = np.array([boxes[k].heading for k in used])
    centre = np.array([[boxes[k].cx, boxes[k].cy] for k in used])
    t_ns = np.array([frames[k].reference_timestamp_ns for k in used], np.int64)
    lo, hi, near = (np.zeros((len(used), 2)) for _ in range(3))
    obs_corners = np.zeros((len(used), 4, 2))
    if anchored:
        for i, k in enumerate(used):
            observed = box_of(frames[k])
            a = refine.anchors(observed, heading[i], ego_xy[k])
            lo[i] = [a[0].lo, a[1].lo]
            hi[i] = [a[0].hi, a[1].hi]
            near[i] = [a[0].near_lo, a[1].near_lo]
            # What the fit recomputes the anchors' extents from along a
            # corrected heading (slf._Batch._pose), as refine.anchors does.
            obs_corners[i] = bev_corners(observed) - [observed.cx, observed.cy]
        # Anchor size changes relative to Phase 3's already-smoothed pose.
        # At size0, zero offset must reproduce Phase 3 exactly. Starting from
        # raw observations instead would fit an unsmoothed trajectory and
        # export a different one after moving() smoothed it a second time.
        initial_size = np.array([boxes[0].l, boxes[0].w])
        along = np.where(
            hi - lo >= initial_size,
            (lo + hi) / 2,
            np.where(near.astype(bool), lo + initial_size / 2, hi - initial_size / 2),
        )
        c, s = np.cos(heading), np.sin(heading)
        centre -= np.column_stack(
            [along[:, 0] * c - along[:, 1] * s, along[:, 0] * s + along[:, 1] * c]
        )
    ground = np.array([boxes[k].cz - boxes[k].h / 2 for k in used])

    r = cfg.crop_px
    grid = (np.arange(r) + 0.5) / r
    origin, dirs, target, valid, footprint, view_frame, shift = ([] for _ in range(7))
    for v in views:
        f, box = frames[v.frame], boxes[v.frame]
        cam = calib.camera(v.camera.cam_id)
        size = (cam.width, cam.height)
        cam_T_world = invert_se3(v.world_T_cam)
        dt = (v.camera.timestamp_ns - f.reference_timestamp_ns) * 1e-9
        moved = Box3D(
            cx=box.cx + velocity[v.frame][0] * dt,
            cy=box.cy + velocity[v.frame][1] * dt,
            cz=box.cz,
            w=box.w,
            l=box.l,
            h=box.h,
            heading=box.heading,
        )
        uv, corner_depth = project(corners(moved), cam.model, cam_T_world)
        if (corner_depth < _MIN_DEPTH_M).any() or not np.isfinite(uv).all():
            continue
        depth = float(
            project(np.array([[moved.cx, moved.cy, moved.cz]]), cam.model, cam_T_world)[
                1
            ][0]
        )
        mask = np.zeros(size[::-1], bool)
        for rec in v.masklets:
            observed = chunks[f.chunk_id].masklets.mask(rec, v.camera.camera_seq)
            if observed.shape != mask.shape:
                raise ValueError(
                    f"camera {cam.cam_id}: mask shape {observed.shape} disagrees "
                    f"with calibrated image shape {mask.shape}"
                )
            mask |= observed
        ys, xs = np.nonzero(mask)
        pts = uv if xs.size == 0 else np.vstack([uv, np.column_stack([xs, ys])])
        rect = _rect(pts, *size)
        if rect is None:
            continue
        pad_u = cfg.crop_margin * (rect[2] - rect[0]) + 2
        pad_v = cfg.crop_margin * (rect[3] - rect[1]) + 2
        u0, v0 = max(rect[0] - pad_u, 0), max(rect[1] - pad_v, 0)
        u1, v1 = min(rect[2] + pad_u, size[0]), min(rect[3] + pad_v, size[1])
        uu, vv = np.meshgrid(u0 + grid * (u1 - u0), v0 + grid * (v1 - v0))
        uu, vv = uu.ravel(), vv.ravel()
        camera_points = cam.model.unproject(np.column_stack([uu, vv]), depth)
        d_cam = camera_points / np.linalg.norm(camera_points, axis=1, keepdims=True)
        # Integer image coordinates denote pixel centres, as in the shared
        # projector and lifting. Flooring would shift supervision by up to
        # one pixel relative to the ray that is actually rendered.
        iu = np.clip(np.rint(uu).astype(int), 0, size[0] - 1)
        iv = np.clip(np.rint(vv).astype(int), 0, size[1] - 1)
        hidden = _occluders(
            v,
            track_id,
            chunks[f.chunk_id],
            f.sweep_id,
            cam.model,
            cam_T_world,
            depth,
            (u0, v0, u1, v1),
            cfg.occluder_margin_m,
            size,
        )
        origin.append(v.world_T_cam[:3, 3])
        dirs.append(d_cam @ v.world_T_cam[:3, :3].T)
        target.append(mask[iv, iu].astype(np.float32))
        valid.append(~hidden[iv, iu] | mask[iv, iu])
        # Separation on the optical-z plane gives the actual metric pixel
        # footprint, including unequal focal lengths, skew and distortion.
        plane = camera_points.reshape(r, r, 3)
        du = np.linalg.norm(np.diff(plane, axis=1), axis=-1)
        dv = np.linalg.norm(np.diff(plane, axis=0), axis=-1)
        footprint.append(max(float(np.median(du)), float(np.median(dv))))
        view_frame.append(index[v.frame])
        shift.append(velocity[v.frame] * dt)

    pts_list, pf = [], []
    for k in used:
        p = points.get(k)
        if p is not None and len(p):
            pts_list.append(p)
            pf.append(np.full(len(p), index[k]))
    pts = np.concatenate(pts_list) if pts_list else np.zeros((0, 3))
    pf = np.concatenate(pf) if pf else np.zeros(0, int)
    if len(pts) > cfg.max_points:
        keep = np.random.default_rng(0).choice(len(pts), cfg.max_points, replace=False)
        pts, pf = pts[np.sort(keep)], pf[np.sort(keep)]

    prior = cfg.size_prior_m.get(cls) if cls is not None else None
    b0 = boxes[used[0]]
    n = len(origin)
    return slf.Problem(
        size0=np.array([b0.l, b0.w, b0.h]),
        prior=None if prior is None else np.asarray(prior, float),
        heading=heading,
        t_s=(t_ns - t_ns[0]) * 1e-9,
        centre_obs=centre,
        obs_corners=obs_corners,
        anchor_near_lo=near.astype(bool),
        anchored=np.full(len(used), anchored),
        ground_z=ground,
        view_frame=np.array(view_frame, int),
        view_shift=np.array(shift).reshape(n, 2),
        origin=np.array(origin).reshape(n, 3),
        dirs=np.array(dirs).reshape(n, r * r, 3),
        target=np.array(target).reshape(n, r * r),
        valid=np.array(valid).reshape(n, r * r),
        footprint=np.array(footprint),
        points=pts,
        point_frame=pf,
        enclose=cls in cfg.enclosing_classes,
        yaw_mode=yaw_mode,
    )


def object_points(xyz: np.ndarray, box: Box3D, margin_m: float) -> np.ndarray:
    """The returns inside an observed box grown by margin_m."""
    from wato_common.tracking import points_in_box

    return xyz[points_in_box(xyz, box, margin_m)]
