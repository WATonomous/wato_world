"""Browser viewer of a chunk's proposals.

Writes <chunk>/viz/proposals_<show>.html (wato_common.viz.point_cloud_html):
each frame's non-ground points (every point inside a box, a sample of the
rest) and the frame's box wireframes, coloured by source or class, over a grey
context cloud from every 20th sweep; a slider steps through frames in time
order.

    python -m wato_proposal_generation viz --view 3d --bag BAG --chunk 0005 --open
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wato_common.tracking import box_corners_3d, points_in_box
from wato_common.viz.point_cloud_html import (
    array_spec,
    empty_spec,
    render_html,
    sample_indices,
)
from wato_proposal_generation.boxes import box_of
from wato_proposal_generation.config import ComponentConfig
from wato_proposal_generation.viz import data

_MAX_POINTS_PER_FRAME = 8_000
_MAX_CONTEXT_PTS = 350_000
_CONTEXT_EVERY_K_SWEEPS = 20
_CONTEXT_COLOR = [0.30, 0.32, 0.36]
_IN_BOX_MARGIN_M = 0.2


def _payload(view: data.ChunkView, bag_id: str, chunk_id: str, show: str) -> dict:
    xyz, step, rgb = [], [], []
    box_step, corners, box_rgb = [], [], []
    for k, frame in enumerate(view.frames):
        pts = view.frame_points(frame)
        colors = np.tile(np.array(data.POINT_RGB, np.uint8), (len(pts), 1))
        in_any = np.zeros(len(pts), bool)
        for p in data.boxes_in(view, frame):
            inside = points_in_box(pts, box_of(p), _IN_BOX_MARGIN_M)
            colors[inside] = data.color(p, show)
            in_any |= inside
            box_step.append(k)
            corners.append(box_corners_3d(box_of(p)))
            box_rgb.append(data.color(p, show))
        # Every point a box claims, then a sample of the rest.
        rest = np.flatnonzero(~in_any)
        keep = np.concatenate(
            [
                np.flatnonzero(in_any),
                rest[sample_indices(len(rest), _MAX_POINTS_PER_FRAME, seed=k)],
            ]
        )
        xyz.append(pts[keep])
        rgb.append(colors[keep])
        step.append(np.full(len(keep), k, np.int32))

    context = []
    rows = sorted(
        view.sweeps.rows, key=lambda r: (r.reference_timestamp_ns, r.sweep_id)
    )
    for r in rows[::_CONTEXT_EVERY_K_SWEEPS]:
        world = view.sweeps.world(r.sweep_id)
        context.append(
            world.xyz if world.ground_mask is None else world.xyz[~world.ground_mask]
        )
    context = np.concatenate(context) if context else np.zeros((0, 3))
    context = context[sample_indices(len(context), _MAX_CONTEXT_PTS, seed=17)]

    xyz = (
        np.concatenate(xyz).astype(np.float32) if xyz else np.zeros((0, 3), np.float32)
    )
    rgb = np.concatenate(rgb) if rgb else np.zeros((0, 3), np.uint8)
    step = np.concatenate(step) if step else np.zeros(0, np.int32)
    parts = [a for a in (context.astype(np.float32), xyz) if len(a)]
    bounds = np.vstack(parts) if parts else np.zeros((1, 3))
    payload = {
        "title": f"proposal_generation chunk {chunk_id} — {show}",
        "bag_id": bag_id,
        "chunk_id": chunk_id,
        "mode": "chunk",
        "static_xyz": array_spec(context, np.float32),
        "dynamic_xyz": array_spec(xyz, np.float32),
        # The slider steps through frames in time order: sweep_ids are one
        # counter per chunk across LiDARs, not time-ordered.
        "dynamic_sweep_id": array_spec(step, np.int32),
        "dynamic_intensity": empty_spec((0,), np.float32),
        "dynamic_rgb": array_spec(rgb, np.uint8),
        "has_intensity": False,
        "bounds_min": bounds.min(axis=0).astype(float).tolist(),
        "bounds_max": bounds.max(axis=0).astype(float).tolist(),
        "static_label": "context",
        "dynamic_label": "frame points",
        "uniform_label": "by source" if show == "sources" else "by class",
        "all_label": "all frames",
        "static_color": _CONTEXT_COLOR,
        "counts": {"points": int(len(xyz)), "boxes": len(box_step)},
    }
    if box_step:
        payload["boxes"] = {
            "sweep_id": array_spec(np.array(box_step, np.int32), np.int32),
            "corners": array_spec(np.stack(corners), np.float32),
            "rgb": array_spec(np.array(box_rgb, np.uint8), np.uint8),
        }
    return payload


def write_browser(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
    *,
    show: str = "sources",
    every: int = 5,
    out_path: str | None = None,
) -> Path:
    """Every `every`-th frame's points and boxes; returns the page's path."""
    view = data.load(cfg, bag_id, chunk_id, show, every)
    out = Path(out_path or data.default_dir(bag_id, chunk_id, f"proposals_{show}.html"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        render_html(_payload(view, bag_id, chunk_id, show)), encoding="utf-8"
    )
    return out
