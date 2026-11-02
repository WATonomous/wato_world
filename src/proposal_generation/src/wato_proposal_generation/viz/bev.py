"""Top-down PNGs: one per frame, centred on the reference LiDAR.

The frame's non-ground points in grey and each shown box's outline, with a
short label, a legend and a 10 m scale bar. World x is right, y up.

    python -m wato_proposal_generation viz --view bev --bag BAG --chunk 0005
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from wato_common.tracking import bev_corners
from wato_proposal_generation.boxes import box_of
from wato_proposal_generation.config import ComponentConfig
from wato_proposal_generation.viz import data

_BG = (16, 17, 19)
_TEXT = (230, 233, 238)


def render_frame(
    points_xyz: np.ndarray,
    boxes: list,
    center_xy: np.ndarray,
    show: str,
    *,
    radius_m: float,
    size_px: int,
    title: str,
) -> Image.Image:
    scale = size_px / (2.0 * radius_m)

    def to_px(xy: np.ndarray) -> np.ndarray:
        u = (xy[:, 0] - center_xy[0]) * scale + size_px / 2.0
        v = size_px / 2.0 - (xy[:, 1] - center_xy[1]) * scale
        return np.column_stack([u, v])

    pixels = np.full((size_px, size_px, 3), _BG, np.uint8)
    if len(points_xyz):
        uv = to_px(points_xyz[:, :2]).astype(np.int64)
        inside = (uv >= 0).all(axis=1) & (uv < size_px).all(axis=1)
        pixels[uv[inside, 1], uv[inside, 0]] = data.POINT_RGB
    img = Image.fromarray(pixels)
    draw = ImageDraw.Draw(img)

    for p in boxes:
        corners = to_px(bev_corners(box_of(p)))
        rgb = data.color(p, show)
        draw.polygon([tuple(c) for c in corners], outline=rgb, width=2)
        top = corners[np.argmin(corners[:, 1])]
        draw.text((top[0] + 3, top[1] - 12), data.label(p, show), fill=rgb)

    draw.text((8, 8), title, fill=_TEXT)
    for k, (name, rgb) in enumerate(data.legend(show)):
        y = 26 + 14 * k
        draw.rectangle([8, y + 2, 18, y + 10], fill=rgb)
        draw.text((24, y), name, fill=_TEXT)
    bar = 10.0 * scale
    y = size_px - 16
    draw.line([(8, y), (8 + bar, y)], fill=_TEXT, width=2)
    draw.text((12 + bar, y - 7), "10 m", fill=_TEXT)
    return img


def write_bev(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
    *,
    show: str = "sources",
    every: int = 10,
    radius_m: float = 50.0,
    size_px: int = 900,
    out_dir: str | None = None,
) -> list[Path]:
    view = data.load(cfg, bag_id, chunk_id, show, every)
    out = Path(out_dir or data.default_dir(bag_id, chunk_id, f"proposals_bev_{show}"))
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for frame in view.frames:
        boxes = data.boxes_in(view, frame)
        center = view.sweeps.world(frame.reference_sweep_id).origin[:2]
        img = render_frame(
            view.frame_points(frame),
            boxes,
            center,
            show,
            radius_m=radius_m,
            size_px=size_px,
            title=(
                f"{bag_id} / {chunk_id} · sweep {frame.reference_sweep_id} · "
                f"{len(boxes)} {show} boxes"
            ),
        )
        path = out / f"{frame.reference_sweep_id:06d}.png"
        img.save(path)
        written.append(path)
    return written
