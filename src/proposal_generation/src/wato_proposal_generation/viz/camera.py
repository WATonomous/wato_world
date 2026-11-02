"""Camera overlays: each shown box projected into the images that see it.

An image shows the boxes of the frames its paired sweeps belong to, projected
with the ego pose at the image's own time (FrameIndex's world_T_ego). The boxes
are at the sweep's time, up to 44 ms away on WATO, so a moving car's box sits a
little off it. JPEG, since a chunk's images are large.

    python -m wato_proposal_generation viz --view camera --bag BAG --chunk 0005
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from wato_common.artifacts.calibration import Calibration
from wato_common.artifacts.frame_index import CameraFrame, FrameIndex
from wato_common.geometry import invert_se3, project_points
from wato_common.tracking import box_corners_3d
from wato_proposal_generation.boxes import box_of
from wato_proposal_generation.config import ComponentConfig
from wato_proposal_generation.viz import data

# box_corners_3d order: bottom face 0-3, top face 4-7.
EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)  # fmt: skip


def draw_boxes(
    img: Image.Image,
    boxes: list,
    K: np.ndarray,
    cam_T_world: np.ndarray,
    show: str,
) -> int:
    """Draw every box with a corner in the image; returns how many."""
    w, h = img.size
    draw = ImageDraw.Draw(img)
    n = 0
    for p in boxes:
        corners = box_corners_3d(box_of(p))
        uv, front = project_points(corners, K, cam_T_world)
        inside = (
            front & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        )
        if not inside.any():
            continue
        rgb = data.color(p, show)
        for a, b in EDGES:
            if front[a] and front[b]:
                draw.line([tuple(uv[a]), tuple(uv[b])], fill=rgb, width=3)
        top = uv[front][np.argmin(uv[front][:, 1])]
        draw.text((top[0] + 4, top[1] - 14), data.label(p, show), fill=rgb)
        n += 1
    return n


def _boxes_for(frame: CameraFrame, view: data.ChunkView, ref_of_sweep: dict) -> list:
    refs = sorted({ref_of_sweep[s] for s in frame.sweep_ids if s in ref_of_sweep})
    return [p for r in refs for p in view.by_frame.get(r, [])]


def write_camera(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
    *,
    show: str = "sources",
    every: int = 10,
    cameras: Optional[list[str]] = None,
    out_dir: str | None = None,
) -> list[Path]:
    """Every `every`-th image of each camera (all cameras unless `cameras`)."""
    view = data.load(cfg, bag_id, chunk_id, show)
    ref_of_sweep = {s: f.reference_sweep_id for f in view.frames for s in f.sweep_ids}
    calib = Calibration.load(bag_id)
    out = Path(out_dir or data.default_dir(bag_id, chunk_id, f"proposals_cam_{show}"))
    written = []
    for cam_id, frames in FrameIndex.load(bag_id, chunk_id).by_camera().items():
        if cameras and cam_id not in cameras:
            continue
        cam = calib.camera(cam_id)
        (out / cam_id).mkdir(parents=True, exist_ok=True)
        for frame in frames[:: max(1, every)]:
            if frame.world_T_ego is None:
                continue  # untrusted pose: nothing to project with
            img = Image.open(frame.image_path).convert("RGB")
            cam_T_world = invert_se3(cam.ego_T_cam) @ invert_se3(frame.world_T_ego)
            boxes = _boxes_for(frame, view, ref_of_sweep)
            n = draw_boxes(img, boxes, cam.K, cam_T_world, show)
            ImageDraw.Draw(img).text(
                (10, 10),
                f"{cam_id} seq {frame.camera_seq} · {n} {show} boxes",
                fill=(255, 255, 255),
            )
            path = out / cam_id / f"{frame.camera_seq:06d}.jpg"
            img.save(path, quality=90)
            written.append(path)
    return written
