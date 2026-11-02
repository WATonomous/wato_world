"""Evidence for a vehicle's ego mask (``ego_mask`` in its config), from a bag's
perception_2d output.

Pixel statistics alone don't find the body: over a drive the rack's grey-level
spread is as large as the scene's (reflections, sun, shadow), and between
consecutive images sky and road barely change either. What does find it is the
pipeline itself: the detector boxes the rack or hood as a vehicle and SAM2
holds it for the whole chunk, a mask at the image edge that doesn't move while
the vehicle drives. So, for a new vehicle, run perception_2d once without an
ego mask on a driving bag, then ``ego-mask`` writes, per camera:

  <cam>_median.png   per-pixel median of images taken metres apart: the scene
                     blurs out, the body stays sharp (half resolution);
  <cam>_overlay.png  that median, the body masklets' masks in red and the
                     config's polygons outlined in green;
  ego_mask.yaml      a draft ``ego_mask`` block: the body masks' union,
                     dilated ``margin_px``, simplified to within 2 px.

Review the overlays before pasting the block: a part the detector never boxed
(the WATO antenna) has no masklet and must be drawn by hand from the median.

Run on the output of a run that used the polygons, any body masklet left is
part of the vehicle they miss (a camera moved, a rack added); its share of
pixels outside the polygons is reported per camera.
"""

from __future__ import annotations

import logging
import os
import textwrap
from typing import Optional

import numpy as np

from wato_common.artifacts import masklets
from wato_common.artifacts.frame_index import CameraFrame, FrameIndex
from wato_perception_2d.config import ComponentConfig
from wato_perception_2d.io import load_chunks

log = logging.getLogger(__name__)

# A body masklet spans at least this share of its camera's images in the
# chunk, touches the image border in every one of BODY_SAMPLES evenly spaced
# masks, each at mask IoU >= BODY_MIN_IOU with the first, while the vehicle
# drives at least BODY_MIN_TRAVEL_M (so a parked chunk, where everything holds
# still, proves nothing).
BODY_MIN_SPAN = 0.5
BODY_SAMPLES = 8
BODY_MIN_IOU = 0.85
BODY_MIN_TRAVEL_M = 20.0
# Median: up to this many images per camera, each at least MEDIAN_STEP_M of
# driving after the previous one.
MEDIAN_IMAGES = 150
MEDIAN_STEP_M = 2.0


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return np.count_nonzero(a & b) / max(1, np.count_nonzero(a | b))


def _touches_border(mask: np.ndarray) -> bool:
    return bool(
        mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
    )


def is_body(masks: list[np.ndarray], *, span: float, travel_m: float) -> bool:
    """True when a masklet's sampled ``masks`` are the vehicle's own body:
    it spans ``span`` of its camera's images, every mask touches the image
    border and holds still while the vehicle travels ``travel_m``."""
    return (
        bool(masks)
        and span >= BODY_MIN_SPAN
        and travel_m >= BODY_MIN_TRAVEL_M
        and all(_touches_border(m) for m in masks)
        and min(_iou(masks[0], m) for m in masks) >= BODY_MIN_IOU
    )


def draft_polygons(mask: np.ndarray, margin_px: int) -> list[list[list[int]]]:
    """Outer outlines of ``mask`` dilated by ``margin_px``, simplified to
    within 2 px; outlines under 200 px² are dropped."""
    import cv2

    m = mask.astype(np.uint8)
    if margin_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1)
        )
        m = cv2.dilate(m, k)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [
        cv2.approxPolyDP(c, 2.0, True)[:, 0, :].tolist()
        for c in contours
        if cv2.contourArea(c) >= 200
    ]


def format_block(
    polygons: dict[str, list[list[list[int]]]],
    image_size: tuple[int, int],
    max_fraction: float,
) -> str:
    """An ``ego_mask`` config block, one polygon per wrapped flow list."""
    lines = [
        "ego_mask:",
        f"  image_size: [{image_size[0]}, {image_size[1]}]",
        f"  max_fraction: {max_fraction}",
        "  polygons:" + ("" if polygons else " {}"),
    ]
    for cam in sorted(polygons):
        lines.append(f"    {cam}:")
        for poly in polygons[cam]:
            body = ", ".join(f"[{x}, {y}]" for x, y in poly)
            wrapped = textwrap.wrap(f"[{body}]", 72, break_long_words=False)
            lines.append("      - " + wrapped[0])
            lines.extend("         " + w for w in wrapped[1:])
    return "\n".join(lines) + "\n"


def _positions(frames: list[CameraFrame]) -> dict[int, Optional[np.ndarray]]:
    """camera_seq → the vehicle's position when the image was taken (None
    without a valid pose)."""
    return {
        f.camera_seq: None if f.world_T_ego is None else f.world_T_ego[:3, 3]
        for f in frames
    }


def _travel(seqs: list[int], pos: dict[int, Optional[np.ndarray]]) -> float:
    pts = [pos[s] for s in seqs if pos.get(s) is not None]
    return float(sum(np.linalg.norm(b - a) for a, b in zip(pts, pts[1:])))


def run(
    cfg: ComponentConfig, *, bag_id: str, out_dir: str, margin_px: int = 6
) -> dict[str, float]:
    """Write the evidence and draft block for ``bag_id`` to ``out_dir``.

    Returns, per camera with body masklets, the share of their pixels outside
    the config's polygons (1.0 for a camera the config doesn't list).
    """
    from PIL import Image, ImageDraw

    from wato_perception_2d.pipeline import _ego_mask

    os.makedirs(out_dir, exist_ok=True)
    body: dict[str, np.ndarray] = {}
    n_body: dict[str, int] = {}
    median_paths: dict[str, list[str]] = {}
    for chunk in load_chunks(bag_id):
        chunk_id = str(chunk["chunk_id"])
        if not masklets.exists(bag_id, chunk_id):
            continue
        by_cam = FrameIndex.load(bag_id, chunk_id).by_camera()
        pos = {cam: _positions(fs) for cam, fs in by_cam.items()}
        for cam, fs in by_cam.items():
            last = None
            for f in fs:
                p = pos[cam][f.camera_seq]
                if p is not None and (
                    last is None or np.linalg.norm(p - last) >= MEDIAN_STEP_M
                ):
                    median_paths.setdefault(cam, []).append(f.image_path)
                    last = p
        for rec in masklets.read(bag_id, chunk_id):
            cam_seqs = sorted(pos.get(rec.cam_id, {}))
            seqs = list(rec.frames_present)
            if not cam_seqs or len(seqs) < BODY_MIN_SPAN * len(cam_seqs):
                continue
            pick = np.linspace(0, len(seqs) - 1, min(BODY_SAMPLES, len(seqs)))
            masks = [
                masklets.read_mask(bag_id, chunk_id, rec.masklet_id, seqs[i])
                for i in sorted(set(pick.round().astype(int)))
            ]
            # Every 10th image, so pose jitter doesn't add up to metres.
            span = [s for s in cam_seqs if seqs[0] <= s <= seqs[-1]][::10]
            if not is_body(
                masks,
                span=len(seqs) / len(cam_seqs),
                travel_m=_travel(span, pos[rec.cam_id]),
            ):
                continue
            union = np.logical_or.reduce(masks)
            body[rec.cam_id] = union | body.get(rec.cam_id, np.zeros_like(union))
            n_body[rec.cam_id] = n_body.get(rec.cam_id, 0) + 1

    uncovered: dict[str, float] = {}
    drafts: dict[str, list[list[list[int]]]] = {}
    size: Optional[tuple[int, int]] = None
    for cam in sorted(set(median_paths) | set(body)):
        paths = median_paths.get(cam, [])
        if len(paths) > MEDIAN_IMAGES:
            paths = [
                paths[i]
                for i in np.linspace(0, len(paths) - 1, MEDIAN_IMAGES).astype(int)
            ]
        median = None
        if paths:
            stack = np.stack(
                [np.asarray(Image.open(p).convert("RGB").reduce(2)) for p in paths]
            )
            median = Image.fromarray(np.median(stack, axis=0).astype(np.uint8))
            median.save(os.path.join(out_dir, f"{cam}_median.png"))
            W, H = median.size[0] * 2, median.size[1] * 2
        mask = body.get(cam)
        if mask is not None:
            H, W = mask.shape
            drafts[cam] = draft_polygons(mask, margin_px)
        if median is None and mask is None:
            continue
        size = (W, H)
        configured = _ego_mask(cfg, cam, W, H) if cam in cfg.ego_mask.polygons else None
        if mask is not None:
            uncovered[cam] = (
                1.0
                if configured is None
                else np.count_nonzero(mask & ~configured) / np.count_nonzero(mask)
            )
        if median is not None:
            overlay = np.asarray(median).copy()
            if mask is not None:
                red = mask[::2, ::2][: overlay.shape[0], : overlay.shape[1]]
                overlay[red] = (0.5 * overlay[red] + [127, 0, 0]).astype(np.uint8)
            img = Image.fromarray(overlay)
            draw = ImageDraw.Draw(img)
            for poly in cfg.ego_mask.polygons.get(cam, []):
                pts = [(x / 2, y / 2) for x, y in poly]
                draw.line(pts + pts[:1], fill=(0, 255, 0), width=1)
            img.save(os.path.join(out_dir, f"{cam}_overlay.png"))
        log.info(
            "%s: %d body masklets%s, %d images in the median",
            cam,
            n_body.get(cam, 0),
            f", {100 * uncovered[cam]:.1f}% of their pixels outside the config's polygons"
            if cam in uncovered
            else "",
            len(paths),
        )

    with open(os.path.join(out_dir, "ego_mask.yaml"), "w") as fh:
        fh.write(
            f"# Draft from {bag_id}:\n"
            "# per camera, the union of the masks of masklets that held still at\n"
            f"# the image border while the vehicle drove, dilated {margin_px} px.\n"
            "# Check <cam>_overlay.png before using it.\n"
        )
        fh.write(format_block(drafts, size or (0, 0), cfg.ego_mask.max_fraction))
    return uncovered
