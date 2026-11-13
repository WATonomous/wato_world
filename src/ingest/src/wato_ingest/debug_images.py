"""debug-bag's camera layers: masks and depth drawn as images (no ROS).

One rendered JPEG per camera image and layer, at the image's resolution:

  masks  the camera image with each of perception_2d's masklets in it tinted
         in a colour per masklet id (an id switch changes colour), outlined,
         and labelled with its class, id and score. Masks are independent
         claims, so overlaps blend.
  depth  perception_2d's metric depth on a fixed log scale (DEPTH_RANGE_M,
         turbo: red near, blue far and beyond; black = no measurement), with
         the frame's fit status, inlier count and RMSE, and a colour bar. A
         frame with no depth file (its fit failed) is black with "no depth".

Rendered rather than raw (mono16 depth, label images) because Foxglove's image
panel shows one image per panel and can't layer a mask over its camera image,
and raw full-resolution depth would add tens of GB to a whole WATO bag. The
values stay in the artifacts: perception_2d's `viz` reads them.

render() runs in debug-bag's worker processes: it reads the artifacts
through wato_common's read halves and returns encoded JPEGs.
"""

from __future__ import annotations

import colorsys
import io
import math
import zlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from wato_common.artifacts import depth, masklets
from wato_common.artifacts.masklets import MaskletRecord

MASKS = "masks"
DEPTH = "depth"
LAYERS = (MASKS, DEPTH)
MASK_ALPHA = 0.45
OUTLINE_PX = 2
DEPTH_RANGE_M = (1.0, 100.0)  # colour scale ends; nearer / farther saturate
DEPTH_TICKS_M = (1, 2, 5, 10, 20, 50, 100)
JPEG_QUALITY = 85
TITLE_PX = 18
LABEL_PX = 14  # a masklet's label: far cars crowd the horizon


@dataclass(frozen=True)
class ImageFrame:
    """One camera image to draw: where its artifacts are and the stamps its
    source message carries, so the drawn images replay in step with it."""

    bag_id: str
    chunk_id: str
    cam_id: str
    camera_seq: int
    image_path: str  # ingest's copy (local path)
    header_ns: int  # the source message's header stamp
    log_ns: int  # its log time
    frame_id: str  # its header.frame_id
    masklets: tuple[MaskletRecord, ...] = ()  # those with a mask in this image


def id_rgb(key: str) -> tuple[float, float, float]:
    """A colour per id, the same on every run; neighbouring ids differ."""
    hue = (zlib.crc32(key.encode()) * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
    return (r, g, b)


# ---------------------------------------------------------------------------
# Drawing (arrays in, arrays out).
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def turbo_lut() -> np.ndarray:
    """(256, 3) uint8 turbo colormap, from its published polynomial fit
    (Mikhailov 2019; fit by Ruofei Du): 0 = blue, 255 = dark red. The fit's
    first 8% fades to near-black, which would read as "no depth", so the
    scale starts past it."""
    x = np.linspace(0.08, 1.0, 256)
    r = 0.13572138 + x * (
        4.61539260
        + x
        * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943)))
    )
    g = 0.09140261 + x * (
        2.19418839
        + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604)))
    )
    b = 0.10667330 + x * (
        12.64194608
        + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973)))
    )
    rgb = np.clip(np.stack([r, g, b], axis=1), 0.0, 1.0)
    return np.round(rgb * 255).astype(np.uint8)


def depth_t(depth_m: np.ndarray) -> np.ndarray:
    """Position on the colour scale: 1 at DEPTH_RANGE_M's near end, 0 at its
    far end, log-spaced (10 m mid-scale), clipped outside it."""
    near, far = DEPTH_RANGE_M
    d = np.clip(depth_m, near, far)
    return (math.log(far) - np.log(d)) / (math.log(far) - math.log(near))


def colour_depth(depth_m: np.ndarray) -> np.ndarray:
    """(H, W) metres -> (H, W, 3) uint8; no measurement (<= 0 or not
    finite) is black."""
    valid = np.isfinite(depth_m) & (depth_m > 0)
    t = depth_t(np.where(valid, depth_m, DEPTH_RANGE_M[1]))
    out = turbo_lut()[np.round(t * 255).astype(np.uint8)]
    out[~valid] = 0
    return out


def _erode(m: np.ndarray) -> np.ndarray:
    """One step of 4-neighbour erosion (pixels on the image edge keep their
    outside neighbour)."""
    e = m.copy()
    e[1:] &= m[:-1]
    e[:-1] &= m[1:]
    e[:, 1:] &= m[:, :-1]
    e[:, :-1] &= m[:, 1:]
    return e


def outline(m: np.ndarray, px: int = OUTLINE_PX) -> np.ndarray:
    """The mask's pixels within px of its boundary."""
    inner = m
    for _ in range(px):
        inner = _erode(inner)
    return m & ~inner


def tint(image: np.ndarray, mask_rgbs: list[tuple[np.ndarray, tuple]]) -> np.ndarray:
    """image (H, W, 3) uint8 with each (bool mask, rgb in [0, 1]) blended in
    at MASK_ALPHA, in order, then outlined in its colour."""
    out = image.astype(np.float32)
    for m, rgb in mask_rgbs:
        if m.shape != image.shape[:2]:
            raise ValueError(f"mask {m.shape} is not the image's {image.shape[:2]}")
        c = np.asarray(rgb, dtype=np.float32) * 255.0
        out[m] = out[m] * (1.0 - MASK_ALPHA) + c * MASK_ALPHA
    for m, rgb in mask_rgbs:
        out[outline(m)] = np.asarray(rgb, dtype=np.float32) * 255.0
    return np.round(out).astype(np.uint8)


@lru_cache(maxsize=None)
def _font(px: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=px)
    except TypeError:  # Pillow without FreeType: the fixed bitmap font
        return ImageFont.load_default()


def _label(draw, size, xy, text: str, fill, ink=(255, 255, 255), px=TITLE_PX):
    """text on a solid box with its top-left at xy, kept inside an image of
    size (W, H)."""
    font = _font(px)
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    w, h = x1 - x0 + 6, y1 - y0 + 4
    W, H = size
    x = min(max(int(xy[0]), 0), max(W - w, 0))
    y = min(max(int(xy[1]), 0), max(H - h, 0))
    draw.rectangle((x, y, x + w, y + h), fill=fill)
    draw.text((x + 3 - x0, y + 2 - y0), text, fill=ink, font=font)


def _rgb255(rgb) -> tuple[int, int, int]:
    return tuple(int(round(c * 255)) for c in rgb)


def draw_masks(
    image: np.ndarray, masks: list[tuple[np.ndarray, tuple, str]], title: str
) -> np.ndarray:
    """The image with each (mask, rgb, text) tinted, outlined and labelled
    above its top-left corner, and title in the image's top-left."""
    from PIL import Image, ImageDraw

    pic = Image.fromarray(tint(image, [(m, rgb) for m, rgb, _ in masks]))
    draw = ImageDraw.Draw(pic)
    for m, rgb, text in masks:
        rows, cols = np.flatnonzero(m.any(axis=1)), np.flatnonzero(m.any(axis=0))
        if rows.size == 0:
            continue
        xy = (cols[0], rows[0] - LABEL_PX - 6)
        _label(draw, pic.size, xy, text, _rgb255(rgb), (0, 0, 0), LABEL_PX)
    _label(draw, pic.size, (0, 0), title, (0, 0, 0))
    return np.asarray(pic)


def draw_depth(depth_m: np.ndarray | None, size: tuple[int, int], title: str):
    """Coloured depth with title and a colour bar; black with the title alone
    when depth_m is None. size: (W, H) of the camera image."""
    from PIL import Image, ImageDraw

    if depth_m is None:
        pic = Image.new("RGB", size)
    else:
        pic = Image.fromarray(colour_depth(depth_m))
    draw = ImageDraw.Draw(pic)
    _label(draw, pic.size, (0, 0), title, (0, 0, 0))
    if depth_m is not None:
        _colour_bar(draw, pic.size)
    return np.asarray(pic)


def _colour_bar(draw, size: tuple[int, int]) -> None:
    """DEPTH_RANGE_M's scale along the image's bottom-left, near on the left."""
    W, H = size
    bar_w, bar_h, pad = min(400, W - 20), 14, 10
    x0, y0 = pad, H - pad - bar_h - TITLE_PX - 8
    lut = turbo_lut()
    for i in range(bar_w):
        c = lut[int(round((1.0 - i / (bar_w - 1)) * 255))]
        draw.line((x0 + i, y0, x0 + i, y0 + bar_h), fill=tuple(int(v) for v in c))
    for d in DEPTH_TICKS_M:
        x = x0 + (1.0 - float(depth_t(np.float32(d)))) * (bar_w - 1)
        draw.line((x, y0 + bar_h, x, y0 + bar_h + 4), fill=(255, 255, 255))
        text = f"{d} m" if d == DEPTH_TICKS_M[-1] else f"{d}"
        _label(draw, size, (x - 6, y0 + bar_h + 5), text, (0, 0, 0))


def jpeg(image: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# One camera image's layers, from the artifacts.
# ---------------------------------------------------------------------------


def mask_title(f: ImageFrame) -> str:
    return f"{f.cam_id} #{f.camera_seq} chunk {f.chunk_id}: {len(f.masklets)} masks"


def mask_text(r: MaskletRecord) -> str:
    return f"{r.cls} {r.masklet_id} {r.score:.2f}"


def depth_title(f: ImageFrame, d: depth.DepthFrame | None) -> str:
    head = f"{f.cam_id} #{f.camera_seq} chunk {f.chunk_id}"
    if d is None:
        return f"{head}: no depth"
    return (
        f"{head}: fit {d.fit_status.name}, {d.n_inliers}/{d.n_anchors} inliers, "
        f"rmse {d.rmse_inliers_m:.2f} m"
    )


def render(f: ImageFrame, layers: tuple[str, ...]) -> dict[str, bytes]:
    """layer -> the JPEG drawn for f."""
    from PIL import Image

    out: dict[str, bytes] = {}
    with Image.open(f.image_path) as im:
        size = im.size
        image = np.asarray(im.convert("RGB")) if MASKS in layers else None
    if MASKS in layers:
        masks = [
            (
                masklets.read_mask(f.bag_id, f.chunk_id, r.masklet_id, f.camera_seq),
                id_rgb(r.masklet_id),
                mask_text(r),
            )
            for r in f.masklets
        ]
        out[MASKS] = jpeg(draw_masks(image, masks, mask_title(f)))
    if DEPTH in layers:
        d = depth.read_frame(f.bag_id, f.chunk_id, f.cam_id, f.camera_seq)
        if d is not None and d.depth_m.shape != (size[1], size[0]):
            raise ValueError(
                f"{f.cam_id} #{f.camera_seq}: depth {d.depth_m.shape} is not the "
                f"image's {(size[1], size[0])}"
            )
        out[DEPTH] = jpeg(
            draw_depth(None if d is None else d.depth_m, size, depth_title(f, d))
        )
    return out
