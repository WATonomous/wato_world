"""Detector + SAM2 video tracker — the primary (only) tracker.

For one camera stream, runs the 2D detector on keyframes and uses SAM2's video
predictor to segment + propagate each detected box into a tracked masklet:

  1. Point SAM2's video loader at the camera's frames: a temp dir of
     sequentially-named JPEGs — symlinks to ingest's own JPEGs when the source
     already is one, re-encoded otherwise — loaded asynchronously by SAM2.
  2. At each keyframe (every ``redetect_every_k`` frames) run the detector →
     class-labeled boxes, dropping those mostly on the ego vehicle. A box
     matching a tracked object (box IoU with its mask bbox ≥
     ``iou_match_threshold``) supports that track (``_Support``); ended tracks
     (below) leave the session. The other boxes, less repeats of a
     higher-scoring box at the same keyframe, get a fresh obj_id via
     ``add_new_points_or_box``.
  3. ``propagate_in_video`` forward to the next keyframe, recording per-object
     masks → ``_Track`` → ``Masklet`` (per-frame PNGs written on a thread pool,
     DINOv2 features every k frames batched over the frame's objects). Each
     mask loses the ego vehicle's pixels and its specks (``_drop_islands``); a
     pixel several masks claim goes to the highest logit. A track ends — and
     is never recorded again — when SAM2 loses its object for
     ``drop_lost_after_frames`` frames, its mask jumps onto another live
     track's object (``_EndedTracks``) or is left owning little of its own
     mask (``_owned_masks``), since SAM2 latches a lost object's id onto a
     different, similar object. It also ends when its mask lies on the ego
     vehicle or inside a vehicle (``_contained``), or when detections stop
     supporting it.

Cost: SAM2 runs memory attention per *object* per frame and keeps every object
it was given, so without pruning cost and VRAM grow with every object ever
detected. Removing ended tracks at the next keyframe bounds that. SAM2 runs
under ``autocast_dtype`` (bfloat16 by default); the detector and DINOv2 stay
float32.

Object ids reset between OOM windows and chunks, and a re-detected object whose
track ended starts a new masklet; the `tracking` component re-links them (and
across cameras) via the DINOv2 embeddings.

SAM2 video predictor API (sam2 package):
    state = predictor.init_state(video_path=<jpeg_dir>, offload_video_to_cpu=True,
                                 async_loading_frames=True)
    predictor.add_new_points_or_box(inference_state=state, frame_idx=i,
                                    obj_id=oid, box=np.array([x1,y1,x2,y2]))
    # the same call on a live obj_id re-prompts it from frame i on
    for f_idx, obj_ids, mask_logits in predictor.propagate_in_video(
            state, start_frame_idx=i, max_frame_num_to_track=n):
        ...   # mask_logits: (num_objs, 1, H, W); obj_ids parallel
    predictor.remove_object(state, obj_id, need_output=False)
    predictor.reset_state(state)
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import tempfile
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

import numpy as np
from scipy import ndimage

from wato_common.artifacts import masklets
from wato_perception_2d.fusion.masklet import Masklet
from wato_perception_2d.io import CameraFrameInfo
from wato_perception_2d.models.detector import Concept, Detection
from wato_perception_2d.models.embeddings import extract_dino_features

log = logging.getLogger(__name__)


class _Track:
    """Accumulator for one tracked obj_id across frames."""

    __slots__ = (
        "masklet_id",
        "cls",
        "score",
        "frames_present",
        "mask_writes",
        "dino_accum",
        "held",
    )

    def __init__(self, masklet_id: str, cls: str, score: float) -> None:
        self.masklet_id = masklet_id
        self.cls = cls
        self.score = score
        self.frames_present: list[int] = []
        # Pending PNG writes, parallel to frames_present; each resolves to the
        # mask's path.
        self.mask_writes: list[Future] = []
        self.dino_accum: list[np.ndarray] = []
        # A tentative track's masks, (camera_seq, packed bits, shape), written
        # once it is confirmed.
        self.held: list[tuple[int, np.ndarray, tuple[int, int]]] = []


def _to_numpy(x):
    """Bulk-transfer a tensor to host once; pass through arrays / lists."""
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _is_cuda_oom(exc: BaseException) -> bool:
    """True for a CUDA out-of-memory failure (retry in smaller windows)."""
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    return "out of memory" in str(exc).lower()


def _empty_cuda_cache() -> None:
    """gc + empty_cache so a window's freed allocations return to the driver."""
    try:
        import gc

        import torch

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _mask_bbox(mask: np.ndarray) -> Optional[tuple[float, float, float, float]]:
    """Tight (x1, y1, x2, y2) box of a boolean mask, or None when empty."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return (float(c0), float(r0), float(c1 + 1), float(r1 + 1))


def _box_iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0.0 else 0.0


def _matches(
    det: Detection,
    boxes: dict[int, tuple[float, float, float, float]],
    iou_threshold: float,
) -> list[int]:
    """Tracks whose mask bbox this frame a detection box matches (box IoU >=
    ``iou_threshold``): the detector still sees each of them, so the box
    supports them rather than starting a new track."""
    return [
        oid for oid, b in boxes.items() if _box_iou(det.box_xyxy, b) >= iou_threshold
    ]


def keyframes(n_frames: int, every_k: int) -> list[int]:
    """Frame indices where the detector runs: [0, k, 2k, …] within [0, n)."""
    if n_frames <= 0:
        return []
    step = max(1, every_k)
    return list(range(0, n_frames, step))


def _frame_masks(foreground, hw: tuple[int, int]) -> list[Optional[np.ndarray]]:
    """Per-object foreground for one frame → one boolean (H, W) mask, or None.

    ``foreground`` is computed on the logits' own device (e.g. ``logits > 0``)
    and moved to host here in one transfer (1 byte/pixel instead of 4, one sync
    instead of one per object). None marks an empty mask (object absent in this
    frame — normal) so the track simply skips the frame.
    """
    m = _to_numpy(foreground)
    if m.ndim == 4:  # (N, 1, H, W)
        m = m[:, 0]
    out: list[Optional[np.ndarray]] = []
    for mi in m:
        if mi.shape != hw:
            from PIL import Image as PILImage

            H, W = hw
            mi = np.asarray(
                PILImage.fromarray(mi.astype(np.uint8)).resize(
                    (W, H), PILImage.NEAREST
                ),
                dtype=bool,
            )
        out.append(mi if mi.any() else None)
    return out


_EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


def _drop_islands(mask: Optional[np.ndarray], fraction: float) -> Optional[np.ndarray]:
    """``mask`` without the specks around its body, or None when nothing is left.

    SAM2 scatters low-confidence specks around a mask (and ownership leaves the
    loser of a contested region a scatter of pixels); lifted to 3D, each speck
    labels LiDAR points of whatever lies behind it. A component (8-connected)
    smaller than ``fraction`` of the mask's largest one is dropped; the largest
    always stays, so small, distant objects survive. 0 disables.
    """
    if mask is None or not mask.any():
        return None
    if fraction <= 0.0:
        return mask
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    labels, n = ndimage.label(mask[r0:r1, c0:c1], structure=_EIGHT_CONNECTED)
    if n < 2:
        return mask
    areas = np.bincount(labels.ravel())[1:]
    keep = np.flatnonzero(areas >= fraction * areas.max()) + 1
    out = np.zeros_like(mask)
    out[r0:r1, c0:c1] = np.isin(labels, keep)
    return out


def _inside_fraction(mask: np.ndarray, region: np.ndarray) -> float:
    """Share of ``mask``'s pixels that fall inside ``region``."""
    return np.count_nonzero(mask & region) / max(1, np.count_nonzero(mask))


def _box_inside_fraction(
    box: tuple[float, float, float, float], region: np.ndarray
) -> float:
    """Share of a box's pixels that fall inside ``region``."""
    H, W = region.shape
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, x2 = max(0, x1), min(W, x2)
    y1, y2 = max(0, y1), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(region[y1:y2, x1:x2].mean())


def _logits_at(mask_logits, rows: list[int], pixels: np.ndarray) -> np.ndarray:
    """(len(rows), len(pixels)) float32 logits of those objects at those flat
    pixel indices — one small transfer instead of whole logit maps."""
    if hasattr(mask_logits, "dim"):  # torch
        import torch

        sub = mask_logits[rows].reshape(len(rows), -1)
        idx = torch.as_tensor(pixels, device=sub.device)
        return sub[:, idx].float().cpu().numpy()
    sub = np.asarray(mask_logits)[rows].reshape(len(rows), -1)
    return sub[:, pixels].astype(np.float32)


def _contained(
    present: list[tuple[int, np.ndarray]],
    obj_cls: dict[int, str],
    *,
    contained_classes: frozenset[str],
    container_classes: frozenset[str],
    fraction: float,
) -> set[int]:
    """Tracks sitting inside a container-class track (a vehicle) this frame.

    SAM2 segments what is seen through a car's windows as objects of their own:
    the driver as a pedestrian, the cabin as another car. Physically these lie
    inside the vehicle's mask (it covers its windows) and end well above its
    bottom, while an object in front of the vehicle stands on the road nearer
    the camera, so its mask reaches the vehicle's bottom or below. A
    ``contained_classes`` mask at least ``fraction`` inside a
    ``container_classes`` mask, ending above that mask's bottom by a tenth of
    its height, is part of the vehicle. Pure function of this frame's masks.
    """
    contained: set[int] = set()
    containers = [
        (oid, m) for oid, m in present if obj_cls.get(oid) in container_classes
    ]
    if not containers:
        return contained
    spans = {}
    for oid, m in containers:
        rows = np.flatnonzero(m.any(axis=1))
        spans[oid] = (rows[0], rows[-1])
    for oid, m in present:
        if obj_cls.get(oid) not in contained_classes:
            continue
        bottom = np.flatnonzero(m.any(axis=1))[-1]
        for cid, cm in containers:
            top, c_bottom = spans[cid]
            if (
                cid != oid
                and bottom < c_bottom - 0.1 * (c_bottom - top)
                and _inside_fraction(m, cm) >= fraction
            ):
                contained.add(oid)
                break
    return contained


def _owned_masks(
    mask_logits,
    obj_ids: list[int],
    present: list[tuple[int, np.ndarray]],
    *,
    island_fraction: float,
    collapse_fraction: float,
) -> tuple[list[tuple[int, np.ndarray]], list[int]]:
    """Give each pixel claimed by several tracks to one; return what each owns
    and the tracks that owned too little of their own mask to be recorded.

    SAM2 infers each object independently, so masks overlap — a bus's mask can
    wrap around a car half-hidden beside it — and an overlapping pixel would be
    lifted to 3D as both objects. Among the tracks still recorded this frame
    (ended ones are already out, so they can't take pixels), a contested pixel
    goes to the highest logit. Where two logits are close the winner flips from
    pixel to pixel, leaving the loser a scatter of specks over the winner; those
    (``_drop_islands``) go to the runner-up claimant instead of punching holes.

    A track left owning under ``collapse_fraction`` of its own mask is returned
    as collapsed and not recorded this frame; the caller ends it once that has
    lasted a few frames. Such a track has latched onto part of an object
    another track holds more surely, too small a part for its IoU with that
    track's mask to reach ``duplicate_mask_iou``. A track left with no pixels
    otherwise just skips the frame.
    """
    if len(present) < 2:
        return present, []
    n = len(present)
    shape = present[0][1].shape
    claims = np.stack([m for _, m in present]).reshape(n, -1)
    contested = np.flatnonzero(np.count_nonzero(claims, axis=0) > 1)
    if contested.size == 0:
        return present, []
    rows = [obj_ids.index(oid) for oid, _ in present]
    logits = np.where(
        claims[:, contested], _logits_at(mask_logits, rows, contested), -np.inf
    )
    rank = np.argsort(-logits, axis=0, kind="stable")
    cols = np.arange(contested.size)
    owned = claims.copy()
    owned[:, contested] = False
    owned[rank[0], contested] = True
    runner = np.full(claims.shape[1], -1)
    has_runner = np.isfinite(logits[rank[1], cols])
    runner[contested[has_runner]] = rank[1][has_runner]

    kept = np.zeros_like(owned)
    for i in range(n):
        m = _drop_islands(owned[i].reshape(shape), island_fraction)
        if m is not None:
            kept[i] = m.reshape(-1)
    lost_px = np.nonzero(owned & ~kept)[1]
    back = runner[lost_px] >= 0
    kept[runner[lost_px[back]], lost_px[back]] = True

    out: list[tuple[int, np.ndarray]] = []
    collapsed: list[int] = []
    for i, (oid, _) in enumerate(present):
        area = np.count_nonzero(kept[i])
        if area < collapse_fraction * np.count_nonzero(claims[i]):
            collapsed.append(oid)
        elif area:
            out.append((oid, kept[i].reshape(shape)))
    return out, collapsed


_JPEG_EXTS = (".jpg", ".jpeg")


def _prepare_frame_dir(frames: list[CameraFrameInfo], images: list[np.ndarray]) -> str:
    """A temp dir of sequential ``00000000.jpg`` frames for SAM2's video loader.

    SAM2's ``init_state`` reads a directory of JPEGs whose names are integers (it
    orders frames by ``int(filename)``), so names are always a clean sequence —
    robust to gaps left by frame-index filtering. A frame ingest already stored
    as JPEG (CompressedImage bytes, written verbatim) is symlinked rather than
    decoded and re-encoded: no CPU cost and no second lossy pass. Anything else
    (PNG from a raw Image topic) is encoded at quality 95 on a thread pool.
    Caller removes the dir afterwards.
    """
    from PIL import Image as PILImage

    tmp = tempfile.mkdtemp(prefix="sam2_frames_")

    def _encode(i: int, img: np.ndarray) -> None:
        PILImage.fromarray(img).save(os.path.join(tmp, f"{i:08d}.jpg"), quality=95)

    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = []
        for i, (frame, img) in enumerate(zip(frames, images)):
            src = frame.image_path
            if src.lower().endswith(_JPEG_EXTS) and os.path.isfile(src):
                os.symlink(os.path.abspath(src), os.path.join(tmp, f"{i:08d}.jpg"))
            else:
                jobs.append(pool.submit(_encode, i, img))
        for j in jobs:
            j.result()
    return tmp


def _autocast(device: str, dtype_name: str):
    """Context factory for SAM2 calls: CUDA autocast at ``dtype_name``, or a
    no-op on CPU / when ``dtype_name`` is "float32"."""
    if dtype_name == "float32" or not str(device).startswith("cuda"):
        return contextlib.nullcontext
    import torch

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    return lambda: torch.autocast("cuda", dtype=dtype)


def _iter_in(ctx_factory, gen):
    """Drive ``gen`` with ``ctx_factory()`` active only while it computes.

    SAM2's ``propagate_in_video`` is a generator: each ``next`` does a frame's
    work. Entering the context per step (rather than around the consumer's
    loop) keeps the loop body — the DINOv2 forward — out of it.
    """
    try:
        while True:
            with ctx_factory():
                try:
                    item = next(gen)
                except StopIteration:
                    return
            yield item
    finally:
        gen.close()


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero(a & b)
    return inter / np.count_nonzero(a | b) if inter else 0.0


class _EndedTracks:
    """Decides, frame by frame, which SAM2 masks still belong to their track.

    SAM2 does not let go of an object it has lost — it left the image, or was
    covered — and within 0–30 frames tends to latch onto a similar object
    nearby, usually one that already has its own track (per-object inference
    has no notion of two objects owning the same pixels). On WATO data every
    such resume was a different object. So a track ends, permanently:

    * when its object has had no mask for ``drop_lost_after_frames``
      consecutive frames (the resume after a gap); or
    * when its mask and another live track's reach mask IoU >=
      ``duplicate_mask_iou`` in the same frame (the jump with no gap, e.g. an
      oncoming car leaving the image as it passes a parked one). Of the two,
      the track whose mask changed most since its previous mask is the one
      that jumped and ends; a track with no previous mask (just prompted)
      counts as having jumped, so an existing track is never displaced.

    The frame loop also ends a track (``end``) whose mask lies mostly on the
    ego vehicle's body, and (``strike``) one that for several frames in a row
    owns little of its own mask once contested pixels are shared out
    (``_owned_masks``) or sits inside a vehicle as part of it
    (``_contained``). An ended track's masks are never recorded again; the
    caller removes it from the SAM2 session at the next keyframe.
    """

    def __init__(self) -> None:
        self.ids: set[int] = set()
        self._empty_run: dict[int, int] = {}
        self._prev: dict[int, np.ndarray] = {}
        self._strikes: dict[str, dict[int, int]] = {}

    def strike(
        self, rule: str, hit: set[int], seen: list[int], *, after_frames: int
    ) -> None:
        """Count consecutive frames each track in ``seen`` met ``rule`` (the ones
        in ``hit``); end a track once it has for ``after_frames`` frames."""
        runs = self._strikes.setdefault(rule, {})
        for oid in seen:
            if oid in hit:
                runs[oid] = runs.get(oid, 0) + 1
                if runs[oid] >= after_frames:
                    self.end(oid)
            else:
                runs.pop(oid, None)

    def claim(
        self,
        obj_ids: list[int],
        masks: list[Optional[np.ndarray]],
        *,
        drop_lost_after_frames: int,
        duplicate_mask_iou: float,
    ) -> list[tuple[int, np.ndarray]]:
        """This frame's (obj_id, mask) pairs that still belong to their track."""
        present: list[tuple[int, np.ndarray]] = []
        for oid, mask in zip(obj_ids, masks):
            if oid in self.ids:
                continue
            if mask is None:
                run = self._empty_run[oid] = self._empty_run.get(oid, 0) + 1
                if 0 < drop_lost_after_frames <= run:
                    self.end(oid)
                continue
            self._empty_run[oid] = 0
            present.append((oid, mask))

        if 0.0 < duplicate_mask_iou <= 1.0:
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    (a, ma), (b, mb) = present[i], present[j]
                    if a in self.ids or b in self.ids:
                        continue
                    if _mask_iou(ma, mb) >= duplicate_mask_iou:
                        self.end(
                            a
                            if self._steadiness(a, ma) < self._steadiness(b, mb)
                            else b
                        )

        present = [(oid, m) for oid, m in present if oid not in self.ids]
        for oid, m in present:
            self._prev[oid] = m
        return present

    def _steadiness(self, oid: int, mask: np.ndarray) -> float:
        prev = self._prev.get(oid)
        return -1.0 if prev is None else _mask_iou(mask, prev)

    def end(self, oid: int) -> None:
        self.ids.add(oid)
        self._prev.pop(oid, None)


class _Support:
    """Detector support for each track over keyframes.

    SAM2 alone keeps following whatever its mask has become (the sky, the
    road, a smear over several objects). A detection box matching a track's
    mask bbox (``_matches``) supports it at that keyframe, so:

    * a confirmed track's keep-alive rises by one at each keyframe a detection
      supports it, up to ``support_keyframes``, and falls by one at each
      keyframe its mask's bbox covers at least ``support_min_area`` of the
      image but no detection supports it; at zero the track ends. 0 disables.
      Only large masks count: the detector reliably boxes what looks large,
      but boxes a distant car rarely, though SAM2 tracks it well;
    * a detection scoring at least ``birth_score`` starts a track; a weaker one
      starts it tentatively, its masks held back, until detections support it
      at ``confirm_keyframes`` keyframes in a row (its own counts); a keyframe
      without support drops it, held masks and all. This needs a detector
      that re-finds what it found; COCO YOLO doesn't re-find distant cars, so
      the default ``birth_score`` 0 confirms every track;
    * its class is the one its supporting detections' scores sum highest for,
      the seed's on a tie (COCO calls the campus shuttle ``truck`` at one
      keyframe and ``bus`` at the next); its score is their mean.
    """

    def __init__(
        self,
        *,
        birth_score: float,
        confirm_keyframes: int,
        support_keyframes: int,
    ) -> None:
        self.birth_score = birth_score
        self.confirm_keyframes = confirm_keyframes
        self.support_keyframes = support_keyframes
        self.tentative: dict[int, int] = {}  # obj_id → keyframes supported so far
        self._alive: dict[int, int] = {}
        self._votes: dict[int, dict[str, list[float]]] = {}

    def born(self, oid: int, det: Detection) -> None:
        self._votes[oid] = {det.cls: [det.score]}
        self._alive[oid] = self.support_keyframes
        if det.score < self.birth_score and self.confirm_keyframes > 1:
            self.tentative[oid] = 1

    def keyframe(
        self,
        tracked: list[int],
        support: dict[int, list[Detection]],
        large: set[int],
    ) -> tuple[list[int], list[int], list[int]]:
        """Count one keyframe's support for the tracks with a mask there
        (``large``: those whose mask is large enough to need it); return the
        tracks (confirmed now, dropped, ended)."""
        confirmed: list[int] = []
        dropped: list[int] = []
        ended: list[int] = []
        for oid in tracked:
            dets = support.get(oid, [])
            for d in dets:
                self._votes[oid].setdefault(d.cls, []).append(d.score)
            if oid in self.tentative:
                if not dets:
                    del self.tentative[oid]
                    dropped.append(oid)
                    continue
                self.tentative[oid] += 1
                if self.tentative[oid] >= self.confirm_keyframes:
                    del self.tentative[oid]
                    confirmed.append(oid)
                continue
            if self.support_keyframes <= 0 or not (dets or oid in large):
                continue
            alive = self._alive[oid] + (1 if dets else -1)
            self._alive[oid] = min(alive, self.support_keyframes)
            if alive <= 0:
                ended.append(oid)
        return confirmed, dropped, ended

    def label(self, oid: int) -> tuple[str, float]:
        votes = self._votes[oid]
        cls = max(votes, key=lambda c: sum(votes[c]))
        return cls, float(np.mean(votes[cls]))


def track_camera(
    predictor,
    detector,
    frames: list[CameraFrameInfo],
    images: list[np.ndarray],
    concepts: list[Concept],
    *,
    bag_id: str,
    chunk_id: str,
    cam_id: str,
    dino_model: str = "dinov2_vitl14",
    dino_every_k: int = 5,
    device: str = "cuda",
    redetect_every_k: int = 10,
    iou_match_threshold: float = 0.5,
    offload_video_to_cpu: bool = True,
    sub_clip_frames: int = 150,
    min_sub_clip_frames: int = 16,
    drop_lost_after_frames: int = 5,
    duplicate_mask_iou: float = 0.7,
    autocast_dtype: str = "bfloat16",
    ego_mask: Optional[np.ndarray] = None,
    ego_max_fraction: float = 0.5,
    island_fraction: float = 0.1,
    collapse_fraction: float = 0.05,
    collapse_after_frames: int = 3,
    contained_classes: tuple[str, ...] = (),
    container_classes: tuple[str, ...] = (),
    containment_fraction: float = 0.9,
    containment_after_frames: int = 3,
    birth_score: float = 0.0,
    confirm_keyframes: int = 3,
    support_keyframes: int = 5,
    support_min_area: float = 0.01,
    reprompt_score: float = 0.8,
    reprompt_below_iou: float = 0.7,
) -> list[Masklet]:
    """Detect + SAM2-track every taxonomy object across one camera's frames.

    Tries the whole clip in one SAM2 session. On a CUDA OOM it falls back to
    windows of ``sub_clip_frames`` (fresh session each); a window that itself
    OOMs is recursively halved down to ``min_sub_clip_frames`` before being
    skipped. Object ids reset between windows (re-linked downstream via DINOv2).

    ``ego_mask`` is the (H, W) region of the image showing the vehicle's own
    body (None: the camera doesn't see it). A detection box, or a track's mask,
    at least ``ego_max_fraction`` inside it is the vehicle itself and never
    becomes or stays a track, and its pixels are cleared from every mask.

    ``birth_score`` / ``confirm_keyframes`` / ``support_keyframes`` /
    ``support_min_area`` are ``_Support``'s. A confirmed track supported at a
    keyframe by exactly one box, scoring at least ``reprompt_score`` and
    supporting no other track, but whose mask bbox has drifted below box IoU
    ``reprompt_below_iou`` with it, is re-prompted with that box there. 0
    disables.

    Returns one Masklet per confirmed track that appeared in ≥1 frame.
    """
    if not frames or not images:
        return []

    kw = dict(
        bag_id=bag_id,
        chunk_id=chunk_id,
        cam_id=cam_id,
        dino_model=dino_model,
        dino_every_k=dino_every_k,
        device=device,
        redetect_every_k=redetect_every_k,
        iou_match_threshold=iou_match_threshold,
        offload_video_to_cpu=offload_video_to_cpu,
        drop_lost_after_frames=drop_lost_after_frames,
        duplicate_mask_iou=duplicate_mask_iou,
        autocast_dtype=autocast_dtype,
        ego_mask=ego_mask,
        ego_max_fraction=ego_max_fraction,
        island_fraction=island_fraction,
        collapse_fraction=collapse_fraction,
        collapse_after_frames=collapse_after_frames,
        contained_classes=frozenset(contained_classes),
        container_classes=frozenset(container_classes),
        containment_fraction=containment_fraction,
        containment_after_frames=containment_after_frames,
        birth_score=birth_score,
        confirm_keyframes=confirm_keyframes,
        support_keyframes=support_keyframes,
        support_min_area=support_min_area,
        reprompt_score=reprompt_score,
        reprompt_below_iou=reprompt_below_iou,
    )

    # Attempt 1: the whole clip in one session.
    try:
        tracks = _track_window(
            predictor, detector, frames, images, concepts, frame_offset=0, **kw
        )
        return _tracks_to_masklets(tracks, bag_id, chunk_id, cam_id)
    except Exception as exc:  # noqa: BLE001
        if not _is_cuda_oom(exc):
            raise
        window = sub_clip_frames if sub_clip_frames and sub_clip_frames > 0 else 150
        floor = max(1, min(min_sub_clip_frames, window))
        log.warning(
            "chunk %s / %s: OOM on full %d-frame clip (%s) — retrying in "
            "%d-frame sub-clips (halving to a %d-frame floor on further OOM).",
            chunk_id,
            cam_id,
            len(frames),
            exc,
            window,
            floor,
        )
        _empty_cuda_cache()

    # Attempt 2: windowed, fresh session each, recursively halving on repeat OOM.
    def _track_range(start: int, stop: int) -> list[Masklet]:
        fseg = frames[start:stop]
        iseg = images[start:stop]
        try:
            tr = _track_window(
                predictor, detector, fseg, iseg, concepts, frame_offset=start, **kw
            )
        except Exception as exc:  # noqa: BLE001
            if not _is_cuda_oom(exc):
                raise
            _empty_cuda_cache()
            n = stop - start
            if n <= floor:
                log.warning(
                    "chunk %s / %s: sub-clip [%d:%d] still OOM at the %d-frame "
                    "floor (%s) — skipping window.",
                    chunk_id,
                    cam_id,
                    start,
                    stop,
                    floor,
                    exc,
                )
                return []
            mid = start + n // 2
            log.warning(
                "chunk %s / %s: sub-clip [%d:%d] OOM (%s) — halving into "
                "[%d:%d] + [%d:%d].",
                chunk_id,
                cam_id,
                start,
                stop,
                exc,
                start,
                mid,
                mid,
                stop,
            )
            return _track_range(start, mid) + _track_range(mid, stop)
        out = _tracks_to_masklets(tr, bag_id, chunk_id, cam_id)
        _empty_cuda_cache()
        return out

    masklets: list[Masklet] = []
    for start in range(0, len(frames), window):
        masklets.extend(_track_range(start, min(start + window, len(frames))))
    return masklets


def _track_window(
    predictor,
    detector,
    frames: list[CameraFrameInfo],
    images: list[np.ndarray],
    concepts: list[Concept],
    *,
    frame_offset: int,
    bag_id: str,
    chunk_id: str,
    cam_id: str,
    dino_model: str,
    dino_every_k: int,
    device: str,
    redetect_every_k: int,
    iou_match_threshold: float,
    offload_video_to_cpu: bool,
    drop_lost_after_frames: int,
    duplicate_mask_iou: float,
    autocast_dtype: str,
    ego_mask: Optional[np.ndarray],
    ego_max_fraction: float,
    island_fraction: float,
    collapse_fraction: float,
    collapse_after_frames: int,
    contained_classes: frozenset[str],
    container_classes: frozenset[str],
    containment_fraction: float,
    containment_after_frames: int,
    birth_score: float,
    confirm_keyframes: int,
    support_keyframes: int,
    support_min_area: float,
    reprompt_score: float,
    reprompt_below_iou: float,
) -> dict[int, _Track]:
    """Detect + propagate over one (windowed) frame range in a single session.

    Every failure propagates: a CUDA OOM so the caller can retry in smaller
    windows, anything else (a missing detector or DINOv2 checkpoint, a failed
    mask write) to stop the run, since returning the tracks so far would write
    a camera with no or partial masklets that looks complete. New
    objects are introduced segment-by-segment between keyframes; ``frame_offset``
    is the window's start index in the full clip, used for the DINOv2 sampling
    cadence and log lines (mask filenames use the absolute camera_seq).
    """
    n = len(frames)
    if n == 0:
        return {}
    seq_by_index = [f.camera_seq for f in frames]
    H, W = images[0].shape[:2]
    if ego_mask is not None and ego_mask.shape != (H, W):
        raise ValueError(
            f"{cam_id}: ego mask is {ego_mask.shape[1]}x{ego_mask.shape[0]} but "
            f"the images are {W}x{H}"
        )
    amp = _autocast(device, autocast_dtype)

    tracks: dict[int, _Track] = {}
    obj_meta: dict[int, tuple[str, float]] = {}  # obj_id → (class, det_score)
    obj_cls: dict[int, str] = {}  # obj_id → class (the vote so far)
    seed_box: dict[int, tuple[float, float, float, float]] = {}
    support = _Support(
        birth_score=birth_score,
        confirm_keyframes=confirm_keyframes,
        support_keyframes=support_keyframes,
    )
    # Boxes whose object SAM2 segmented as the vehicle's own body: the detector
    # finds the same box at every keyframe, so they aren't prompted again.
    ego_boxes: list[tuple[float, float, float, float]] = []
    kfs = keyframes(n, redetect_every_k)
    kf_set = set(kfs)
    # Masks propagated into upcoming keyframes, for matching re-detections. Only
    # keyframes are kept: holding every frame's masks costs GBs on long clips.
    masks_at_kf: dict[int, dict[int, np.ndarray]] = {}
    live: set[int] = set()  # obj_ids currently in the SAM2 session
    ended = _EndedTracks()  # lost / jumped objects: never recorded again
    next_obj_id = 0

    tmpdir: Optional[str] = None
    state = None
    writer = ThreadPoolExecutor(max_workers=4, thread_name_prefix="mask_png")
    try:
        tmpdir = _prepare_frame_dir(frames, images)
        with amp():
            state = predictor.init_state(
                video_path=tmpdir,
                offload_video_to_cpu=offload_video_to_cpu,
                async_loading_frames=True,
            )

        for ki, kf in enumerate(kfs):
            is_last = ki == len(kfs) - 1
            stop = (n - 1) if is_last else kfs[ki + 1]  # last frame idx to reach
            record_stop = n if is_last else kfs[ki + 1]  # exclusive record bound

            # This keyframe's boxes, highest score first; a box mostly on the
            # vehicle's own body is the vehicle. One matching a track's mask
            # (propagated into this frame) supports it; the rest are births.
            existing = masks_at_kf.pop(kf, {})
            boxes = {oid: _mask_bbox(m) for oid, m in existing.items()}
            backing: dict[int, list[Detection]] = {}
            births: list[Detection] = []
            shared_box: set[int] = set()  # id() of boxes matching several tracks
            for det in sorted(
                detector.detect(images[kf], concepts), key=lambda d: -d.score
            ):
                if ego_mask is not None and (
                    _box_inside_fraction(det.box_xyxy, ego_mask) >= ego_max_fraction
                    or any(
                        _box_iou(det.box_xyxy, b) >= iou_match_threshold
                        for b in ego_boxes
                    )
                ):
                    continue
                matched = _matches(det, boxes, iou_match_threshold)
                if len(matched) > 1:
                    shared_box.add(id(det))
                for oid in matched:
                    backing.setdefault(oid, []).append(det)
                if not matched:
                    births.append(det)

            large = {
                oid
                for oid, (x1, y1, x2, y2) in boxes.items()
                if (x2 - x1) * (y2 - y1) >= support_min_area * H * W
            }
            confirmed, dropped, unsupported = support.keyframe(
                list(existing), backing, large
            )
            for oid in confirmed:
                if oid in tracks:
                    _flush_held(tracks[oid], writer, bag_id, chunk_id)
            for oid in dropped:
                ended.end(oid)
                tracks.pop(oid, None)
            for oid in unsupported:
                ended.end(oid)
            for oid in backing:
                obj_cls[oid] = support.label(oid)[0]

            # A confident box backing one track alone, whose mask has drifted
            # from it, re-prompts that track here.
            for oid, dets in backing.items():
                det = dets[0]
                if (
                    len(dets) == 1
                    and oid not in ended.ids
                    and oid not in support.tentative
                    and id(det) not in shared_box
                    and det.score >= reprompt_score
                    and _box_iou(det.box_xyxy, boxes[oid]) < reprompt_below_iou
                ):
                    with amp():
                        _add_object(predictor, state, kf, oid, det, new_segment=True)

            # Tracks that ended since the last keyframe (or just now) stopped
            # recording at once; remove them from the session here (between
            # propagations) so SAM2 stops paying for them.
            for oid in sorted(ended.ids & live):
                with amp():
                    predictor.remove_object(state, oid, need_output=False)
                live.discard(oid)

            # Introduce the boxes no track accounts for, less those repeating a
            # box accepted earlier at this keyframe — the detector's NMS is per
            # class, so one shuttle can come back as both "truck" and "bus";
            # highest score first, so the better label wins.
            accepted: list[tuple[float, float, float, float]] = []
            for det in births:
                if any(
                    _box_iou(det.box_xyxy, b) >= iou_match_threshold for b in accepted
                ):
                    continue
                accepted.append(det.box_xyxy)
                oid = next_obj_id
                next_obj_id += 1
                obj_meta[oid] = (det.cls, det.score)
                obj_cls[oid] = det.cls
                seed_box[oid] = det.box_xyxy
                with amp():
                    added = _add_object(
                        predictor, state, kf, oid, det, new_segment=bool(live)
                    )
                if not added:
                    # Mid-stream add unsupported on this build → stop introducing
                    # new objects; existing ones still propagate to the clip end.
                    del obj_meta[oid]
                    break
                live.add(oid)
                support.born(oid, det)

            # Nothing to track in this segment (e.g. an empty road at the first
            # keyframe): SAM2 refuses to propagate zero objects, so skip ahead to
            # the next keyframe's detections instead of abandoning the window.
            if not live:
                continue

            # Propagate this segment; record [kf, record_stop), keep the masks at
            # the next keyframe (which the next segment records as its own
            # keyframe) for its re-detection match.
            for f_idx, obj_ids, mask_logits in _iter_in(
                amp,
                predictor.propagate_in_video(
                    state, start_frame_idx=kf, max_frame_num_to_track=stop - kf
                ),
            ):
                f = int(f_idx)
                ids = [int(o) for o in obj_ids]
                masks = _frame_masks(mask_logits > 0, (H, W))
                # A box the ego check let through can still be segmented as the
                # vehicle's body (a box over the rack and the road beside it),
                # and a mask can drift onto it: such a track ends. Every other
                # mask loses the body's pixels.
                if ego_mask is not None:
                    for oid, m in zip(ids, masks):
                        if (
                            m is not None
                            and oid not in ended.ids
                            and _inside_fraction(m, ego_mask) >= ego_max_fraction
                        ):
                            ended.end(oid)
                            if oid in seed_box:
                                ego_boxes.append(seed_box[oid])
                    masks = [None if m is None else m & ~ego_mask for m in masks]
                masks = [_drop_islands(m, island_fraction) for m in masks]
                # Decide which tracks end on SAM2's raw masks (a jump shows as
                # two tracks on one object), then split contested pixels among
                # the survivors only.
                present = ended.claim(
                    ids,
                    masks,
                    drop_lost_after_frames=drop_lost_after_frames,
                    duplicate_mask_iou=duplicate_mask_iou,
                )
                # What is seen through a vehicle's windows (the driver, the
                # cabin) is part of the vehicle: not recorded, its pixels left
                # to the vehicle, ended if it stays.
                inside = _contained(
                    present,
                    obj_cls,
                    contained_classes=contained_classes,
                    container_classes=container_classes,
                    fraction=containment_fraction,
                )
                ended.strike(
                    "contained",
                    inside,
                    [oid for oid, _ in present],
                    after_frames=containment_after_frames,
                )
                present = [(oid, m) for oid, m in present if oid not in inside]
                shared = [oid for oid, _ in present]
                present, collapsed = _owned_masks(
                    mask_logits,
                    ids,
                    present,
                    island_fraction=island_fraction,
                    collapse_fraction=collapse_fraction,
                )
                ended.strike(
                    "collapse",
                    set(collapsed),
                    shared,
                    after_frames=collapse_after_frames,
                )
                if f in kf_set and f > kf:
                    masks_at_kf[f] = dict(present)
                if f < record_stop and present:
                    _record_frame(
                        tracks,
                        obj_meta,
                        present,
                        support.tentative,
                        frame_offset + f,
                        seq_by_index[f],
                        images[f],
                        writer,
                        bag_id,
                        chunk_id,
                        dino_model,
                        dino_every_k,
                        device,
                    )
                if f >= stop:
                    break
        # A track still tentative at the end has only held masks, so it is
        # never written (_tracks_to_masklets skips it).
        for oid, track in tracks.items():
            track.cls, track.score = support.label(oid)
    finally:
        # Let queued mask PNGs land before the tracks are handed back; a failed
        # write surfaces from Future.result() in _track_to_masklet.
        writer.shutdown(wait=True)
        if state is not None:
            try:
                predictor.reset_state(state)
            except Exception:  # noqa: BLE001
                pass
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return tracks


# A new object could not be added mid-stream — warn once per process, not once
# per keyframe across every camera.
_warned_midstream_add = False


def _add_object(
    predictor, state, frame_idx: int, obj_id: int, det: Detection, *, new_segment: bool
) -> bool:
    """Prompt SAM2 with one detection box; return False if a mid-stream add fails.

    Adding to a session with nothing live (``new_segment=False``) must succeed
    (the error propagates). Adding alongside live objects, some SAM2 builds
    reject a new object after propagation has started; we isolate that (warn
    once, return False) so existing tracks still finish — graceful degradation
    of the re-detection feature, not garbage output.
    """
    global _warned_midstream_add
    box = np.asarray(det.box_xyxy, dtype=np.float32)
    try:
        predictor.add_new_points_or_box(
            inference_state=state, frame_idx=frame_idx, obj_id=obj_id, box=box
        )
        return True
    except Exception as exc:  # noqa: BLE001
        if not new_segment:
            raise  # nothing live to fall back on — a real error
        if not _warned_midstream_add:
            _warned_midstream_add = True
            log.warning(
                "SAM2: could not add a new object mid-stream (%s) — new objects "
                "entering after a clip's first keyframe will be missed until the "
                "next OOM window / chunk. Lower nothing; this is build-dependent.",
                exc,
            )
        return False


def _record_frame(
    tracks: dict[int, _Track],
    obj_meta: dict[int, tuple[str, float]],
    present: list[tuple[int, np.ndarray]],
    tentative: dict[int, int],
    frame_index_abs: int,
    camera_seq: int,
    image_rgb: np.ndarray,
    writer: ThreadPoolExecutor,
    bag_id: str,
    chunk_id: str,
    dino_model: str,
    dino_every_k: int,
    device: str,
) -> None:
    """Append one frame's non-empty masks to their tracks.

    Mask PNGs are queued on ``writer`` (PIL releases the GIL while encoding, so
    they overlap the next frame's GPU work); a ``tentative`` track's are held
    until it is confirmed (``_flush_held``). On DINOv2 frames every present
    object is embedded in one batched forward.
    """
    for obj_id, mask in present:
        track = tracks.get(obj_id)
        if track is None:
            cls, score = obj_meta.get(obj_id, ("object", 0.0))
            track = _Track(masklet_id=str(uuid.uuid4())[:12], cls=cls, score=score)
            tracks[obj_id] = track
        if obj_id in tentative:
            track.held.append((camera_seq, np.packbits(mask), mask.shape))
            continue
        track.frames_present.append(camera_seq)
        track.mask_writes.append(
            writer.submit(
                masklets.write_mask,
                bag_id,
                chunk_id,
                track.masklet_id,
                camera_seq,
                mask,
            )
        )
    if dino_every_k > 0 and frame_index_abs % dino_every_k == 0:
        # A failure raises: masklets without embeddings can't be re-linked by
        # `tracking`, and a missing model would fail on every frame.
        feats = extract_dino_features(
            image_rgb, [m for _, m in present], dino_model, device
        )
        for (obj_id, _), feat in zip(present, feats):
            if feat is not None:
                tracks[obj_id].dino_accum.append(feat)


def _flush_held(
    track: _Track, writer: ThreadPoolExecutor, bag_id: str, chunk_id: str
) -> None:
    """Queue a just-confirmed track's held masks, oldest first (before any
    later frame is recorded, so frames_present stays in time order)."""
    for camera_seq, bits, shape in track.held:
        mask = np.unpackbits(bits, count=shape[0] * shape[1]).reshape(shape)
        track.frames_present.append(camera_seq)
        track.mask_writes.append(
            writer.submit(
                masklets.write_mask,
                bag_id,
                chunk_id,
                track.masklet_id,
                camera_seq,
                mask.astype(bool),
            )
        )
    track.held = []


def _tracks_to_masklets(
    tracks: dict[int, _Track], bag_id: str, chunk_id: str, cam_id: str
) -> list[Masklet]:
    return [
        _track_to_masklet(t, bag_id, chunk_id, cam_id)
        for t in tracks.values()
        if t.frames_present
    ]


def _track_to_masklet(
    track: _Track, bag_id: str, chunk_id: str, cam_id: str
) -> Masklet:
    dino = (
        np.mean(track.dino_accum, axis=0).astype(np.float32)
        if track.dino_accum
        else None
    )
    # Re-sort the (frames_present, mask_paths) pair by camera_seq so the masklet
    # is time-ordered regardless of yield order.
    order = sorted(
        range(len(track.frames_present)), key=track.frames_present.__getitem__
    )
    frames_present = [track.frames_present[i] for i in order]
    mask_paths = [track.mask_writes[i].result() for i in order]
    return Masklet(
        masklet_id=track.masklet_id,
        bag_id=bag_id,
        chunk_id=chunk_id,
        cam_id=cam_id,
        cls=track.cls,
        score=track.score,
        frames_present=frames_present,
        mask_paths=mask_paths,
        dino_feature=dino,
        tracker_backend="sam2",
    )
