"""Detector + SAM2-family video tracking (stock SAM2 or DAM4SAM memory).

For one camera stream, runs the 2D detector on keyframes and uses SAM2's video
predictor to segment + propagate each detected box into a tracked masklet:

  1. Point SAM2's video loader at the camera's frames: a temp dir of
     sequentially-named JPEGs — symlinks to ingest's own JPEGs when the source
     already is one, re-encoded otherwise — loaded asynchronously by SAM2.
  2. At each keyframe (every ``redetect_every_k`` frames) run the detector →
     class-labeled boxes, dropping those mostly on the ego vehicle. A box
     matching a tracked object (box IoU with its mask bbox ≥
     ``iou_match_threshold``) supports that track (``_Support``); ended tracks
     (below) leave the session. The other boxes, highest score first and less
     repeats of a higher-scoring box at the same keyframe, must pass the birth
     gate (``birth_gate``), which tests them against the masks of the live
     tracks and of the births already accepted at that keyframe: SAM2MOT's
     untracked area (default) or SAM 3's mask IoU. Those that pass get a fresh
     obj_id via ``add_new_points_or_box``; the rest are dropped.
  3. ``propagate_in_video`` forward to the next keyframe, recording per-object
     masks → ``_Track`` → ``Masklet`` (per-frame PNGs written on a thread pool,
     DINOv2 features every k frames batched over the frame's objects). Each
     mask loses the ego vehicle's pixels and its specks (``_drop_islands``).
     Independent masks may overlap: their competing pixel claims survive to
     lifting and geometric refinement. A track ends when SAM2 loses its object
     for ``drop_lost_after_frames`` frames, its mask lies on the ego vehicle,
     or detections stop supporting it. Overlap with another track does not end
     it or remove its claimed pixels.
  4. Back-fill (``backfill_frames``): after the forward pass, each track born
     after the window's first image is propagated backwards from its birth
     prompt (``_backfill``), adding the images before its first detection
     that no recorded track already covers.

Cost: SAM2 runs memory attention per *object* per frame and keeps every object
it was given, so without pruning cost and VRAM grow with every object ever
detected. Removing ended tracks at the next keyframe bounds that. SAM2 runs
under ``autocast_dtype`` (bfloat16 by default); the detector and DINOv2 stay
float32.

Object ids reset between OOM windows and chunks, and a re-detected object whose
track ended starts a new masklet. Cross-camera and cross-chunk identity belongs
to downstream components; these masklet IDs are local to this tracking episode.

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
from wato_common.artifacts.frame_index import CameraFrame
from wato_perception_2d.fusion.masklet import Masklet
from wato_perception_2d.config import MaskRefinementConfig
from wato_perception_2d.models.detector import Concept, Detection
from wato_perception_2d.models.embeddings import extract_dino_features
from wato_perception_2d.models.mask_refinement import (
    KeyframeMaskRefiner,
    stability_scores,
)

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

    SAM2 scatters low-confidence specks around a mask; lifted to 3D, each speck
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


def _untracked_share(
    box: tuple[float, float, float, float], tracked: Optional[np.ndarray]
) -> float:
    """Share of a box's pixels outside every tracked mask (``tracked``: their
    union, None for none): SAM2MOT's object-addition test."""
    if tracked is None:
        return 1.0
    return 1.0 - _box_inside_fraction(box, tracked)


def _prompt_mask(prompted, obj_id: int, hw: tuple[int, int]) -> Optional[np.ndarray]:
    """The prompted object's mask from a prompt call's ``(frame_idx, obj_ids,
    mask_logits)`` return (SAM2 returns every object's), or None if empty."""
    _, ids, logits = prompted
    i = [int(o) for o in ids].index(obj_id)
    return _frame_masks(logits[i : i + 1] > 0, hw)[0]


_JPEG_EXTS = (".jpg", ".jpeg")


def _prepare_frame_dir(frames: list[CameraFrame], images: list[np.ndarray]) -> str:
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
    """End tracks with absent masks, plus explicit ego/detector-support ends.

    Overlap is evidence for downstream 3D hypotheses, so another track's mask
    cannot end this track. Ended tracks are removed at the next keyframe.
    """

    def __init__(self) -> None:
        self.ids: set[int] = set()
        self._empty_run: dict[int, int] = {}

    def claim(
        self,
        obj_ids: list[int],
        masks: list[Optional[np.ndarray]],
        *,
        drop_lost_after_frames: int,
    ) -> list[tuple[int, np.ndarray]]:
        """Every nonempty independent mask from a track that has not ended."""
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
        return present

    def end(self, oid: int) -> None:
        self.ids.add(oid)


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
    frames: list[CameraFrame],
    images: list[np.ndarray],
    concepts: list[Concept],
    *,
    bag_id: str,
    chunk_id: str,
    cam_id: str,
    tracker_backend: str,
    dino_model: str = "dinov2_vitl14",
    dino_every_k: int = 5,
    device: str = "cuda",
    redetect_every_k: int = 10,
    iou_match_threshold: float = 0.5,
    offload_video_to_cpu: bool = True,
    sub_clip_frames: int = 150,
    min_sub_clip_frames: int = 16,
    drop_lost_after_frames: int = 5,
    autocast_dtype: str = "bfloat16",
    ego_mask: Optional[np.ndarray] = None,
    ego_max_fraction: float = 0.5,
    island_fraction: float = 0.1,
    birth_score: float = 0.0,
    confirm_keyframes: int = 3,
    support_keyframes: int = 5,
    support_min_area: float = 0.01,
    reprompt_score: float = 0.8,
    reprompt_below_iou: float = 0.7,
    birth_gate: str = "untracked_area",
    birth_untracked_fraction: float = 0.5,
    birth_max_mask_iou: float = 0.1,
    backfill_frames: int = 20,
    refinement: Optional[MaskRefinementConfig] = None,
    diagnostics: Optional[list[dict]] = None,
) -> list[Masklet]:
    """Detect + SAM2-track every taxonomy object across one camera's frames.

    Tries the whole clip in one SAM2 session. On a CUDA OOM it falls back to
    windows of ``sub_clip_frames`` (fresh session each); a window that itself
    OOMs is recursively halved down to ``min_sub_clip_frames`` before raising.
    Object ids reset between windows (``tracking`` associates 3D boxes, not
    masklet ids).

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

    A box no track accounts for starts one only if it passes ``birth_gate``:
    "untracked_area" (SAM2MOT: more than ``birth_untracked_fraction`` of the
    box outside every tracked mask), "mask_iou" (SAM 3: SAM2's image mask for
    the box below ``birth_max_mask_iou`` IoU with every tracked mask) or
    "none". ``backfill_frames`` (0 disables) bounds ``_backfill``, which
    extends each track born after its window's first image backwards.

    ``refinement`` optionally proposes image-predictor masks for births and
    corrections; shape-only checks are separately selectable.

    ``tracker_backend`` ("sam2" / "dam4sam", the predictor's) is recorded on
    every masklet. ``diagnostics``, when given, is extended with the camera's
    per-frame mask events (with SAM2's predicted IoU and object-score logit
    when the predictor reports them), birth-gate verdicts, back-fill
    summaries and refinement prompt events (mask_quality.py), and
    only once every window has succeeded: a failed full-clip or halved
    attempt adds none. Each mask event costs a GPU sync, so None skips them.

    Returns one Masklet per confirmed track that appeared in ≥1 frame.
    """
    if not frames or not images:
        return []
    if birth_gate not in ("untracked_area", "mask_iou", "none"):
        raise ValueError(f"unknown birth_gate {birth_gate!r}")

    refinement = refinement or MaskRefinementConfig()
    events: Optional[list[dict]] = [] if diagnostics is not None else None

    def finish(result: list[Masklet]) -> list[Masklet]:
        if diagnostics is not None:
            diagnostics.extend(events)
        return result

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
        autocast_dtype=autocast_dtype,
        ego_mask=ego_mask,
        ego_max_fraction=ego_max_fraction,
        island_fraction=island_fraction,
        birth_score=birth_score,
        confirm_keyframes=confirm_keyframes,
        support_keyframes=support_keyframes,
        support_min_area=support_min_area,
        reprompt_score=reprompt_score,
        reprompt_below_iou=reprompt_below_iou,
        birth_gate=birth_gate,
        birth_untracked_fraction=birth_untracked_fraction,
        birth_max_mask_iou=birth_max_mask_iou,
        backfill_frames=backfill_frames,
        refinement=refinement,
    )

    def attempt(start: int, stop: int) -> dict[int, _Track]:
        window_events: Optional[list[dict]] = [] if events is not None else None
        tracks = _track_window(
            predictor,
            detector,
            frames[start:stop],
            images[start:stop],
            concepts,
            frame_offset=start,
            quality_events=window_events,
            **kw,
        )
        if window_events is not None:
            events.extend(window_events)
        return tracks

    def to_masklets(tracks: dict[int, _Track]) -> list[Masklet]:
        return _tracks_to_masklets(tracks, bag_id, chunk_id, cam_id, tracker_backend)

    # Attempt 1: the whole clip in one session.
    try:
        tracks = attempt(0, len(frames))
        return finish(to_masklets(tracks))
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
        try:
            tr = attempt(start, stop)
        except Exception as exc:  # noqa: BLE001
            if not _is_cuda_oom(exc):
                raise
            _empty_cuda_cache()
            n = stop - start
            if n <= floor:
                raise RuntimeError(
                    f"chunk {chunk_id} / {cam_id}: sub-clip [{start}:{stop}] "
                    f"still OOM at the {floor}-frame floor; camera incomplete"
                ) from exc
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
        out = to_masklets(tr)
        _empty_cuda_cache()
        return out

    masklets: list[Masklet] = []
    for start in range(0, len(frames), window):
        masklets.extend(_track_range(start, min(start + window, len(frames))))
    return finish(masklets)


def _track_window(
    predictor,
    detector,
    frames: list[CameraFrame],
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
    autocast_dtype: str,
    ego_mask: Optional[np.ndarray],
    ego_max_fraction: float,
    island_fraction: float,
    birth_score: float,
    confirm_keyframes: int,
    support_keyframes: int,
    support_min_area: float,
    reprompt_score: float,
    reprompt_below_iou: float,
    birth_gate: str,
    birth_untracked_fraction: float,
    birth_max_mask_iou: float,
    backfill_frames: int,
    refinement: MaskRefinementConfig,
    quality_events: Optional[list[dict]],
) -> dict[int, _Track]:
    """Detect + propagate over one (windowed) frame range in a single session.

    Every failure propagates: a CUDA OOM so the caller can retry in smaller
    windows, anything else (a missing detector or DINOv2 checkpoint, a failed
    mask write) to stop the run, since returning the tracks so far would write
    a camera with no or partial masklets that looks complete. New
    objects are introduced segment-by-segment between keyframes; ``frame_offset``
    is the window's start index in the full clip, used for the DINOv2 sampling
    cadence and log lines (mask filenames use the absolute camera_seq).
    ``quality_events`` (None: off) collects track_camera's diagnostics.
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
    stability_at_kf: dict[int, dict[int, float]] = {}
    mask_prompts: set[tuple[int, int]] = set()
    live: set[int] = set()  # obj_ids currently in the SAM2 session
    ended = _EndedTracks()  # lost / unsupported objects: never recorded again
    next_obj_id = 0
    refiner = KeyframeMaskRefiner(predictor, refinement) if refinement.enabled else None
    # The image predictor a keyframe encodes at most once: refinement's, and
    # the mask_iou birth gate's.
    image_predictor = refiner
    if image_predictor is None and birth_gate == "mask_iou":
        image_predictor = KeyframeMaskRefiner(predictor, refinement)
    # Back-fill inputs: each birth after the window's first image (its
    # keyframe and prompt), and each recorded frame's union of track masks.
    birth_prompts: dict[int, tuple[int, tuple[str, object]]] = {}
    tracked_at: dict[int, np.ndarray] = {}  # frame → packed union
    gate_tested = gate_refused = 0

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
            if image_predictor is not None:
                image_predictor.reset()
            image_ready = False

            def encoded():
                """The image predictor, with this keyframe encoded once."""
                nonlocal image_ready
                if not image_ready:
                    image_predictor.set_image(images[kf])
                    image_ready = True
                return image_predictor

            def propose(det: Detection, oid: int, action: str):
                with amp():
                    candidate = encoded().propose(det.box_xyxy, ego_mask)
                event = {
                    "kind": "prompt",
                    "camera_seq": seq_by_index[kf],
                    "window_start": frame_offset,
                    "object_id": oid,
                    "action": action,
                    "det_score": float(det.score),
                    **candidate.diagnostics(),
                    "applied": False,
                }
                if quality_events is not None:
                    quality_events.append(event)
                return candidate, event

            is_last = ki == len(kfs) - 1
            stop = (n - 1) if is_last else kfs[ki + 1]  # last frame idx to reach
            record_stop = n if is_last else kfs[ki + 1]  # exclusive record bound

            # This keyframe's boxes, highest score first; a box mostly on the
            # vehicle's own body is the vehicle. One matching a track's mask
            # (propagated into this frame) supports it; the rest are births.
            existing = masks_at_kf.pop(kf, {})
            old_stability = stability_at_kf.pop(kf, {})
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

            # Association is unchanged. The opt-in image pass can also detect
            # a silhouette disagreement hidden by a well-overlapping bbox.
            for oid, dets in backing.items():
                det = dets[0]
                if (
                    len(dets) == 1
                    and oid not in ended.ids
                    and oid not in support.tentative
                    and id(det) not in shared_box
                    and det.score >= reprompt_score
                ):
                    drifted = _box_iou(det.box_xyxy, boxes[oid]) < reprompt_below_iou
                    corrected = False
                    if refiner is not None and (
                        drifted or refinement.correct_shape_disagreement
                    ):
                        candidate, event = propose(det, oid, "correction")
                        event["trigger"] = "box_drift" if drifted else "shape_check"
                        if candidate.mask is not None:
                            agreement = _mask_iou(candidate.mask, existing[oid])
                            event["propagated_mask_iou"] = agreement
                            event["propagated_stability"] = old_stability.get(oid, 0.0)
                            # A major disagreement could be the wrong instance.
                            # Record it for review instead of transferring its ID.
                            if (
                                refinement.correction_min_mask_iou
                                <= agreement
                                < refinement.correction_below_mask_iou
                                and candidate.stability >= old_stability.get(oid, 0.0)
                            ):
                                with amp():
                                    predictor.add_new_mask(
                                        state, kf, oid, candidate.mask
                                    )
                                mask_prompts.add((kf, oid))
                                event["applied"] = corrected = True
                            else:
                                event["reason"] = "correction_guard_rejected"
                    if drifted and not corrected:
                        with amp():
                            _add_object(predictor, state, kf, oid, det)
                        if refiner is not None:
                            event["box_fallback"] = True

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
            # highest score first, so the better label wins. The birth gate
            # tests each against the masks of the tracks still live here and
            # of the births accepted before it. SAM2MOT defines its untracked
            # area over existing tracks; counting this keyframe's births too
            # makes a part box fail behind its car's box at any score.
            gated = birth_gate != "none" and bool(births)
            tracked_masks = (
                [m for oid, m in existing.items() if oid not in ended.ids]
                if gated
                else []
            )
            tracked_union = (
                np.logical_or.reduce(tracked_masks) if tracked_masks else None
            )
            accepted: list[tuple[float, float, float, float]] = []
            for det in births:
                if any(
                    _box_iou(det.box_xyxy, b) >= iou_match_threshold for b in accepted
                ):
                    continue
                if gated:
                    if birth_gate == "untracked_area":
                        measure = _untracked_share(det.box_xyxy, tracked_union)
                        passed = measure > birth_untracked_fraction
                    elif not tracked_masks:  # nothing to overlap: skip encoding
                        measure, passed = 0.0, True
                    else:
                        with amp():
                            det_mask = encoded().box_mask(det.box_xyxy)
                        if ego_mask is not None:
                            det_mask = det_mask & ~ego_mask
                        measure = max(
                            (_mask_iou(det_mask, m) for m in tracked_masks),
                            default=0.0,
                        )
                        passed = measure < birth_max_mask_iou
                    gate_tested += 1
                    gate_refused += not passed
                    if quality_events is not None:
                        quality_events.append(
                            {
                                "kind": "birth",
                                "camera_seq": seq_by_index[kf],
                                "window_start": frame_offset,
                                "object_id": next_obj_id if passed else None,
                                "cls": det.cls,
                                "det_score": float(det.score),
                                "box": [float(v) for v in det.box_xyxy],
                                "gate": birth_gate,
                                (
                                    "untracked_fraction"
                                    if birth_gate == "untracked_area"
                                    else "max_mask_iou"
                                ): float(measure),
                                "accepted": passed,
                            }
                        )
                    if not passed:
                        continue
                accepted.append(det.box_xyxy)
                oid = next_obj_id
                next_obj_id += 1
                obj_meta[oid] = (det.cls, det.score)
                seed_box[oid] = det.box_xyxy
                candidate = None
                if refiner is not None:
                    candidate, event = propose(det, oid, "birth")
                mask_seed = candidate is not None and candidate.mask is not None
                with amp():
                    if mask_seed:
                        prompted = predictor.add_new_mask(
                            state, kf, oid, candidate.mask
                        )
                        mask_prompts.add((kf, oid))
                        event["applied"] = True
                    else:
                        prompted = _add_object(predictor, state, kf, oid, det)
                live.add(oid)
                support.born(oid, det)
                if backfill_frames > 0 and kf > 0:
                    birth_prompts[oid] = (
                        kf,
                        ("mask", candidate.mask) if mask_seed else ("box", det.box_xyxy),
                    )
                if gated:
                    born = _prompt_mask(prompted, oid, (H, W))
                    if born is not None:
                        if ego_mask is not None:
                            born = born & ~ego_mask
                        tracked_masks.append(born)
                        tracked_union = (
                            born if tracked_union is None else tracked_union | born
                        )

            if image_predictor is not None:
                image_predictor.reset()  # free image features before propagation

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
                # Stability costs a GPU sync per frame: only for diagnostics,
                # and at keyframes for refinement's correction guard.
                stable = (
                    stability_scores(mask_logits, refinement.stability_offset)
                    if quality_events is not None
                    or (refiner is not None and f in kf_set and f > kf)
                    else None
                )
                raw_areas = (
                    [int(m.sum()) if m is not None else 0 for m in masks]
                    if quality_events is not None
                    else None
                )
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
                # Preserve every live object's independent claims. Neither
                # logits nor containment can discard a competing hypothesis
                # before its LiDAR support has been geometrically refined.
                present = ended.claim(
                    ids,
                    masks,
                    drop_lost_after_frames=drop_lost_after_frames,
                )
                if f in kf_set and f > kf:
                    masks_at_kf[f] = dict(present)
                if backfill_frames > 0 and f < record_stop and present:
                    tracked_at[f] = np.packbits(
                        np.logical_or.reduce([m for _, m in present])
                    )
                    if stable is not None:
                        stability_at_kf[f] = dict(zip(ids, map(float, stable)))
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
                if f < record_stop and quality_events is not None:
                    saved = {oid: int(m.sum()) for oid, m in present}
                    for oid, area, stability in zip(ids, raw_areas, stable):
                        kept = saved.get(oid, 0)
                        quality_events.append(
                            {
                                "kind": "mask",
                                "camera_seq": seq_by_index[f],
                                "window_start": frame_offset,
                                "object_id": oid,
                                "logit_source": "mask_prompt"
                                if (f, oid) in mask_prompts
                                else "decoder",
                                "stability": float(stability),
                                "raw_area": area,
                                "postprocess_area": kept,
                                "removed_fraction": 1 - kept / area if area else 0.0,
                                "ended": oid in ended.ids,
                                "tentative": oid in support.tentative,
                                **(
                                    {"memory": predictor.memory_diagnostics(state, oid)}
                                    if hasattr(predictor, "memory_diagnostics")
                                    else {}
                                ),
                                # SAM2's own confidence (predicted IoU, object
                                # score), which published trackers gate on.
                                **(
                                    predictor.frame_quality(state, oid)
                                    if hasattr(predictor, "frame_quality")
                                    else {}
                                ),
                            }
                        )
                if f >= stop:
                    break
        if gate_tested:
            log.info(
                "chunk %s / %s [%d:%d]: birth gate (%s) refused %d of %d new boxes",
                chunk_id,
                cam_id,
                frame_offset,
                frame_offset + n,
                birth_gate,
                gate_refused,
                gate_tested,
            )

        # Back-fill each written track born after the window's first image
        # from its birth prompt, before its first recorded frame.
        seeds: dict[int, list[tuple[int, tuple[str, object]]]] = {}
        for oid, (kf, prompt) in birth_prompts.items():
            track = tracks.get(oid)
            if (
                track is not None
                and oid not in support.tentative
                and track.frames_present
                and min(track.frames_present) == seq_by_index[kf]
            ):
                seeds.setdefault(kf, []).append((oid, prompt))
        if seeds:
            with amp():
                predictor.reset_state(state)  # free the forward memories
            filled = _backfill(
                predictor,
                frames,
                images,
                seeds,
                tracked_at,
                span=backfill_frames,
                amp=amp,
                offload_video_to_cpu=offload_video_to_cpu,
                ego_mask=ego_mask,
                ego_max_fraction=ego_max_fraction,
                island_fraction=island_fraction,
                drop_lost_after_frames=drop_lost_after_frames,
                untracked_fraction=birth_untracked_fraction,
                quality_events=quality_events,
                seq_by_index=seq_by_index,
                frame_offset=frame_offset,
            )
            for oid, found in filled.items():
                track = tracks[oid]
                for f, mask in found:
                    track.frames_present.append(seq_by_index[f])
                    track.mask_writes.append(
                        writer.submit(
                            masklets.write_mask,
                            bag_id,
                            chunk_id,
                            track.masklet_id,
                            seq_by_index[f],
                            mask,
                        )
                    )
        tracked_at.clear()

        # A track still tentative at the end has only held masks, so it is
        # never written (_tracks_to_masklets skips it).
        for oid, track in tracks.items():
            track.cls, track.score = support.label(oid)
        saved_frames = {oid: set(track.frames_present) for oid, track in tracks.items()}
        for event in quality_events or ():
            track = tracks.get(event["object_id"])
            event["masklet_id"] = track.masklet_id if track is not None else None
            if event["kind"] == "mask":
                event["saved_area"] = (
                    event["postprocess_area"]
                    if event["camera_seq"] in saved_frames.get(event["object_id"], ())
                    else 0
                )
    finally:
        # Let queued mask PNGs land before the tracks are handed back; a failed
        # write surfaces from Future.result() in _track_to_masklet.
        writer.shutdown(wait=True)
        if image_predictor is not None:
            image_predictor.reset()
        if state is not None:
            try:
                predictor.reset_state(state)
            except Exception:  # noqa: BLE001
                pass
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return tracks


def _backfill(
    predictor,
    frames: list[CameraFrame],
    images: list[np.ndarray],
    seeds: dict[int, list[tuple[int, tuple[str, object]]]],
    tracked_at: dict[int, np.ndarray],
    *,
    span: int,
    amp,
    offload_video_to_cpu: bool,
    ego_mask: Optional[np.ndarray],
    ego_max_fraction: float,
    island_fraction: float,
    drop_lost_after_frames: int,
    untracked_fraction: float,
    quality_events: Optional[list[dict]],
    seq_by_index: list[int],
    frame_offset: int,
) -> dict[int, list[tuple[int, np.ndarray]]]:
    """Propagate tracks backwards from their births: window frame → mask each.

    CTRL extends tracks backwards after forward tracking ("once detected,
    never lost"), OC-SORT writes back a track's unconfirmed head, and SAM4D
    propagates each object first detected at a later keyframe back to the
    start; SAM2 was trained on reversed clips too. ``seeds`` maps a window
    frame to the tracks born there with their birth prompts (box or mask).

    One pass feeds the window's frames to the predictor in reverse order
    (a frame dir from the latest birth back), so both backends run their
    ordinary forward propagation. Each track is re-prompted at its birth image
    and followed back at most ``span`` images, stopping where the forward
    rules would end it read in reverse (``drop_lost_after_frames`` empty
    masks, a mask on the ego vehicle) or where its mask's bbox fails
    SAM2MOT's untracked-area test against the union of masks recorded there
    (``tracked_at``): that object is already tracked, or SAM2 has latched onto
    one. Masks get the forward post-processing (ego pixels, islands).
    """
    H, W = images[0].shape[:2]
    starts = sorted(seeds, reverse=True)  # latest birth first
    lo = max(0, starts[-1] - span)
    order = list(range(starts[0], lo - 1, -1))  # window frames, latest first
    pos = {f: r for r, f in enumerate(order)}
    born: dict[int, int] = {}  # obj_id → birth frame
    limit: dict[int, int] = {}  # obj_id → earliest frame it may reach
    done: dict[int, int] = {}  # obj_id → earliest frame handled so far
    empty_run: dict[int, int] = {}
    stopped: dict[int, str] = {}
    live: set[int] = set()
    out: dict[int, list[tuple[int, np.ndarray]]] = {}
    # SAM2's predicted IoU per added mask, for diagnostics.
    quality: dict[int, list[Optional[float]]] = {}
    reports_quality = quality_events is not None and hasattr(predictor, "frame_quality")

    tmpdir: Optional[str] = None
    state = None
    try:
        tmpdir = _prepare_frame_dir(
            [frames[f] for f in order], [images[f] for f in order]
        )
        with amp():
            state = predictor.init_state(
                video_path=tmpdir,
                offload_video_to_cpu=offload_video_to_cpu,
                async_loading_frames=True,
            )
        for i, kf in enumerate(starts):
            for oid in sorted(o for o in live if o in stopped):
                with amp():
                    predictor.remove_object(state, oid, need_output=False)
                live.discard(oid)
            for oid, (kind, prompt) in seeds[kf]:
                with amp():
                    if kind == "mask":
                        predictor.add_new_mask(state, pos[kf], oid, prompt)
                    else:
                        predictor.add_new_points_or_box(
                            inference_state=state,
                            frame_idx=pos[kf],
                            obj_id=oid,
                            box=np.asarray(prompt, dtype=np.float32),
                        )
                live.add(oid)
                born[oid] = done[oid] = kf  # the birth image is already recorded
                limit[oid] = max(lo, kf - span)
            running = {o for o in live if o not in stopped}
            if not running:
                continue
            # Propagate until every running track is exhausted, or to the next
            # birth (where the next segment carries the running ones on).
            end = min(limit[o] for o in running)
            if i + 1 < len(starts):
                end = max(end, starts[i + 1])
            union_f, union = None, None
            for r, obj_ids, mask_logits in _iter_in(
                amp,
                predictor.propagate_in_video(
                    state,
                    start_frame_idx=pos[kf],
                    max_frame_num_to_track=pos[end] - pos[kf],
                ),
            ):
                f = order[int(r)]
                ids = [int(o) for o in obj_ids]
                masks = _frame_masks(mask_logits > 0, (H, W))
                for oid, mask in zip(ids, masks):
                    if oid in stopped or f >= done[oid]:
                        continue
                    done[oid] = f
                    if mask is not None and ego_mask is not None:
                        if _inside_fraction(mask, ego_mask) >= ego_max_fraction:
                            stopped[oid] = "ego"
                            continue
                        mask = mask & ~ego_mask
                    mask = _drop_islands(mask, island_fraction)
                    if mask is None:
                        run = empty_run[oid] = empty_run.get(oid, 0) + 1
                        if 0 < drop_lost_after_frames <= run:
                            stopped[oid] = "lost"
                    else:
                        empty_run[oid] = 0
                        if f in tracked_at:
                            if union_f != f:
                                union_f = f
                                union = np.unpackbits(
                                    tracked_at[f], count=H * W
                                ).reshape(H, W).astype(bool)
                            box = _mask_bbox(mask)
                            if _untracked_share(box, union) <= untracked_fraction:
                                stopped[oid] = "tracked"
                                continue
                        out.setdefault(oid, []).append((f, mask))
                        if reports_quality:
                            quality.setdefault(oid, []).append(
                                predictor.frame_quality(state, oid).get("predicted_iou")
                            )
                    if oid not in stopped and f <= limit[oid]:
                        stopped[oid] = "span" if f == born[oid] - span else "window"
                if all(o in stopped for o in live):
                    break
    finally:
        if state is not None:
            try:
                predictor.reset_state(state)
            except Exception:  # noqa: BLE001
                pass
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)

    if quality_events is not None:
        for oid, kf in born.items():
            quality_events.append(
                {
                    "kind": "backfill",
                    "camera_seq": seq_by_index[kf],
                    "window_start": frame_offset,
                    "object_id": oid,
                    "frames_added": len(out.get(oid, ())),
                    "stop": stopped.get(oid, "window"),
                    # newest image first, when the predictor reports it
                    "predicted_iou": quality.get(oid, []),
                }
            )
    return out


def _add_object(predictor, state, frame_idx: int, obj_id: int, det: Detection):
    """Prompt with a box; returns SAM2's ``(frame_idx, obj_ids, mask_logits)``.
    The pinned SAM2 supports late births; errors raise.

    In particular, an OOM must reach the window retry, and a failed correction
    or late birth must never leave a successful-looking partial camera.
    """
    return predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        box=np.asarray(det.box_xyxy, dtype=np.float32),
    )


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
        # A failure raises: a configured embedding must not go silently
        # missing, and a missing model would fail on every frame.
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
    tracks: dict[int, _Track],
    bag_id: str,
    chunk_id: str,
    cam_id: str,
    tracker_backend: str,
) -> list[Masklet]:
    return [
        _track_to_masklet(t, bag_id, chunk_id, cam_id, tracker_backend)
        for t in tracks.values()
        if t.frames_present
    ]


def _track_to_masklet(
    track: _Track, bag_id: str, chunk_id: str, cam_id: str, tracker_backend: str
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
        tracker_backend=tracker_backend,
    )
