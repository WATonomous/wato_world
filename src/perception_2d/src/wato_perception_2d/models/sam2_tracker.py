"""Detector + SAM2 video tracker — the primary (only) tracker.

For one camera stream, runs the 2D detector on keyframes and uses SAM2's video
predictor to segment + propagate each detected box into a tracked masklet:

  1. Write the camera's frames to a temp JPEG dir and ``init_state`` on it
     (SAM2's video loader takes a directory of sequentially-named JPEGs).
  2. At each keyframe (every ``redetect_every_k`` frames) run the detector →
     class-labeled boxes. Boxes that overlap an already-tracked object (mask IoU
     ≥ ``iou_match_threshold``) are dropped as existing; the genuinely new ones
     get a fresh obj_id via ``add_new_points_or_box``.
  3. ``propagate_in_video`` forward to the next keyframe, recording per-object
     masks → ``_Track`` → ``Masklet`` (per-frame PNGs + DINOv2 features every k).

The detector emits one box per object, so each physical object becomes a single
masklet — no synonym explosion, far fewer masks for downstream to consume.

Object ids reset between OOM sub-clip windows (and between chunks) — exactly the
discontinuity the downstream `tracking` component re-links via the DINOv2
embeddings persisted here. Cross-camera identity is likewise deferred downstream.

SAM2 video predictor API (sam2 package):
    state = predictor.init_state(video_path=<jpeg_dir>, offload_video_to_cpu=True)
    predictor.add_new_points_or_box(inference_state=state, frame_idx=i,
                                    obj_id=oid, box=np.array([x1,y1,x2,y2]))
    for f_idx, obj_ids, mask_logits in predictor.propagate_in_video(
            state, start_frame_idx=i, max_frame_num_to_track=n):
        ...   # mask_logits: (num_objs, 1, H, W); obj_ids parallel
    predictor.reset_state(state)
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import uuid
from typing import Optional

import numpy as np

from wato_perception_2d.fusion.masklet import Masklet
from wato_perception_2d.io import CameraFrameInfo
from wato_perception_2d.models.detector import Concept, Detection
from wato_perception_2d.models.embeddings import extract_dino_feature

log = logging.getLogger(__name__)


class _Track:
    """Accumulator for one tracked obj_id across frames."""

    __slots__ = (
        "masklet_id",
        "cls",
        "score",
        "frames_present",
        "mask_paths",
        "dino_accum",
    )

    def __init__(self, masklet_id: str, cls: str, score: float) -> None:
        self.masklet_id = masklet_id
        self.cls = cls
        self.score = score
        self.frames_present: list[int] = []
        self.mask_paths: list[str] = []
        self.dino_accum: list[np.ndarray] = []


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


def _is_existing(
    det: Detection, existing_masks: dict[int, np.ndarray], iou_threshold: float
) -> bool:
    """True when a detection box already matches a tracked object's mask bbox.

    Compares the detection box against the bounding box of each currently-tracked
    object's mask at this frame; ``>= iou_threshold`` means "already tracked", so
    the detection is not added as a new object. Pure function — unit-tested.
    """
    for mask in existing_masks.values():
        bbox = _mask_bbox(mask)
        if bbox is not None and _box_iou(det.box_xyxy, bbox) >= iou_threshold:
            return True
    return False


def keyframes(n_frames: int, every_k: int) -> list[int]:
    """Frame indices where the detector runs: [0, k, 2k, …] within [0, n)."""
    if n_frames <= 0:
        return []
    step = max(1, every_k)
    return list(range(0, n_frames, step))


def _logit_to_mask(logit, hw: tuple[int, int]) -> Optional[np.ndarray]:
    """SAM2 per-object logits → boolean mask at the frame's (H, W), or None.

    Returns None for an empty mask (object absent in this frame — normal) so the
    track simply skips the frame.
    """
    arr = _to_numpy(logit)
    if arr.ndim == 3:  # (1, H, W)
        arr = arr[0]
    if arr.ndim != 2:
        return None
    m = arr > 0.0
    if m.shape != hw:
        from PIL import Image as PILImage

        H, W = hw
        m = np.asarray(
            PILImage.fromarray(m.astype(np.uint8)).resize((W, H), PILImage.NEAREST),
            dtype=bool,
        )
    return m if m.any() else None


def _save_mask(
    base_dir: str, masklet_id: str, camera_seq: int, mask: np.ndarray
) -> str:
    from PIL import Image as PILImage

    d = os.path.join(base_dir, masklet_id)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{camera_seq:06d}.png")
    PILImage.fromarray(mask.astype(np.uint8) * 255).save(path)
    return path


def _write_frames_jpeg(images: list[np.ndarray]) -> str:
    """Write frames to a temp dir as sequential JPEGs for SAM2's video loader.

    SAM2's ``init_state`` reads a directory of JPEGs whose names are integers (it
    orders frames by ``int(filename)``), so we always re-encode to a clean
    ``00000000.jpg`` sequence — robust to the source format (ingest may write PNG)
    and to gaps left by frame-index filtering. Caller removes the dir afterwards.
    """
    from PIL import Image as PILImage

    tmp = tempfile.mkdtemp(prefix="sam2_frames_")
    for i, img in enumerate(images):
        PILImage.fromarray(img).save(os.path.join(tmp, f"{i:08d}.jpg"), quality=95)
    return tmp


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
    masks_2d_base_dir: str,
    dino_model: str = "dinov2_vitl14",
    dino_every_k: int = 5,
    device: str = "cuda",
    redetect_every_k: int = 10,
    iou_match_threshold: float = 0.5,
    offload_video_to_cpu: bool = True,
    sub_clip_frames: int = 150,
    min_sub_clip_frames: int = 16,
) -> list[Masklet]:
    """Detect + SAM2-track every taxonomy object across one camera's frames.

    Tries the whole clip in one SAM2 session. On a CUDA OOM it falls back to
    windows of ``sub_clip_frames`` (fresh session each); a window that itself
    OOMs is recursively halved down to ``min_sub_clip_frames`` before being
    skipped. Object ids reset between windows (re-linked downstream via DINOv2).

    Returns one Masklet per tracked object that appeared in ≥1 frame.
    """
    if not frames or not images:
        return []

    kw = dict(
        bag_id=bag_id,
        chunk_id=chunk_id,
        cam_id=cam_id,
        masks_2d_base_dir=masks_2d_base_dir,
        dino_model=dino_model,
        dino_every_k=dino_every_k,
        device=device,
        redetect_every_k=redetect_every_k,
        iou_match_threshold=iou_match_threshold,
        offload_video_to_cpu=offload_video_to_cpu,
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
    masks_2d_base_dir: str,
    dino_model: str,
    dino_every_k: int,
    device: str,
    redetect_every_k: int,
    iou_match_threshold: float,
    offload_video_to_cpu: bool,
) -> dict[int, _Track]:
    """Detect + propagate over one (windowed) frame range in a single session.

    Re-raises a CUDA OOM so the caller can retry in smaller windows; swallows any
    other failure (logging it) and returns the tracks accumulated so far. New
    objects are introduced segment-by-segment between keyframes; ``frame_offset``
    is the window's start index in the full clip, used for the DINOv2 sampling
    cadence and log lines (mask filenames use the absolute camera_seq).
    """
    n = len(frames)
    if n == 0:
        return {}
    seq_by_index = [f.camera_seq for f in frames]
    H, W = images[0].shape[:2]

    tracks: dict[int, _Track] = {}
    obj_meta: dict[int, tuple[str, float]] = {}  # obj_id → (class, det_score)
    masks_at_frame: dict[int, dict[int, np.ndarray]] = {}  # frame_idx → {oid: mask}
    next_obj_id = 0

    tmpdir: Optional[str] = None
    state = None
    try:
        tmpdir = _write_frames_jpeg(images)
        state = predictor.init_state(
            video_path=tmpdir, offload_video_to_cpu=offload_video_to_cpu
        )

        kfs = keyframes(n, redetect_every_k)
        for ki, kf in enumerate(kfs):
            is_last = ki == len(kfs) - 1
            stop = (n - 1) if is_last else kfs[ki + 1]  # last frame idx to reach
            record_stop = n if is_last else kfs[ki + 1]  # exclusive record bound

            # Introduce objects detected at this keyframe that aren't already
            # tracked (matched against masks propagated into this frame).
            existing = masks_at_frame.get(kf, {})
            for det in detector.detect(images[kf], concepts):
                if _is_existing(det, existing, iou_match_threshold):
                    continue
                oid = next_obj_id
                next_obj_id += 1
                obj_meta[oid] = (det.cls, det.score)
                if not _add_object(predictor, state, kf, oid, det, new_segment=kf > 0):
                    # Mid-stream add unsupported on this build → stop introducing
                    # new objects; existing ones still propagate to the clip end.
                    del obj_meta[oid]
                    break

            # Propagate this segment; record [kf, record_stop), update masks for
            # the next keyframe match (incl. the boundary frame, which the next
            # segment records as its own keyframe).
            for f_idx, obj_ids, mask_logits in predictor.propagate_in_video(
                state, start_frame_idx=kf, max_frame_num_to_track=stop - kf
            ):
                fmap = masks_at_frame.setdefault(int(f_idx), {})
                for j in range(len(obj_ids)):
                    oid = int(obj_ids[j])
                    mask = _logit_to_mask(mask_logits[j], (H, W))
                    if mask is None:
                        continue
                    fmap[oid] = mask
                    if int(f_idx) < record_stop:
                        _record_frame(
                            tracks,
                            obj_meta,
                            oid,
                            frame_offset + int(f_idx),
                            seq_by_index[int(f_idx)],
                            mask,
                            images[int(f_idx)],
                            masks_2d_base_dir,
                            dino_model,
                            dino_every_k,
                            device,
                        )
                if int(f_idx) >= stop:
                    break
    except Exception as exc:  # noqa: BLE001
        if _is_cuda_oom(exc):
            raise  # let the caller retry this range in smaller windows
        log.warning(
            "chunk %s / %s: SAM2 tracking failed (%s) — keeping %d tracks so far.",
            chunk_id,
            cam_id,
            exc,
            len(tracks),
            exc_info=True,
        )
    finally:
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

    A new object at frame 0 must succeed (the error propagates). At a later
    keyframe, some SAM2 builds reject adding an object after propagation has
    started; we isolate that (warn once, return False) so existing tracks still
    finish — graceful degradation of the re-detection feature, not garbage output.
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
            raise  # a frame-0 failure is a real error
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
    obj_id: int,
    frame_index_abs: int,
    camera_seq: int,
    mask: np.ndarray,
    image_rgb: np.ndarray,
    masks_2d_base_dir: str,
    dino_model: str,
    dino_every_k: int,
    device: str,
) -> None:
    track = tracks.get(obj_id)
    if track is None:
        cls, score = obj_meta.get(obj_id, ("object", 0.0))
        track = _Track(masklet_id=str(uuid.uuid4())[:12], cls=cls, score=score)
        tracks[obj_id] = track
    mask_path = _save_mask(masks_2d_base_dir, track.masklet_id, camera_seq, mask)
    track.frames_present.append(camera_seq)
    track.mask_paths.append(mask_path)
    if dino_every_k > 0 and frame_index_abs % dino_every_k == 0:
        # DINOv2 (torch.hub) is independent of SAM2 and only feeds the downstream
        # tracking ReID — a failure here must NOT abort the camera. Isolate it.
        try:
            feat = extract_dino_feature(image_rgb, mask, dino_model, device)
        except Exception as exc:  # noqa: BLE001
            feat = None
            log.warning(
                "DINOv2 embedding failed at frame %d (%s) — keeping mask, "
                "skipping embedding.",
                frame_index_abs,
                exc,
            )
        if feat is not None:
            track.dino_accum.append(feat)


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
    mask_paths = [track.mask_paths[i] for i in order]
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
