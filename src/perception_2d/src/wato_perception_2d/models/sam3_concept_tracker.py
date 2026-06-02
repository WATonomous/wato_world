"""SAM 3.1 multiplex concept-video tracker — the primary (only) tracker.

For one camera stream, runs Meta's SAM 3.1 multiplex video predictor once per
taxonomy concept (a short text phrase). The predictor detects, segments, and
*tracks* every instance of the concept across the whole frame sequence, assigning
each a persistent object id. We group its per-frame outputs by (concept, obj_id)
into `Masklet`s — replacing the old discovery + per-frame segmenter + IoU tracker
stack with the model's own tracking.

API (Sam3MultiplexVideoPredictor, from the `sam3` package):
    sid = predictor.handle_request({"type":"start_session","resource_path":frames})["session_id"]
    predictor.handle_request({"type":"reset_session","session_id":sid})
    out = predictor.handle_request({"type":"add_prompt","session_id":sid,
                                    "frame_index":0,"text":concept})["outputs"]
    for r in predictor.handle_stream_request({"type":"propagate_in_video",
             "session_id":sid,"propagation_direction":"forward","start_frame_index":0}):
        ...   # r["frame_index"], r["outputs"]
    predictor.handle_request({"type":"close_session","session_id":sid})

`outputs` carries parallel arrays: out_obj_ids, out_probs, out_binary_masks
(N, H, W bool), out_boxes_xywh. `resource_path` accepts a list of PIL images.
"""

from __future__ import annotations

import itertools
import logging
import os
import time
import uuid
from typing import Optional

import numpy as np

from wato_perception_2d.fusion.masklet import Masklet
from wato_perception_2d.io import CameraFrameInfo
from wato_perception_2d.models.embeddings import extract_dino_feature

log = logging.getLogger(__name__)


class _Track:
    """Accumulator for one (concept, obj_id) across frames."""

    __slots__ = (
        "masklet_id",
        "cls",
        "score",
        "frames_present",
        "mask_paths",
        "dino_accum",
    )

    def __init__(self, masklet_id: str, cls: str) -> None:
        self.masklet_id = masklet_id
        self.cls = cls
        self.score = 0.0
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
    """True for a CUDA out-of-memory failure.

    Matches torch.cuda.OutOfMemoryError by name (so torch needn't be imported
    at module scope) and also the generic "CUDA out of memory" RuntimeError some
    paths raise. Used to distinguish an OOM (retry in smaller sub-clips) from any
    other tracking failure (swallow + keep partial tracks).
    """
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    return "out of memory" in str(exc).lower()


def _empty_cuda_cache() -> None:
    """gc + empty_cache so a window's freed allocations return to the driver
    before the next window's session is built (best-effort, no-op on CPU)."""
    try:
        import gc

        import torch

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _start_session(predictor, pil_frames: list, offload_video_to_cpu: bool) -> str:
    """Start a multiplex video session on an in-memory list of PIL frames.

    The shared Sam3BasePredictor.start_session passes init_state kwargs
    (offload_state_to_cpu / video_loader_type) that the multiplex model's
    init_state rejects, so we call init_state directly with the args it accepts
    and register the session in the predictor's state map ourselves. The
    add_prompt / propagate paths filter kwargs by signature, so only
    start_session needs this workaround.

    offload_video_to_cpu keeps the full (N, 3, image_size, image_size) frame
    stack in host RAM instead of pinning it all on the GPU; _get_image_feature
    streams one frame to the GPU per propagation step. Essential for long
    panoramic clips that would otherwise OOM at init_state.
    """
    inference_state = predictor.model.init_state(
        resource_path=pil_frames,
        offload_video_to_cpu=offload_video_to_cpu,
        async_loading_frames=getattr(predictor, "async_loading_frames", False),
    )
    sid = str(uuid.uuid4())
    now = time.time()
    predictor._all_inference_states[sid] = {
        "state": inference_state,
        "session_id": sid,
        "start_time": now,
        "last_use_time": now,
    }
    return sid


def track_camera_concepts(
    predictor,
    frames: list[CameraFrameInfo],
    images: list[np.ndarray],
    concept_prompts: list[tuple[str, str]],
    *,
    bag_id: str,
    chunk_id: str,
    cam_id: str,
    masks_2d_base_dir: str,
    dino_model: str = "dinov2_vitl14",
    dino_every_k: int = 5,
    device: str = "cuda",
    output_prob_thresh: float = 0.5,
    offload_video_to_cpu: bool = True,
    sub_clip_frames: int = 150,
    min_sub_clip_frames: int = 16,
) -> list[Masklet]:
    """Track every concept across one camera's frames → list of Masklets.

    Tries the whole clip in one SAM 3.1 session. On a CUDA OOM (the multiplex
    tracker's per-clip memory grows with sequence length until even a 24 GB card
    OOMs mid-propagation), it falls back to processing the frames in windows of
    ``sub_clip_frames``, one fresh session each. A window that *itself* OOMs
    (a pathologically dense stretch) is not dropped: it is recursively halved
    and each half retried, down to a ``min_sub_clip_frames`` floor — only a
    range that still OOMs at the floor is skipped. So a dense region fragments
    finely instead of losing a whole ``sub_clip_frames``-wide swath. Object ids
    reset between windows — exactly the same discontinuity as between chunks —
    which the downstream `tracking` component re-links via the DINOv2 embeddings
    persisted here. Graceful degradation: the full-clip path is unchanged when it
    fits, and a clip that no longer fits yields windowed masklets instead of
    dying.

    Args:
        predictor: a Sam3MultiplexVideoPredictor (from get_sam3_predictor()).
        frames: CameraFrameInfo sorted by camera_seq.
        images: (H, W, 3) uint8 RGB arrays, parallel to `frames`.
        concept_prompts: list of (text_prompt, canonical_class) — one per
            taxonomy class. text_prompt seeds SAM 3.1; canonical_class is the cls.
        masks_2d_base_dir: where per-masklet PNG dirs are written.
        offload_video_to_cpu: keep the frame stack in host RAM and stream it to
            the GPU per frame (avoids the up-front full-video VRAM allocation).
        sub_clip_frames: initial window size (frames) for the OOM fallback.
        min_sub_clip_frames: floor below which a still-OOMing window is skipped
            rather than halved further.

    Returns one Masklet per (concept, obj_id) that the predictor tracked.
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
        output_prob_thresh=output_prob_thresh,
        offload_video_to_cpu=offload_video_to_cpu,
    )

    # Attempt 1: the whole clip in one session.
    try:
        tracks = _track_window(predictor, frames, images, concept_prompts, frame_offset=0, **kw)
        return _tracks_to_masklets(tracks, bag_id, chunk_id, cam_id)
    except Exception as exc:  # noqa: BLE001
        # _track_window only re-raises CUDA OOM; any other failure is swallowed
        # there (keeping partial tracks). Anything else here is unexpected.
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

    # Attempt 2: windowed, fresh session each. A window that itself OOMs
    # (pathologically dense) is recursively halved down to ``floor`` before
    # being skipped, so a busy stretch fragments finely instead of dropping a
    # whole ``window``-wide swath.
    def _track_range(start: int, stop: int) -> list[Masklet]:
        fseg = frames[start:stop]
        iseg = images[start:stop]
        try:
            tr = _track_window(
                predictor, fseg, iseg, concept_prompts, frame_offset=start, **kw
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
    frames: list[CameraFrameInfo],
    images: list[np.ndarray],
    concept_prompts: list[tuple[str, str]],
    *,
    frame_offset: int,
    bag_id: str,
    chunk_id: str,
    cam_id: str,
    masks_2d_base_dir: str,
    dino_model: str,
    dino_every_k: int,
    device: str,
    output_prob_thresh: float,
    offload_video_to_cpu: bool,
) -> dict[tuple[str, int], _Track]:
    """Run every concept over one (windowed) frame range in a single session.

    Re-raises a CUDA OOM so the caller can retry in smaller windows; swallows any
    other failure (logging it) and returns the tracks accumulated so far — a
    DINOv2 / serialization hiccup must not abort the camera. ``frame_offset`` is
    the window's start index in the full clip, used only for DINOv2 sampling
    cadence and log lines (mask filenames use the absolute camera_seq).
    """
    from PIL import Image as PILImage

    pil_frames = [PILImage.fromarray(img) for img in images]
    seq_by_index = [f.camera_seq for f in frames]
    n_frames = len(pil_frames)

    tracks: dict[tuple[str, int], _Track] = {}

    sid: Optional[str] = None
    try:
        sid = _start_session(predictor, pil_frames, offload_video_to_cpu)
        # SAM 3.1 has copied the frames into inference_state; drop our PIL list so
        # CPython reclaims it (~4 GB for a 650-frame 1008px clip) rather than
        # pinning it for the whole propagation.
        del pil_frames

        for text_prompt, canonical in concept_prompts:
            predictor.handle_request({"type": "reset_session", "session_id": sid})

            resp = predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": sid,
                    "frame_index": 0,
                    "text": text_prompt,
                    "output_prob_thresh": output_prob_thresh,
                }
            )

            # Drain the propagation stream incrementally: each frame's masks are
            # written to a PNG and released as it arrives. Buffering the whole
            # stream first (the previous behaviour) held every frame's mask
            # tensors — GPU tensors at full video resolution — live for the
            # entire propagation, which pins them past SAM 3.1's own
            # offload/trim and grows memory with clip length. The `seen` set
            # keeps each frame processed exactly once (the stream re-yields the
            # prompted frame 0). frames are emitted in increasing order, so
            # per-track lists stay time-ordered (re-sorted in _track_to_masklet
            # for safety).
            seen_frames: set[int] = set()
            for r in itertools.chain(
                [resp],
                predictor.handle_stream_request(
                    {
                        "type": "propagate_in_video",
                        "session_id": sid,
                        "propagation_direction": "forward",
                        "start_frame_index": 0,
                        "output_prob_thresh": output_prob_thresh,
                    }
                ),
            ):
                fi = int(r["frame_index"])
                if fi in seen_frames or not (0 <= fi < n_frames):
                    continue
                seen_frames.add(fi)
                _accumulate_frame(
                    tracks,
                    canonical,
                    frame_offset + fi,
                    seq_by_index[fi],
                    r["outputs"],
                    image_rgb=images[fi],
                    masks_2d_base_dir=masks_2d_base_dir,
                    dino_model=dino_model,
                    dino_every_k=dino_every_k,
                    device=device,
                )
    except Exception as exc:  # noqa: BLE001
        if _is_cuda_oom(exc):
            raise  # let the caller retry this range in smaller windows
        log.warning(
            "chunk %s / %s: SAM 3.1 tracking failed (%s) — keeping %d tracks so far.",
            chunk_id,
            cam_id,
            exc,
            len(tracks),
            exc_info=True,
        )
    finally:
        if sid is not None:
            try:
                predictor.handle_request({"type": "close_session", "session_id": sid})
            except Exception:  # noqa: BLE001
                pass

    return tracks


def _tracks_to_masklets(
    tracks: dict[tuple[str, int], _Track], bag_id: str, chunk_id: str, cam_id: str
) -> list[Masklet]:
    return [
        _track_to_masklet(t, bag_id, chunk_id, cam_id)
        for t in tracks.values()
        if t.frames_present
    ]


def _accumulate_frame(
    tracks: dict[tuple[str, int], _Track],
    canonical: str,
    frame_index: int,
    camera_seq: int,
    outputs: dict,
    *,
    image_rgb: np.ndarray,
    masks_2d_base_dir: str,
    dino_model: str,
    dino_every_k: int,
    device: str,
) -> None:
    masks = outputs.get("out_binary_masks")
    obj_ids = outputs.get("out_obj_ids")
    if masks is None or obj_ids is None:
        return
    obj_ids_np = _to_numpy(obj_ids)
    probs = outputs.get("out_probs")
    probs_np = _to_numpy(probs) if probs is not None else None
    H, W = image_rgb.shape[:2]

    for i in range(len(obj_ids_np)):
        m = _to_numpy(masks[i]).astype(bool)
        if m.shape != (H, W):
            continue
        oid = int(obj_ids_np[i])
        prob = float(probs_np[i]) if probs_np is not None else 0.0
        key = (canonical, oid)
        track = tracks.get(key)
        if track is None:
            track = _Track(masklet_id=str(uuid.uuid4())[:12], cls=canonical)
            tracks[key] = track
        mask_path = _save_mask(masks_2d_base_dir, track.masklet_id, camera_seq, m)
        track.frames_present.append(camera_seq)
        track.mask_paths.append(mask_path)
        track.score = max(track.score, prob)
        if dino_every_k > 0 and frame_index % dino_every_k == 0:
            # DINOv2 (torch.hub) is independent of SAM 3.1 and only feeds the
            # downstream tracking ReID — a failure here (e.g. the torch.hub
            # "No module named 'data'" sys.path issue) must NOT abort the
            # camera's SAM tracking. Isolate it: drop the embedding, keep masks.
            try:
                feat = extract_dino_feature(image_rgb, m, dino_model, device)
            except Exception as exc:  # noqa: BLE001
                feat = None
                log.warning(
                    "DINOv2 embedding failed at frame %d (%s) — keeping mask, "
                    "skipping embedding.",
                    frame_index,
                    exc,
                )
            if feat is not None:
                track.dino_accum.append(feat)


def _save_mask(
    base_dir: str, masklet_id: str, camera_seq: int, mask: np.ndarray
) -> str:
    from PIL import Image as PILImage

    d = os.path.join(base_dir, masklet_id)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{camera_seq:06d}.png")
    PILImage.fromarray(mask.astype(np.uint8) * 255).save(path)
    return path


def _track_to_masklet(
    track: _Track, bag_id: str, chunk_id: str, cam_id: str
) -> Masklet:
    dino = (
        np.mean(track.dino_accum, axis=0).astype(np.float32)
        if track.dino_accum
        else None
    )
    # frames_present / mask_paths are appended in stream order (already
    # increasing for forward propagation); re-sort the pair by camera_seq so
    # the masklet is time-ordered regardless of yield order.
    order = sorted(range(len(track.frames_present)), key=track.frames_present.__getitem__)
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
        tracker_backend="sam3",
    )
