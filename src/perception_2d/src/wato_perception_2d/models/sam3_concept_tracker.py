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

import logging
import os
import time
import uuid
from typing import Optional

import numpy as np

from wato_perception_2d.fusion.masklet import Masklet
from wato_perception_2d.io import CameraFrameInfo
from wato_perception_2d.models.reid import extract_dino_feature

log = logging.getLogger(__name__)


class _Track:
    """Accumulator for one (concept, obj_id) across frames."""

    __slots__ = ("masklet_id", "cls", "score", "frames_present", "mask_paths", "dino_accum")

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


def _start_session(predictor, pil_frames: list) -> str:
    """Start a multiplex video session on an in-memory list of PIL frames.

    The shared Sam3BasePredictor.start_session passes init_state kwargs
    (offload_state_to_cpu / video_loader_type) that the multiplex model's
    init_state rejects, so we call init_state directly with the args it accepts
    and register the session in the predictor's state map ourselves. The
    add_prompt / propagate paths filter kwargs by signature, so only
    start_session needs this workaround.
    """
    inference_state = predictor.model.init_state(
        resource_path=pil_frames,
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
) -> list[Masklet]:
    """Track every concept across one camera's frames → list of Masklets.

    Args:
        predictor: a Sam3MultiplexVideoPredictor (from get_sam3_predictor()).
        frames: CameraFrameInfo sorted by camera_seq.
        images: (H, W, 3) uint8 RGB arrays, parallel to `frames`.
        concept_prompts: list of (text_prompt, canonical_class) — one per
            taxonomy class. text_prompt seeds SAM 3.1; canonical_class is the cls.
        masks_2d_base_dir: where per-masklet PNG dirs are written.

    Returns one Masklet per (concept, obj_id) that the predictor tracked.
    """
    if not frames or not images:
        return []

    from PIL import Image as PILImage

    pil_frames = [PILImage.fromarray(img) for img in images]
    seq_by_index = [f.camera_seq for f in frames]
    n_frames = len(pil_frames)

    tracks: dict[tuple[str, int], _Track] = {}

    sid: Optional[str] = None
    try:
        sid = _start_session(predictor, pil_frames)

        for text_prompt, canonical in concept_prompts:
            predictor.handle_request({"type": "reset_session", "session_id": sid})

            outputs_by_frame: dict[int, dict] = {}
            resp = predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": sid,
                    "frame_index": 0,
                    "text": text_prompt,
                    "output_prob_thresh": output_prob_thresh,
                }
            )
            outputs_by_frame[int(resp["frame_index"])] = resp["outputs"]

            for r in predictor.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": sid,
                    "propagation_direction": "forward",
                    "start_frame_index": 0,
                    "output_prob_thresh": output_prob_thresh,
                }
            ):
                outputs_by_frame[int(r["frame_index"])] = r["outputs"]

            for fi in sorted(outputs_by_frame):
                if not (0 <= fi < n_frames):
                    continue
                _accumulate_frame(
                    tracks,
                    canonical,
                    fi,
                    seq_by_index[fi],
                    outputs_by_frame[fi],
                    image_rgb=images[fi],
                    masks_2d_base_dir=masks_2d_base_dir,
                    dino_model=dino_model,
                    dino_every_k=dino_every_k,
                    device=device,
                )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "chunk %s / %s: SAM 3.1 tracking failed (%s) — keeping %d tracks so far.",
            chunk_id, cam_id, exc, len(tracks),
        )
    finally:
        if sid is not None:
            try:
                predictor.handle_request({"type": "close_session", "session_id": sid})
            except Exception:  # noqa: BLE001
                pass

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
            feat = extract_dino_feature(image_rgb, m, dino_model, device)
            if feat is not None:
                track.dino_accum.append(feat)


def _save_mask(base_dir: str, masklet_id: str, camera_seq: int, mask: np.ndarray) -> str:
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
    return Masklet(
        masklet_id=track.masklet_id,
        bag_id=bag_id,
        chunk_id=chunk_id,
        cam_id=cam_id,
        cls=track.cls,
        score=track.score,
        frames_present=track.frames_present,
        mask_paths=track.mask_paths,
        dino_feature=dino,
        tracker_backend="sam3",
    )
