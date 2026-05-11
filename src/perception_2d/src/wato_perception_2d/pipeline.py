"""perception_2d pipeline orchestrator.

Per-chunk steps (mirrors the lidar_preprocessing chunk-parallel pattern):
  A. For each camera, detect objects per frame (GroundingDINO).
  B. For each camera, segment detected objects (SAM2) with optional LiDAR
     cross-modal point prompts (SAM4D-style).
  C. For each camera, track masks across frames (IoU tracker + DINOv2 embeds).
  D. Cross-camera merge: assign global_object_id by 3D proximity of centroid
     back-projections (uses project_points from wato_common.geometry).
  E. Write detections_2d.parquet and tracklets_2d.parquet.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from typing import Optional

import numpy as np

from wato_common.artifact_store import (
    chunks_index_path,
    detections_2d_path,
    ensure_local_dir,
    local_path,
    masks_2d_dir,
    tracklets_2d_path,
)
from wato_common.geometry import unflatten_se3
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import MASKLET_SCHEMA, MaskletRow, encode_int_list, encode_str_list
from wato_perception_2d.config import ComponentConfig
from wato_perception_2d.cross_cam_merge import merge_cross_camera
from wato_perception_2d.detector import GroundingDINODetector
from wato_perception_2d.io import (
    CameraFrameInfo,
    CalibrationInfo,
    load_calibration,
    load_chunks,
    load_dynamic_lidar_points,
    load_frame_index,
)
from wato_perception_2d.segmenter import SAM2Segmenter
from wato_perception_2d.tracker_2d import Masklet, Tracker2D

log = logging.getLogger(__name__)


def _load_image(path: str) -> Optional[np.ndarray]:
    """Return (H, W, 3) uint8 RGB array, or None on error."""
    try:
        from PIL import Image as PILImage
        img = PILImage.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.array(img, dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to load image %s: %s", path, exc)
        return None


def _project_lidar_prompts(
    lidar_world_pts: Optional[np.ndarray],
    K: np.ndarray,
    cam_T_world: np.ndarray,
    max_points: int,
) -> Optional[np.ndarray]:
    """Project world-frame LiDAR points into image space for cross-modal prompts.

    Returns (M, 2) float32 pixel coordinates of visible points, or None.
    """
    if lidar_world_pts is None or lidar_world_pts.shape[0] == 0:
        return None
    from wato_common.geometry import project_points
    pix, valid = project_points(lidar_world_pts, K, cam_T_world)
    if not valid.any():
        return None
    pix_valid = pix[valid].astype(np.float32)
    if pix_valid.shape[0] > max_points:
        idx = np.random.choice(pix_valid.shape[0], max_points, replace=False)
        pix_valid = pix_valid[idx]
    return pix_valid


def _chunk_complete(bag_id: str, chunk_id: str) -> bool:
    """True when both output parquets exist for this chunk."""
    return (
        os.path.exists(local_path(detections_2d_path(bag_id, chunk_id)))
        and os.path.exists(local_path(tracklets_2d_path(bag_id, chunk_id)))
    )


def _process_chunk(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
    detector: GroundingDINODetector,
    segmenter: SAM2Segmenter,
) -> None:
    """Run steps A–E for one chunk."""
    log.info("perception_2d: chunk %s", chunk_id)

    frames = load_frame_index(bag_id, chunk_id)
    if not frames:
        log.warning("chunk %s: empty frame index — skipping", chunk_id)
        _write_empty(bag_id, chunk_id)
        return

    calibration = load_calibration(bag_id)
    text_prompts = cfg.text_prompts()
    masks_base = ensure_local_dir(masks_2d_dir(bag_id, chunk_id))

    # Group frames by camera.
    frames_by_cam: dict[str, list[CameraFrameInfo]] = defaultdict(list)
    for f in frames:
        frames_by_cam[f.cam_id].append(f)
    # Sort each camera's frames by sequence number.
    for cam_id in frames_by_cam:
        frames_by_cam[cam_id].sort(key=lambda f: f.camera_seq)

    all_masklets: list[Masklet] = []

    for cam_id, cam_frames in frames_by_cam.items():
        calib = calibration.get(cam_id)
        if calib is None:
            log.warning("chunk %s: no calibration for camera %s — skipping", chunk_id, cam_id)
            continue

        tracker = Tracker2D(
            bag_id=bag_id,
            chunk_id=chunk_id,
            cam_id=cam_id,
            masks_2d_base_dir=masks_base,
            dino_model=cfg.reid_features.model,
            dino_every_k=cfg.reid_features.every_k_frames,
        )

        for frame in cam_frames:
            image = _load_image(frame.image_path)
            if image is None:
                continue
            H, W = image.shape[:2]

            # SAM4D cross-modal: project LiDAR dynamic points as SAM2 prompts.
            lidar_prompts = None
            if cfg.use_lidar_prompts and frame.valid_pose and frame.world_T_ego_flat:
                world_T_ego = unflatten_se3(frame.world_T_ego_flat)
                from wato_common.geometry import invert_se3
                ego_T_cam = calib.ego_T_cam
                cam_T_ego = invert_se3(ego_T_cam)
                cam_T_world = cam_T_ego @ invert_se3(world_T_ego)
                lidar_pts = load_dynamic_lidar_points(bag_id, chunk_id, frame.sweep_id)
                lidar_prompts = _project_lidar_prompts(
                    lidar_pts, calib.K, cam_T_world, cfg.lidar_prompt_max_points
                )

            # Step A: detect.
            detections = detector.detect(
                image, text_prompts, box_threshold=cfg.detector_score_threshold
            )
            # Map raw synonym labels to canonical class names.
            for det in detections:
                det.class_name = cfg.class_from_synonym(det.class_name)

            # Step B: segment.
            seg_detections = segmenter.segment(image, detections, lidar_prompts)

            # Step C: track frame.
            tracker.update(frame.camera_seq, image, seg_detections)

        cam_masklets = tracker.finalize()
        log.info(
            "chunk %s / %s: %d masklets from %d frames",
            chunk_id, cam_id, len(cam_masklets), len(cam_frames),
        )
        all_masklets.extend(cam_masklets)

    # Step D: cross-camera merge using 3D centroid proximity.
    if len(frames_by_cam) > 1 and all_masklets:
        # Build per-camera world_T_ego from the last valid frame of each camera.
        world_T_ego_by_cam: dict[str, np.ndarray] = {}
        for cam_id, cam_frames in frames_by_cam.items():
            for f in reversed(cam_frames):
                if f.valid_pose and f.world_T_ego_flat:
                    world_T_ego_by_cam[cam_id] = unflatten_se3(f.world_T_ego_flat)
                    break

        # Use the last sweep's dynamic points as LiDAR for depth estimation.
        last_sweep_id = max(f.sweep_id for f in frames)
        lidar_world_pts = load_dynamic_lidar_points(bag_id, chunk_id, last_sweep_id)

        all_masklets = merge_cross_camera(
            all_masklets,
            calibration,
            world_T_ego_by_cam,
            lidar_world_pts,
            radius_m=cfg.cross_camera_match_radius_m,
        )

    # Step E: write output parquets.
    _write_masklets(bag_id, chunk_id, all_masklets)
    log.info("chunk %s: wrote %d masklets", chunk_id, len(all_masklets))


def _masklet_to_row(mkl: Masklet) -> dict:
    dino_path: Optional[str] = None
    if mkl.dino_feature is not None:
        dino_dir = os.path.dirname(mkl.mask_paths[0]) if mkl.mask_paths else ""
        if dino_dir:
            dino_path = os.path.join(dino_dir, "dino_feature.npy")
            np.save(dino_path, mkl.dino_feature)

    return MaskletRow(
        masklet_id=mkl.masklet_id,
        bag_id=mkl.bag_id,
        chunk_id=mkl.chunk_id,
        cam_id=mkl.cam_id,
        cls=mkl.cls,
        score=mkl.score,
        frames_present=encode_int_list(mkl.frames_present),
        mask_path=mkl.mask_paths[0] if mkl.mask_paths else "",
        dino_feature_path=dino_path,
        global_object_id=mkl.global_object_id,
    ).model_dump()


def _write_masklets(bag_id: str, chunk_id: str, masklets: list[Masklet]) -> None:
    rows = [_masklet_to_row(m) for m in masklets]
    write_table(rows, MASKLET_SCHEMA, detections_2d_path(bag_id, chunk_id))
    write_table(rows, MASKLET_SCHEMA, tracklets_2d_path(bag_id, chunk_id))


def _write_empty(bag_id: str, chunk_id: str) -> None:
    write_table([], MASKLET_SCHEMA, detections_2d_path(bag_id, chunk_id))
    write_table([], MASKLET_SCHEMA, tracklets_2d_path(bag_id, chunk_id))


def run(
    cfg: ComponentConfig,
    *,
    bag_id: str,
    chunk_id: Optional[str] = None,
    force: bool = False,
) -> None:
    """Process one bag (or one chunk) end-to-end.

    Skips chunks whose output parquets already exist unless force=True.
    """
    chunks = load_chunks(bag_id)
    if not chunks:
        raise FileNotFoundError(
            f"No chunks found for bag {bag_id!r} — run ingest first."
        )

    if chunk_id:
        chunks = [c for c in chunks if c["chunk_id"] == chunk_id]
        if not chunks:
            raise ValueError(f"chunk_id {chunk_id!r} not found for bag {bag_id!r}")

    # Build models once and reuse across chunks.
    detector = GroundingDINODetector(device=None)
    segmenter = SAM2Segmenter(checkpoint=cfg.sam2_checkpoint, device=None)

    n_ok = n_skip = 0
    for chunk in chunks:
        cid = chunk["chunk_id"]
        if not force and _chunk_complete(bag_id, cid):
            log.info("chunk %s already complete — skipping (use force=True to re-run)", cid)
            n_skip += 1
            continue
        try:
            _process_chunk(cfg, bag_id, cid, detector, segmenter)
            n_ok += 1
        except Exception:  # noqa: BLE001
            log.exception("perception_2d: chunk %s failed", cid)

    log.info(
        "perception_2d complete for bag %s: %d processed, %d skipped",
        bag_id, n_ok, n_skip,
    )
