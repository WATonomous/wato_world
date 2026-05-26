"""perception_2d v2 pipeline orchestrator.

Per-chunk steps:
  Image branch (per camera, per frame):
    1. Florence-2 → RegionProposals
    2. Filter proposals against prompts.yaml allowlist
    3. SAM3.segment() text-prompted → SegmentedDetections
    4. phrase_dedup.coarsen_synonyms() → deduped detections
  Depth branch (per camera, per frame, parallel with image branch):
    5. DepthAnythingV2.infer() → relative_depth, confidence
    6. Load static LiDAR points for this sweep
    7. depth_align.build_anchor_pairs() → d_lidar, d_da
    8. depth_align.ransac_affine_fit() → affine params
    9. depth_align.apply_affine() → metric_depth (float16)
   10. Write depth_2d/<cam>/<frame>.npz
  Converge:
   11. phrase_dedup.nms_3d() using metric_depth (or nms_2d_fallback)
   12. tracker.update()
  After all frames:
   13. tracker.finalize() → cam_masklets
   14. cross_cam_merge() using depth_2d artifacts
   15. Write detections_2d.parquet + tracklets_2d.parquet
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict, deque
from typing import Optional

import numpy as np

from wato_common.artifact_store import (
    depth_2d_path,
    detections_2d_path,
    ensure_local_dir,
    local_path,
    masks_2d_dir,
    tracklets_2d_path,
)
from wato_common.geometry import invert_se3, unflatten_se3
from wato_common.io.parquet_io import write_table
from wato_common.schemas import MASKLET_SCHEMA, MaskletRow, encode_int_list
from wato_perception_2d.config import ComponentConfig
from wato_perception_2d.cross_cam_merge import merge_cross_camera
from wato_perception_2d.depth import DepthAnythingV2
from wato_perception_2d.depth_align import apply_affine, build_anchor_pairs, ransac_affine_fit
from wato_perception_2d.discovery import Florence2Discovery
from wato_perception_2d.io import (
    CameraFrameInfo,
    load_calibration,
    load_chunks,
    load_frame_index,
    load_static_lidar_points,
)
from wato_perception_2d.phrase_dedup import coarsen_synonyms, nms_2d_fallback, nms_3d
from wato_perception_2d.sam3_tracker import SAM3Tracker
from wato_perception_2d.segmenter import SAM3Segmenter
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


def _chunk_complete(bag_id: str, chunk_id: str) -> bool:
    """True when both output parquets exist for this chunk."""
    return os.path.exists(
        local_path(detections_2d_path(bag_id, chunk_id))
    ) and os.path.exists(local_path(tracklets_2d_path(bag_id, chunk_id)))


def _write_depth_artifact(
    bag_id: str,
    chunk_id: str,
    cam_id: str,
    frame_seq: int,
    metric_depth: np.ndarray,
    fit_params: dict,
) -> None:
    out_path = local_path(depth_2d_path(bag_id, chunk_id, cam_id, frame_seq))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        depth_m=metric_depth,
        affine_a=np.float32(fit_params["a"]),
        affine_b=np.float32(fit_params["b"]),
        n_anchors=np.int32(fit_params.get("n_anchors", 0)),
        n_inliers=np.int32(fit_params["n_inliers"]),
        rmse_inliers_m=np.float32(fit_params["rmse_inliers_m"]),
        fit_status=np.int32(fit_params["fit_status"]),
    )


def _process_chunk(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
    discovery: Florence2Discovery,
    segmenter: SAM3Segmenter,
    depth_model: Optional[DepthAnythingV2],
) -> None:
    """Run all pipeline steps for one chunk."""
    log.info("perception_2d: chunk %s", chunk_id)

    frames = load_frame_index(bag_id, chunk_id)
    if not frames:
        log.warning("chunk %s: empty frame index — skipping", chunk_id)
        _write_empty(bag_id, chunk_id)
        return

    calibration = load_calibration(bag_id)
    text_prompt_set = set(cfg.text_prompts())
    masks_base = ensure_local_dir(masks_2d_dir(bag_id, chunk_id))

    frames_by_cam: dict[str, list[CameraFrameInfo]] = defaultdict(list)
    for f in frames:
        frames_by_cam[f.cam_id].append(f)
    for cam_id in frames_by_cam:
        frames_by_cam[cam_id].sort(key=lambda f: f.camera_seq)

    all_masklets: list[Masklet] = []

    for cam_id, cam_frames in frames_by_cam.items():
        calib = calibration.get(cam_id)
        if calib is None:
            log.warning("chunk %s: no calibration for %s — skipping", chunk_id, cam_id)
            continue

        if cfg.tracker.backend == "sam3":
            tracker: SAM3Tracker | Tracker2D = SAM3Tracker(
                bag_id=bag_id,
                chunk_id=chunk_id,
                cam_id=cam_id,
                masks_2d_base_dir=masks_base,
                checkpoint=cfg.segmentation.checkpoint,
                dino_model=cfg.reid.model,
                dino_every_k=cfg.reid.every_k_frames,
            )
            tracker.reset()
        else:
            tracker = Tracker2D(
                bag_id=bag_id,
                chunk_id=chunk_id,
                cam_id=cam_id,
                masks_2d_base_dir=masks_base,
                dino_model=cfg.reid.model,
                dino_every_k=cfg.reid.every_k_frames,
            )

        # Rolling fallback affine params for depth alignment.
        fallback_window: deque[tuple[float, float]] = deque(
            maxlen=cfg.depth.fallback_window
        )

        for frame in cam_frames:
            image = _load_image(frame.image_path)
            if image is None:
                continue
            H, W = image.shape[:2]

            # --- Image branch ---
            proposals = discovery.propose(image)
            # Filter to prompts allowlist (case-insensitive substring match).
            if text_prompt_set:
                proposals = [
                    p for p in proposals
                    if any(kw in p.phrase.lower() for kw in text_prompt_set)
                    and p.confidence >= cfg.discovery.min_confidence
                ]

            seg_detections = segmenter.segment(image, proposals)
            seg_detections = [
                sd for sd in seg_detections
                if sd.sam3_score >= cfg.segmentation.min_score
            ]
            seg_detections = coarsen_synonyms(
                seg_detections, cfg.phrase_dedup.synonym_clip_threshold
            )

            # --- Depth branch ---
            metric_depth = np.zeros((H, W), dtype=np.float16)
            fit_params: dict = {"a": 1.0, "b": 0.0, "n_inliers": 0, "rmse_inliers_m": 0.0, "fit_status": 2}

            if cfg.depth.enabled and depth_model is not None:
                rel_depth, confidence = depth_model.infer(image)
                static_pts = load_static_lidar_points(bag_id, chunk_id, frame.sweep_id)

                if static_pts is not None and frame.valid_pose and frame.world_T_ego_flat:
                    world_T_ego = unflatten_se3(frame.world_T_ego_flat)
                    cam_T_ego = invert_se3(calib.ego_T_cam)
                    cam_T_world = cam_T_ego @ invert_se3(world_T_ego)

                    d_lidar, d_da = build_anchor_pairs(
                        static_pts,
                        rel_depth,
                        calib.K,
                        cam_T_world,
                        (W, H),
                        sky_mask_top_fraction=cfg.depth.sky_mask_top_fraction,
                    )

                    fallback = fallback_window[-1] if fallback_window else None
                    fit_params = ransac_affine_fit(
                        d_lidar,
                        d_da,
                        n_iter=cfg.depth.ransac_n_iter,
                        inlier_threshold_m=cfg.depth.ransac_inlier_threshold_m,
                        fallback=fallback,
                    )
                    fit_params["n_anchors"] = int(len(d_lidar))
                    if fit_params["fit_status"] == 0:
                        fallback_window.append((fit_params["a"], fit_params["b"]))

                metric_depth = apply_affine(rel_depth, fit_params["a"], fit_params["b"])

            _write_depth_artifact(bag_id, chunk_id, cam_id, frame.camera_seq, metric_depth, fit_params)

            # --- Converge: 3D NMS then track ---
            if fit_params["fit_status"] < 2 and frame.valid_pose and frame.world_T_ego_flat:
                world_T_ego = unflatten_se3(frame.world_T_ego_flat)
                seg_detections = nms_3d(
                    seg_detections,
                    metric_depth.astype(np.float32),
                    calib.K,
                    world_T_ego,
                    calib.ego_T_cam,
                    radius_m=cfg.phrase_dedup.nms_3d_radius_m,
                )
            else:
                seg_detections = nms_2d_fallback(
                    seg_detections, cfg.phrase_dedup.nms_2d_iou
                )

            tracker.update(frame.camera_seq, image, seg_detections)

        cam_masklets = tracker.finalize()
        log.info(
            "chunk %s / %s: %d masklets from %d frames",
            chunk_id, cam_id, len(cam_masklets), len(cam_frames),
        )
        all_masklets.extend(cam_masklets)

    # Cross-camera merge using depth_2d artifacts for back-projection depth.
    if len(frames_by_cam) > 1 and all_masklets:
        world_T_ego_by_cam: dict[str, np.ndarray] = {}
        for cam_id, cam_frames in frames_by_cam.items():
            for f in reversed(cam_frames):
                if f.valid_pose and f.world_T_ego_flat:
                    world_T_ego_by_cam[cam_id] = unflatten_se3(f.world_T_ego_flat)
                    break

        all_masklets = merge_cross_camera(
            all_masklets,
            calibration,
            world_T_ego_by_cam,
            bag_id=bag_id,
            chunk_id=chunk_id,
            radius_m=cfg.cross_cam.match_radius_m,
        )

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
        raw_phrase=mkl.cls,
        sam3_score=mkl.score,
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
    discovery = Florence2Discovery(
        model_id=cfg.discovery.model, device=None
    )
    segmenter = SAM3Segmenter(checkpoint=cfg.segmentation.checkpoint, device=None)
    depth_model: Optional[DepthAnythingV2] = None
    if cfg.depth.enabled:
        depth_model = DepthAnythingV2(model_size=cfg.depth.model, device=None)

    n_ok = n_skip = 0
    for chunk in chunks:
        cid = chunk["chunk_id"]
        if not force and _chunk_complete(bag_id, cid):
            log.info(
                "chunk %s already complete — skipping (use force=True to re-run)", cid
            )
            n_skip += 1
            continue
        try:
            _process_chunk(cfg, bag_id, cid, discovery, segmenter, depth_model)
            n_ok += 1
        except Exception:  # noqa: BLE001
            log.exception("perception_2d: chunk %s failed", cid)

    log.info(
        "perception_2d complete for bag %s: %d processed, %d skipped",
        bag_id, n_ok, n_skip,
    )
