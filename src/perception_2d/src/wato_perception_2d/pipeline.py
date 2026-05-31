"""perception_2d pipeline orchestrator.

Per chunk, per camera:
  Depth branch (per frame):
    1. DepthAnythingV2.infer() → relative_depth
    2. Load static LiDAR points for this sweep
    3. depth_align.build_anchor_pairs() → d_lidar, d_da
    4. depth_align.ransac_affine_fit() → affine params
    5. depth_align.apply_affine() → metric_depth
    6. Write depth_2d/<cam>/<frame>.npz (skipped on total fit failure)
  Tracking branch (whole camera stream):
    7. SAM 3.1 multiplex concept-video predictor: for each taxonomy concept,
       detect + segment + track every instance → Masklets with persistent IDs.
  After all cameras:
    8. cross_cam_merge() using depth_2d artifacts
    9. Write detections_2d.parquet + tracklets_2d.parquet

SAM 3.1 is the only tracker; if its predictor can't load, the chunk writes empty
outputs and logs an error (no hand-rolled fallback).
"""

from __future__ import annotations

import logging
import os
import pickle
import shutil
from collections import defaultdict, deque
from typing import Optional

from tqdm.auto import tqdm

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
from wato_perception_2d.fusion.cross_cam_merge import merge_cross_camera
from wato_perception_2d.fusion.depth_align import apply_affine, build_anchor_pairs, ransac_affine_fit
from wato_perception_2d.fusion.masklet import Masklet
from wato_perception_2d.io import (
    CalibrationInfo,
    CameraFrameInfo,
    clear_lidar_caches,
    load_calibration,
    load_chunks,
    load_frame_index,
    load_static_lidar_points,
)
from wato_perception_2d.models._sam3_runtime import get_sam3_predictor
from wato_perception_2d.models.depth import DepthAnythingV2
from wato_perception_2d.models.sam3_concept_tracker import track_camera_concepts

log = logging.getLogger(__name__)


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


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


def _run_depth_branch(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
    cam_id: str,
    frame: CameraFrameInfo,
    image: np.ndarray,
    calib: CalibrationInfo,
    depth_model: Optional[DepthAnythingV2],
    fallback_window: deque,
) -> None:
    """Estimate + LiDAR-align metric depth for one frame and write the artifact.

    Writes nothing on total failure (fit_status==2 / depth disabled): downstream
    treats a missing depth_2d artifact identically to a written fit_status==2 one.
    """
    if not (cfg.depth.enabled and depth_model is not None):
        return
    H, W = image.shape[:2]
    fit_params: dict = {"a": 1.0, "b": 0.0, "n_inliers": 0, "rmse_inliers_m": 0.0, "fit_status": 2}

    rel_depth = depth_model.infer(image)
    static_pts = load_static_lidar_points(bag_id, chunk_id, frame.sweep_id)

    if static_pts is not None and frame.valid_pose and frame.world_T_ego_flat:
        world_T_ego = unflatten_se3(frame.world_T_ego_flat)
        cam_T_world = invert_se3(calib.ego_T_cam) @ invert_se3(world_T_ego)
        d_lidar, d_da = build_anchor_pairs(
            static_pts, rel_depth, calib.K, cam_T_world, (W, H),
            sky_mask_top_fraction=cfg.depth.sky_mask_top_fraction,
        )
        fallback = fallback_window[-1] if fallback_window else None
        fit_params = ransac_affine_fit(
            d_lidar, d_da,
            n_iter=cfg.depth.ransac_n_iter,
            inlier_threshold_m=cfg.depth.ransac_inlier_threshold_m,
            min_anchors=cfg.depth.min_lidar_anchors,
            fallback=fallback,
        )
        fit_params["n_anchors"] = int(len(d_lidar))
        if fit_params["fit_status"] == 0:
            fallback_window.append((fit_params["a"], fit_params["b"]))

    if fit_params["fit_status"] != 2:
        metric_depth = apply_affine(
            rel_depth, fit_params["a"], fit_params["b"], out_dtype=cfg.depth.output_dtype
        )
        _write_depth_artifact(
            bag_id, chunk_id, cam_id, frame.camera_seq, metric_depth, fit_params
        )


def _partial_dir(bag_id: str, chunk_id: str) -> str:
    """Directory holding per-camera resume pickles for this chunk."""
    masks_base = local_path(masks_2d_dir(bag_id, chunk_id))
    return os.path.join(os.path.dirname(masks_base), ".perception_2d_partial")


def _cam_partial_path(bag_id: str, chunk_id: str, cam_id: str) -> str:
    return os.path.join(_partial_dir(bag_id, chunk_id), f"{cam_id}.pkl")


def _load_cam_partial(
    bag_id: str, chunk_id: str, cam_id: str
) -> Optional[list[Masklet]]:
    """Return cached per-camera masklets if a usable partial exists.

    A zero-length cached list is treated as "no usable cache" rather than "done":
    producing zero masklets is almost always a silent failure (model unavailable),
    and trusting it would let one bad run permanently skip a camera on resume.
    """
    p = _cam_partial_path(bag_id, chunk_id, cam_id)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "rb") as f:
            cached = pickle.load(f)
    except Exception as exc:  # noqa: BLE001
        log.warning("partial cache at %s unreadable (%s) — discarding", p, exc)
        try:
            os.remove(p)
        except OSError:
            pass
        return None
    if not cached:
        return None
    return cached


def _save_cam_partial(
    bag_id: str, chunk_id: str, cam_id: str, masklets: list[Masklet]
) -> None:
    """Persist per-camera masklets so abort doesn't lose this camera's work.

    Skips writing an empty result (a likely silent failure) so the next run
    re-attempts this camera rather than trusting an empty cache.
    """
    if not masklets:
        log.info(
            "%s produced 0 masklets — skipping partial save so the next run "
            "re-attempts this camera instead of trusting an empty cache",
            cam_id,
        )
        return
    p = _cam_partial_path(bag_id, chunk_id, cam_id)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(masklets, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, p)


def _process_chunk(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
    predictor,
    depth_model: Optional[DepthAnythingV2],
    device: str,
    *,
    force: bool = False,
) -> None:
    """Run all pipeline steps for one chunk."""
    frames = load_frame_index(bag_id, chunk_id)
    if not frames:
        log.warning("chunk %s: empty frame index — skipping", chunk_id)
        _write_empty(bag_id, chunk_id)
        return

    if predictor is None:
        log.error(
            "chunk %s: SAM 3.1 predictor unavailable — writing empty outputs. "
            "Install the `sam3` package and fetch facebook/sam3.1.", chunk_id,
        )
        _write_empty(bag_id, chunk_id)
        return

    calibration = load_calibration(bag_id)
    concepts = cfg.concept_prompts()
    masks_base = ensure_local_dir(masks_2d_dir(bag_id, chunk_id))

    partial_dir = _partial_dir(bag_id, chunk_id)
    if force and os.path.isdir(partial_dir):
        shutil.rmtree(partial_dir)

    frames_by_cam: dict[str, list[CameraFrameInfo]] = defaultdict(list)
    for f in frames:
        frames_by_cam[f.cam_id].append(f)
    for cam_id in frames_by_cam:
        frames_by_cam[cam_id].sort(key=lambda f: f.camera_seq)

    log.info(
        "chunk %s: %d frames across %d cameras; %d concepts; depth=%s",
        chunk_id, len(frames), len(frames_by_cam), len(concepts),
        "on" if cfg.depth.enabled else "off",
    )

    all_masklets: list[Masklet] = []

    for cam_id, cam_frames in frames_by_cam.items():
        calib = calibration.get(cam_id)
        if calib is None:
            log.warning("chunk %s: no calibration for %s — skipping", chunk_id, cam_id)
            continue

        cached = _load_cam_partial(bag_id, chunk_id, cam_id)
        if cached is not None:
            log.info(
                "chunk %s / %s: resuming %d cached masklets — skipping reprocessing",
                chunk_id, cam_id, len(cached),
            )
            all_masklets.extend(cached)
            continue

        log.info("chunk %s / %s: starting (%d frames)", chunk_id, cam_id, len(cam_frames))

        # Load images + run the per-frame depth branch; collect valid frames for
        # the whole-stream SAM 3.1 tracking pass.
        fallback_window: deque[tuple[float, float]] = deque(
            maxlen=cfg.depth.fallback_window
        )
        valid_frames: list[CameraFrameInfo] = []
        valid_images: list[np.ndarray] = []
        for frame in tqdm(cam_frames, desc=f"{chunk_id}/{cam_id} depth", unit="frame", leave=False):
            image = _load_image(frame.image_path)
            if image is None:
                continue
            valid_frames.append(frame)
            valid_images.append(image)
            _run_depth_branch(
                cfg, bag_id, chunk_id, cam_id, frame, image, calib, depth_model, fallback_window
            )

        # SAM 3.1 multiplex concept-video tracking over the whole camera stream.
        cam_masklets = track_camera_concepts(
            predictor,
            valid_frames,
            valid_images,
            concepts,
            bag_id=bag_id,
            chunk_id=chunk_id,
            cam_id=cam_id,
            masks_2d_base_dir=masks_base,
            dino_model=cfg.reid.model,
            dino_every_k=cfg.reid.every_k_frames,
            device=device,
            output_prob_thresh=cfg.segmentation.output_prob_thresh,
        )
        log.info(
            "chunk %s / %s: %d masklets from %d frames",
            chunk_id, cam_id, len(cam_masklets), len(valid_frames),
        )
        _save_cam_partial(bag_id, chunk_id, cam_id, cam_masklets)
        all_masklets.extend(cam_masklets)

    # Cross-camera merge using depth_2d artifacts for back-projection depth.
    if len(frames_by_cam) > 1 and all_masklets:
        # Pose indexed by (cam, frame) so the cross-cam lift uses the ego pose at
        # the masklet's actual last frame, not some later frame.
        pose_by_cam_seq: dict[tuple[str, int], np.ndarray] = {
            (f.cam_id, f.camera_seq): unflatten_se3(f.world_T_ego_flat)
            for f in frames
            if f.valid_pose and f.world_T_ego_flat
        }
        all_masklets = merge_cross_camera(
            all_masklets,
            calibration,
            pose_by_cam_seq,
            bag_id=bag_id,
            chunk_id=chunk_id,
            radius_m=cfg.cross_cam.match_radius_m,
        )

    _write_masklets(bag_id, chunk_id, all_masklets, cfg.synonym_to_class_map())
    log.info("chunk %s: wrote %d masklets", chunk_id, len(all_masklets))

    # Final parquets committed — the per-camera resume cache is now redundant.
    if os.path.isdir(partial_dir):
        shutil.rmtree(partial_dir, ignore_errors=True)

    # Release this chunk's cached lidar index/points before moving on.
    clear_lidar_caches(bag_id, chunk_id)


def _masklet_to_row(mkl: Masklet, syn2cls: dict[str, str]) -> dict:
    # mask_path is the masklet's per-frame-PNG directory (schema contract),
    # not a single file. dino_feature.npy lands in the same directory.
    mask_dir = os.path.dirname(mkl.mask_paths[0]) if mkl.mask_paths else ""

    dino_path: Optional[str] = None
    if mkl.dino_feature is not None and mask_dir:
        dino_path = os.path.join(mask_dir, "dino_feature.npy")
        np.save(dino_path, mkl.dino_feature)

    # cls is canonicalised to the taxonomy; raw detected phrase preserved.
    canonical = syn2cls.get(mkl.cls.lower(), mkl.cls)

    return MaskletRow(
        masklet_id=mkl.masklet_id,
        bag_id=mkl.bag_id,
        chunk_id=mkl.chunk_id,
        cam_id=mkl.cam_id,
        cls=canonical,
        score=mkl.score,
        frames_present=encode_int_list(mkl.frames_present),
        mask_path=mask_dir,
        dino_feature_path=dino_path,
        global_object_id=mkl.global_object_id,
        raw_phrase=mkl.cls,
        sam3_score=mkl.score,
        tracker_backend=mkl.tracker_backend,
    ).model_dump()


def _write_masklets(
    bag_id: str, chunk_id: str, masklets: list[Masklet], syn2cls: dict[str, str]
) -> None:
    rows = [_masklet_to_row(m, syn2cls) for m in masklets]
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
    if not os.path.exists(cfg.prompts_path):
        log.warning(
            "prompts.yaml not found at %s — using the hardcoded fallback concept "
            "list. Mount the taxonomy file or set prompts_path.", cfg.prompts_path,
        )

    chunks = load_chunks(bag_id)
    if not chunks:
        raise FileNotFoundError(
            f"No chunks found for bag {bag_id!r} — run ingest first."
        )

    if chunk_id:
        chunks = [c for c in chunks if c["chunk_id"] == chunk_id]
        if not chunks:
            raise ValueError(f"chunk_id {chunk_id!r} not found for bag {bag_id!r}")

    log.info("=" * 70)
    log.info(
        "perception_2d: bag=%s, %d chunk(s) to process, force=%s",
        bag_id, len(chunks), force,
    )
    log.info("=" * 70)

    device = _default_device()
    # Build models once and reuse across chunks.
    predictor = get_sam3_predictor(
        version=cfg.segmentation.version, use_fa3=cfg.segmentation.use_fa3
    )
    depth_model: Optional[DepthAnythingV2] = None
    if cfg.depth.enabled:
        depth_model = DepthAnythingV2(model_size=cfg.depth.model, device=None)

    n_ok = n_skip = 0
    for chunk in chunks:
        cid = chunk["chunk_id"]
        if not force and _chunk_complete(bag_id, cid):
            log.info("chunk %s already complete — skipping (use force=True)", cid)
            n_skip += 1
            continue
        try:
            _process_chunk(cfg, bag_id, cid, predictor, depth_model, device, force=force)
            n_ok += 1
        except Exception:  # noqa: BLE001
            log.exception("perception_2d: chunk %s failed", cid)

    log.info(
        "perception_2d complete for bag %s: %d processed, %d skipped",
        bag_id, n_ok, n_skip,
    )
