"""perception_2d pipeline orchestrator.

Each chunk runs as two passes so the heavy GPU models never share VRAM (DA-V2
Large + SAM2 + DINOv2 don't fit together on ~6 GB):

  PASS 1 — depth, per camera: DA-V2 disparity (batched), fitted per frame to
    static LiDAR points projected with the pose at the frame's own time
    (1/d = scale·disp + shift, RANSAC) → depth_2d/<cam>/<frame>.npz. DA-V2 is
    then unloaded.
  PASS 2 — tracking, per camera stream: the detector (detection.backend:
    YOLO26 or GroundingDINO; vocabulary from discovery.backend) boxes
    keyframes, SAM2's video predictor masks and tracks each box, DINOv2 embeds
    each masklet → tracklets_2d.parquet + detections_2d.parquet + mask PNGs.

Model loading and inference are fail-loud: a detector, SAM2 or DINOv2 failure
stops the run rather than leaving a camera with fewer masklets or no
embeddings. SAM2 + the detector are import-checked, and the SAM2, YOLO and
DINOv2 weights load-checked, before the depth pass. Cross-camera identity is
left to the `tracking` component.
"""

from __future__ import annotations

import logging
import os
import pickle
import shutil
from collections import deque
from typing import Optional, Union

from tqdm.auto import tqdm

import numpy as np

from wato_common import manifest as common_manifest
from wato_common.artifact_store import (
    chunk_root,
    depth_2d_dir,
    depth_stats_path,
    detections_2d_path,
    frame_index_path,
    local_path,
    masklet_embeddings_path,
    masks_2d_dir,
    poses_path,
    tracklets_2d_path,
)
from wato_common.artifacts import depth, masklets
from wato_common.artifacts.calibration import CameraCalibration
from wato_common.artifacts import calibration_refinement as camera_calibration
from wato_common.artifacts.frame_index import CameraFrame, FrameIndex
from wato_common.geometry import invert_se3
from wato_perception_2d.config import ComponentConfig
from wato_perception_2d.model_registry import revisions as model_revisions
from wato_perception_2d.fusion.depth_align import (
    build_anchor_pairs,
    disparity_to_depth,
    ransac_disparity_fit,
)
from wato_perception_2d.fusion.masklet import Masklet
from wato_perception_2d.io import (
    clear_lidar_caches,
    load_chunks,
    load_static_lidar_points,
)
from wato_perception_2d import mask_quality
from wato_perception_2d.models._sam2_runtime import (
    check_sam2_cuda_extension,
    get_sam2_predictor,
    propagation_provenance,
    release_sam2_predictor,
    sam2_importable,
)
from wato_perception_2d.models.depth import DepthAnythingV2
from wato_perception_2d.models.detector import (
    GroundingDinoDetector,
    detector_importable,
)
from wato_perception_2d.models.discovery import Florence2Discovery
from wato_perception_2d.models import embeddings
from wato_perception_2d.models.embeddings import check_model as check_dino_model
from wato_perception_2d.models.yolo_detector import (
    YoloDetector,
    check_class_map,
    yolo_importable,
)
from wato_perception_2d.models.sam2_tracker import track_camera

log = logging.getLogger(__name__)

# A (text_prompt, canonical_class) concept fed to the detector.
Concept = tuple[str, str]
# Either box source; both expose detect(image_rgb, concepts) -> list[Detection].
Detector = Union[GroundingDinoDetector, YoloDetector]


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


def _ego_mask(cfg: ComponentConfig, cam_id: str, width: int, height: int):
    """(H, W) bool region of ``cam_id``'s images showing the ego vehicle, or
    None when the camera has no polygons (it doesn't see the car).

    Raises ValueError when the polygons were drawn on a different image size.
    """
    polygons = cfg.ego_mask.polygons.get(cam_id)
    if not polygons:
        return None
    if tuple(cfg.ego_mask.image_size) != (width, height):
        raise ValueError(
            f"ego_mask polygons are drawn on {cfg.ego_mask.image_size[0]}x"
            f"{cfg.ego_mask.image_size[1]} images but {cam_id}'s are "
            f"{width}x{height}"
        )
    from PIL import Image as PILImage
    from PIL import ImageDraw

    canvas = PILImage.new("1", (width, height))
    draw = ImageDraw.Draw(canvas)
    for poly in polygons:
        draw.polygon([(float(x), float(y)) for x, y in poly], fill=1)
    return np.asarray(canvas, dtype=bool)


def _chunk_complete(bag_id: str, chunk_id: str) -> bool:
    """True when both output parquets exist for this chunk."""
    return masklets.exists(bag_id, chunk_id)


def _discover_concepts_florence2(
    discovery: Florence2Discovery,
    images: list[np.ndarray],
    cfg: ComponentConfig,
) -> list[Concept]:
    """Open-vocabulary concept source for one camera stream.

    Runs Florence-2 on every ``discovery.sample_every_k``-th frame, keeps
    phrases scoring >= ``discovery.min_confidence``, and pools them into one
    deduped concept set.  Each phrase is canonicalised through the prompts.yaml
    synonym map (unknown phrases keep their raw text as the canonical class).
    The pooled set becomes the detector's text query for this camera.
    """
    syn2cls = cfg.synonym_to_class_map()
    step = max(1, cfg.discovery.sample_every_k)
    concepts: dict[str, Concept] = {}  # keyed by lowercased phrase
    for img in images[::step]:
        for p in discovery.propose(img):
            if p.confidence < cfg.discovery.min_confidence:
                continue
            phrase = p.phrase.strip()
            if not phrase:
                continue
            key = phrase.lower()
            concepts.setdefault(key, (phrase, syn2cls.get(key, phrase)))
    return list(concepts.values())


def _write_depth_artifact(
    bag_id: str,
    chunk_id: str,
    cam_id: str,
    frame_seq: int,
    metric_depth: np.ndarray,
    fit_params: dict,
) -> None:
    depth.write_frame(
        bag_id,
        chunk_id,
        cam_id,
        frame_seq,
        depth.DepthFrame(
            depth_m=metric_depth,
            disp_scale=fit_params["scale"],
            disp_shift=fit_params["shift"],
            n_anchors=fit_params.get("n_anchors", 0),
            n_inliers=fit_params["n_inliers"],
            rmse_inliers_m=fit_params["rmse_inliers_m"],
            fit_status=depth.FitStatus(fit_params["fit_status"]),
        ),
    )


def _align_and_write_depth(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
    cam_id: str,
    frame: CameraFrame,
    disparity: np.ndarray,
    calib: CameraCalibration,
    fallback_window: deque,
) -> None:
    """LiDAR-align one frame's DA-V2 disparity to metric depth and write it.

    The caller runs DA-V2 (batched); this is the per-frame sequential tail.
    Writes nothing on FitStatus.FAILED (a missing frame means no usable depth).
    The world-frame static points are projected with the frame's own pose
    (CameraFrame.world_T_ego, at the image's time); a frame without one gets
    no anchors.
    """
    H, W = disparity.shape[:2]
    fit_params: dict = {
        "scale": 1.0,
        "shift": 0.0,
        "n_inliers": 0,
        "rmse_inliers_m": 0.0,
        "fit_status": depth.FitStatus.FAILED,
    }

    # Every sweep this image is nearest to (3 LiDARs on the WATO rig). Their
    # static points are world-frame, so they combine directly; the corner
    # scanners add anchors where the centre one doesn't reach.
    per_sweep = [
        load_static_lidar_points(bag_id, chunk_id, sid) for sid in frame.sweep_ids
    ]
    per_sweep = [p for p in per_sweep if p is not None and len(p)]
    static_pts = np.concatenate(per_sweep) if per_sweep else None

    if static_pts is not None and frame.world_T_ego is not None:
        cam_T_world = invert_se3(calib.ego_T_cam) @ invert_se3(frame.world_T_ego)
        d_lidar, disp = build_anchor_pairs(
            static_pts,
            disparity,
            calib.model_for_image((W, H)),
            cam_T_world,
            (W, H),
            sky_mask_top_fraction=cfg.depth.sky_mask_top_fraction,
        )
        fallback = fallback_window[-1] if fallback_window else None
        fit_params = ransac_disparity_fit(
            d_lidar,
            disp,
            n_iter=cfg.depth.ransac_n_iter,
            inlier_threshold_m=cfg.depth.ransac_inlier_threshold_m,
            min_anchors=cfg.depth.min_lidar_anchors,
            fallback=fallback,
        )
        fit_params["n_anchors"] = int(len(d_lidar))
        if fit_params["fit_status"] == depth.FitStatus.OK:
            fallback_window.append((fit_params["scale"], fit_params["shift"]))

    if fit_params["fit_status"] != depth.FitStatus.FAILED:
        metric_depth = disparity_to_depth(
            disparity,
            fit_params["scale"],
            fit_params["shift"],
            out_dtype=cfg.depth.output_dtype,
        )
        _write_depth_artifact(
            bag_id, chunk_id, cam_id, frame.camera_seq, metric_depth, fit_params
        )


def _partial_dir(bag_id: str, chunk_id: str) -> str:
    """Directory holding per-camera resume pickles for this chunk."""
    return os.path.join(
        local_path(chunk_root(bag_id, chunk_id)), ".perception_2d_partial"
    )


def _cam_partial_path(bag_id: str, chunk_id: str, cam_id: str) -> str:
    return os.path.join(_partial_dir(bag_id, chunk_id), f"{cam_id}.pkl")


def _load_cam_partial(
    bag_id: str, chunk_id: str, cam_id: str, cfg: ComponentConfig, device: str
) -> Optional[list[Masklet]]:
    """Return cached per-camera masklets if a readable partial exists.

    An empty list is a real result (an empty road): tracking failures raise
    instead of returning fewer masklets.
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
    if (
        not isinstance(cached, dict)
        or cached.get("config") != cfg.model_dump(mode="json")
        or cached.get("mask_propagation")
        != propagation_provenance(cfg.segmentation.backend, device)
    ):
        raise RuntimeError(
            f"{cam_id}: resume cache config/propagation policy differs or is unrecorded; "
            "rerun with --force to avoid mixing mask policies"
        )
    return cached["masklets"]


def _save_cam_partial(
    bag_id: str,
    chunk_id: str,
    cam_id: str,
    cam_masklets: list[Masklet],
    cfg: ComponentConfig,
    device: str,
) -> None:
    """Persist per-camera masklets so abort doesn't lose this camera's work."""
    p = _cam_partial_path(bag_id, chunk_id, cam_id)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(
            {
                "config": cfg.model_dump(mode="json"),
                "mask_propagation": propagation_provenance(
                    cfg.segmentation.backend, device
                ),
                "masklets": cam_masklets,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    os.replace(tmp, p)


def _free_cuda() -> None:
    """Return freed allocations to the CUDA driver (best-effort, no-op on CPU).

    gc.collect() first so tensors held only by unreachable cycles (e.g. a model's
    intermediate objects that reference GPU buffers) are dropped before
    empty_cache, which only frees blocks with no live references.
    """
    try:
        import gc

        import torch

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _log_vram(tag: str) -> None:
    """Log reserved/allocated VRAM (best-effort, no-op on CPU), so drift between
    cameras shows in the logs before it becomes an OOM."""
    try:
        import torch

        if torch.cuda.is_available():
            log.info(
                "VRAM after %s: %.1fGB reserved, %.1fGB allocated",
                tag,
                torch.cuda.memory_reserved() / 1e9,
                torch.cuda.memory_allocated() / 1e9,
            )
    except Exception:  # noqa: BLE001
        pass


def _inference_ctx():
    """torch.inference_mode() when torch is present, else a no-op context."""
    try:
        import torch

        return torch.inference_mode()
    except Exception:  # noqa: BLE001
        import contextlib

        return contextlib.nullcontext()


def _check_tracking_weights(cfg: ComponentConfig, detector: Detector) -> None:
    """Raise unless the tracking pass's weights are there, before the depth pass.

    The tracking pass loads them after a chunk's depth pass (~25 min on a WATO
    chunk), which is where a missing checkpoint would otherwise surface. None of
    this takes VRAM: YOLO and DINOv2 load on the CPU. GroundingDINO loads
    straight onto the GPU, so it is left to its first detect().
    """
    if not os.path.isfile(cfg.segmentation.checkpoint):
        raise RuntimeError(
            f"SAM2 checkpoint {cfg.segmentation.checkpoint!r} not found. Fetch it "
            "with scripts/fetch_models.py."
        )
    if isinstance(detector, YoloDetector):
        detector.load()
    if cfg.embeddings.every_k_frames > 0:
        check_dino_model(cfg.embeddings.model)


def _process_chunk(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
    discovery: Optional[Florence2Discovery],
    detector: Detector,
    fixed_concepts: list[Concept],
    device: str,
    *,
    force: bool = False,
) -> None:
    """Run one chunk as two VRAM-disjoint passes: depth, then tracking.

    DA-V2 (depth) and the detector + SAM2 (tracking) are never built at the same
    time, so only one heavy model set occupies the GPU at once.
    """
    selection = camera_calibration.resolve(bag_id, cfg.calibration_source)
    camera_calibration.bind_outputs(
        bag_id,
        chunk_id,
        "perception_2d",
        selection,
        force=force,
        existing_outputs=_chunk_complete(bag_id, chunk_id)
        or os.path.isdir(local_path(depth_2d_dir(bag_id, chunk_id)))
        or os.path.isdir(_partial_dir(bag_id, chunk_id)),
    )
    if force:
        # Start the chunk clean. Mask dirs are named by masklet_id, which is new
        # each run, so without this a rerun leaves the previous run's masks
        # beside the new ones (unreferenced by tracklets_2d, but there when
        # browsing masks_2d); a depth frame whose fit now fails would keep its
        # old file, breaking "missing file = no usable depth".
        for d in (
            _partial_dir(bag_id, chunk_id),
            local_path(masks_2d_dir(bag_id, chunk_id)),
            local_path(depth_2d_dir(bag_id, chunk_id)),
            mask_quality.directory(bag_id, chunk_id),
        ):
            if os.path.isdir(d):
                shutil.rmtree(d)

    frames_by_cam = FrameIndex.load(bag_id, chunk_id).by_camera()
    n_images = sum(len(v) for v in frames_by_cam.values())
    if not n_images:
        log.warning("chunk %s: empty frame index — skipping", chunk_id)
        _write_empty(cfg, bag_id, chunk_id)
        return

    # There are frames to track: confirm SAM2 + the detector import before the
    # expensive depth pass.
    if not sam2_importable():
        raise RuntimeError(
            f"chunk {chunk_id}: SAM2 (`sam2`) is not importable but "
            f"{n_images} frames need tracking. Install the `sam2` package and "
            "fetch facebook/sam2.1-hiera-large into the HF cache."
        )
    backend_ok, backend_pkg = (
        (yolo_importable, "ultralytics")
        if cfg.detection.backend == "yolo"
        else (detector_importable, "transformers")
    )
    if not backend_ok():
        raise RuntimeError(
            f"chunk {chunk_id}: the {cfg.detection.backend} detector backend "
            f"(`{backend_pkg}`) is not importable but {n_images} frames need "
            f"detection. Install {backend_pkg} and fetch the checkpoint "
            "(scripts/fetch_models.py)."
        )
    _check_tracking_weights(cfg, detector)
    check_sam2_cuda_extension(device)

    calib = camera_calibration.resolve(
        bag_id, cfg.calibration_source, frames_by_cam
    ).calibration
    calibration = {cam_id: calib.camera(cam_id) for cam_id in calib.cameras}

    partial_dir = _partial_dir(bag_id, chunk_id)

    log.info(
        "chunk %s: %d images across %d cameras; discovery=%s; depth=%s",
        chunk_id,
        n_images,
        len(frames_by_cam),
        cfg.discovery.backend,
        "on" if cfg.depth.enabled else "off",
    )

    # PASS 1 — depth. DA-V2 is the only heavy model resident; it is freed at the
    # end of the pass, before the detector / SAM2 are built.
    if cfg.depth.enabled:
        _run_depth_pass(cfg, bag_id, chunk_id, frames_by_cam, calibration)
        # Depth is done; drop the per-sweep LiDAR cache so it doesn't sit in RAM
        # through tracking.
        clear_lidar_caches(bag_id, chunk_id)

    # PASS 2 — tracking. Detector + SAM2 only; DA-V2's VRAM is freed.
    all_masklets = _run_tracking_pass(
        cfg,
        bag_id,
        chunk_id,
        frames_by_cam,
        calibration,
        discovery,
        detector,
        fixed_concepts,
        device,
    )

    # PASS 3 — appearance descriptors from the masks just written, then the
    # tables that mark the chunk complete.
    _write_masklets(cfg, bag_id, chunk_id, all_masklets, frames_by_cam, device)
    log.info("chunk %s: wrote %d masklets", chunk_id, len(all_masklets))

    # Final parquets committed — the per-camera resume cache is now redundant.
    if os.path.isdir(partial_dir):
        shutil.rmtree(partial_dir, ignore_errors=True)


def _run_depth_pass(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
    frames_by_cam: dict[str, list[CameraFrame]],
    calibration: dict[str, CameraCalibration],
) -> None:
    """PASS 1: write metric depth for every frame, then free DA-V2's VRAM.

    DA-V2 is built here (not in run()) and unloaded in the finally block so it
    never co-resides with the tracker. Frames within a camera are pushed through
    the backbone ``cfg.depth.batch_size`` at a time (they share one resolution);
    the LiDAR alignment that follows each batch stays strictly sequential, so the
    fallback-window state is identical to per-frame processing.
    """
    depth_model = DepthAnythingV2(model_size=cfg.depth.model, device=None)
    batch_size = max(1, cfg.depth.batch_size)
    try:
        for cam_id, cam_frames in frames_by_cam.items():
            calib = calibration.get(cam_id)
            if calib is None:
                continue  # no calibration → the tracking pass skips it too
            fallback_window: deque[tuple[float, float]] = deque(
                maxlen=cfg.depth.fallback_window
            )
            with tqdm(
                total=len(cam_frames),
                desc=f"{chunk_id}/{cam_id} depth",
                unit="frame",
                leave=False,
            ) as pbar:
                for start in range(0, len(cam_frames), batch_size):
                    batch_frames = cam_frames[start : start + batch_size]
                    images: list[np.ndarray] = []
                    kept_frames: list[CameraFrame] = []
                    for frame in batch_frames:
                        image = _load_image(frame.image_path)
                        if image is None:
                            continue
                        images.append(image)
                        kept_frames.append(frame)
                    if images:
                        disparities, _ = depth_model.infer_batch(
                            images, batch_size=batch_size
                        )
                        for frame, disparity in zip(kept_frames, disparities):
                            _align_and_write_depth(
                                cfg,
                                bag_id,
                                chunk_id,
                                cam_id,
                                frame,
                                disparity,
                                calib,
                                fallback_window,
                            )
                    pbar.update(len(batch_frames))
    finally:
        # Drop DA-V2 from VRAM before the tracking pass builds the detector/SAM2.
        depth_model.unload()
        del depth_model
        _free_cuda()


def _run_tracking_pass(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
    frames_by_cam: dict[str, list[CameraFrame]],
    calibration: dict[str, CameraCalibration],
    discovery: Optional[Florence2Discovery],
    detector: Detector,
    fixed_concepts: list[Concept],
    device: str,
) -> list[Masklet]:
    """PASS 2: detector + SAM2 video tracking per camera → masklets.

    Images are re-loaded here (the depth pass already discarded them) so only
    one camera's frames are resident at a time.
    """
    predictor = get_sam2_predictor(
        checkpoint=cfg.segmentation.checkpoint,
        config_file=cfg.segmentation.config,
        device=device,
        backend=cfg.segmentation.backend,
        dam4sam_update_delta=cfg.segmentation.dam4sam.update_delta,
        dam4sam_memory_min_predicted_iou=(
            cfg.segmentation.dam4sam.memory_min_predicted_iou
        ),
    )
    if predictor is None:
        # sam2 imports (checked before the depth pass) but the predictor still
        # couldn't be built — e.g. the checkpoint isn't present. Fail loud.
        raise RuntimeError(
            f"chunk {chunk_id}: SAM2 predictor could not be built. Place the "
            f"checkpoint at {cfg.segmentation.checkpoint} (scripts/fetch_models.py)."
        )

    all_masklets: list[Masklet] = []
    with _inference_ctx():
        for cam_id, cam_frames in frames_by_cam.items():
            calib = calibration.get(cam_id)
            if calib is None:
                log.warning(
                    "chunk %s: no calibration for %s — skipping", chunk_id, cam_id
                )
                continue

            cached = _load_cam_partial(bag_id, chunk_id, cam_id, cfg, device)
            if cached is not None:
                log.info(
                    "chunk %s / %s: resuming %d cached masklets — skipping reprocessing",
                    chunk_id,
                    cam_id,
                    len(cached),
                )
                all_masklets.extend(cached)
                continue

            log.info(
                "chunk %s / %s: tracking (%d frames)", chunk_id, cam_id, len(cam_frames)
            )

            valid_frames: list[CameraFrame] = []
            valid_images: list[np.ndarray] = []
            for frame in cam_frames:
                image = _load_image(frame.image_path)
                if image is None:
                    continue
                valid_frames.append(frame)
                valid_images.append(image)

            # Class vocabulary for this camera: open-vocab (Florence-2) or fixed.
            if discovery is not None:
                concepts = _discover_concepts_florence2(discovery, valid_images, cfg)
            else:
                concepts = fixed_concepts
            ego = (
                _ego_mask(
                    cfg, cam_id, valid_images[0].shape[1], valid_images[0].shape[0]
                )
                if valid_images
                else None
            )

            events: Optional[list[dict]] = [] if cfg.segmentation.mask_quality else None
            cam_masklets = track_camera(
                predictor,
                detector,
                valid_frames,
                valid_images,
                concepts,
                bag_id=bag_id,
                chunk_id=chunk_id,
                cam_id=cam_id,
                tracker_backend=cfg.segmentation.backend,
                device=device,
                redetect_every_k=cfg.detection.redetect_every_k,
                iou_match_threshold=cfg.tracking.iou_match_threshold,
                offload_video_to_cpu=cfg.tracking.offload_video_to_cpu,
                sub_clip_frames=cfg.tracking.sub_clip_frames,
                min_sub_clip_frames=cfg.tracking.min_sub_clip_frames,
                drop_lost_after_frames=cfg.tracking.drop_lost_after_frames,
                autocast_dtype=cfg.tracking.autocast_dtype,
                ego_mask=ego,
                ego_max_fraction=cfg.ego_mask.max_fraction,
                island_fraction=cfg.tracking.island_fraction,
                birth_score=cfg.tracking.birth_score,
                confirm_keyframes=cfg.tracking.confirm_keyframes,
                support_keyframes=cfg.tracking.support_keyframes,
                support_min_area=cfg.tracking.support_min_area,
                reprompt_score=cfg.tracking.reprompt_score,
                reprompt_below_iou=cfg.tracking.reprompt_below_iou,
                birth_gate=cfg.tracking.birth_gate,
                birth_untracked_fraction=cfg.tracking.birth_untracked_fraction,
                birth_max_mask_iou=cfg.tracking.birth_max_mask_iou,
                backfill_frames=cfg.tracking.backfill_frames,
                refinement=cfg.segmentation.refinement,
                diagnostics=events,
            )
            if events is not None:
                mask_quality.write(
                    bag_id,
                    chunk_id,
                    cam_id,
                    backend=cfg.segmentation.backend,
                    refinement=cfg.segmentation.refinement.model_dump(mode="json"),
                    events=events,
                )
            log.info(
                "chunk %s / %s: %d masklets from %d frames",
                chunk_id,
                cam_id,
                len(cam_masklets),
                len(valid_frames),
            )
            # Log the post-camera baseline so accumulation across cameras is
            # visible early.
            _log_vram(f"{cam_id} tracking")
            _save_cam_partial(bag_id, chunk_id, cam_id, cam_masklets, cfg, device)
            all_masklets.extend(cam_masklets)

    return all_masklets


def _masklet_record(mkl: Masklet, syn2cls: dict[str, str]) -> masklets.MaskletRecord:
    return masklets.MaskletRecord(
        masklet_id=mkl.masklet_id,
        cam_id=mkl.cam_id,
        # cls is canonicalised to the taxonomy; raw detected phrase preserved.
        cls=syn2cls.get(mkl.cls.lower(), mkl.cls),
        score=mkl.score,
        frames_present=tuple(mkl.frames_present),
        raw_phrase=mkl.cls,
        det_score=mkl.score,
        tracker_backend=mkl.tracker_backend,
    )


def _write_embeddings(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
    records: list[masklets.MaskletRecord],
    frames_by_cam: dict[str, list[CameraFrame]],
    device: str,
) -> None:
    """PASS 3: one appearance descriptor per (masklet, embedded image), from
    the masks the tracking pass wrote (models/embeddings.py)."""
    found = embeddings.embed_chunk(
        bag_id,
        chunk_id,
        records,
        frames_by_cam,
        model_name=cfg.embeddings.model,
        every_k_frames=cfg.embeddings.every_k_frames,
        device=device,
    )
    masklets.write_embeddings(bag_id, chunk_id, found)
    log.info(
        "chunk %s: %d descriptors over %d of %d masklets",
        chunk_id,
        len(found),
        len(set(found.masklet_ids.tolist())),
        len(records),
    )


def _write_masklets(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
    all_masklets: list[Masklet],
    frames_by_cam: dict[str, list[CameraFrame]],
    device: str,
) -> None:
    """The chunk's descriptors, then its tables: the tables mark the chunk
    complete, so a run that stops in between is redone, not left without
    descriptors."""
    syn2cls = cfg.synonym_to_class_map()
    records = [_masklet_record(m, syn2cls) for m in all_masklets]
    if cfg.embeddings.every_k_frames > 0:
        _write_embeddings(cfg, bag_id, chunk_id, records, frames_by_cam, device)
    masklets.write(bag_id, chunk_id, records)


def embed(
    cfg: ComponentConfig,
    *,
    bag_id: str,
    chunk_id: Optional[str] = None,
    force: bool = False,
) -> None:
    """Write the descriptors of chunks already tracked, from their masks on
    disk: a chunk tracked before masklet_embeddings.npz existed, or with
    another descriptor. Masks, tables and the chunk's manifest are untouched;
    the file names the model and pooling that made it.
    """
    if cfg.embeddings.every_k_frames <= 0:
        raise ValueError("embeddings.every_k_frames is 0: embeddings are off")
    chunks = load_chunks(bag_id)
    if chunk_id:
        chunks = [c for c in chunks if c["chunk_id"] == chunk_id]
    if not chunks:
        raise FileNotFoundError(f"no chunk {chunk_id or ''} for bag {bag_id!r}")
    check_dino_model(cfg.embeddings.model)
    device = _default_device()
    for chunk in chunks:
        cid = chunk["chunk_id"]
        if not _chunk_complete(bag_id, cid):
            raise FileNotFoundError(
                f"chunk {cid}: no masklets to embed (run perception_2d first)"
            )
        if masklets.embeddings_exist(bag_id, cid) and not force:
            log.info("chunk %s already embedded — skipping (use --force)", cid)
            continue
        frames_by_cam = FrameIndex.load(bag_id, cid).by_camera()
        _write_embeddings(
            cfg, bag_id, cid, masklets.read(bag_id, cid), frames_by_cam, device
        )


def _write_empty(cfg: ComponentConfig, bag_id: str, chunk_id: str) -> None:
    if cfg.embeddings.every_k_frames > 0:
        _write_embeddings(cfg, bag_id, chunk_id, [], {}, "cpu")
    masklets.write(bag_id, chunk_id, [])


def build_detector(cfg: ComponentConfig, fixed_concepts: list[Concept]):
    """The configured per-keyframe box source (``detection.backend``).

    Both backends expose ``detect(image_rgb, concepts) -> list[Detection]``. The
    detector is built here but loads its weights lazily on the first detect() —
    in the tracking pass, after the depth pass — so it stays out of VRAM during
    depth. Configuration mismatches are rejected here, before any chunk runs.
    """
    det = cfg.detection
    if det.backend == "grounding_dino":
        return GroundingDinoDetector(
            model_id=det.model,
            box_threshold=det.box_threshold,
            text_threshold=det.text_threshold,
            nms_iou=det.nms_iou,
            device=None,
        )
    if cfg.discovery.backend != "fixed":
        raise ValueError(
            "detection.backend: yolo has a fixed (COCO) vocabulary and can't be "
            f"prompted by discovery.backend: {cfg.discovery.backend}. Use "
            "discovery.backend: fixed, or detection.backend: grounding_dino for "
            "open-vocabulary detection."
        )
    misnamed, unused, uncovered = check_class_map(det.yolo.class_map, fixed_concepts)
    taxonomy = sorted({c for _, c in fixed_concepts})
    if misnamed:
        raise ValueError(
            f"detection.yolo.class_map maps to {sorted(misnamed)}, which nearly "
            f"match taxonomy classes {taxonomy} but aren't one; those detections "
            f"would be silently dropped. Fix the spelling, or check that "
            f"{cfg.prompts_path} (synonym → canonical class) is mounted."
        )
    if unused:
        log.info(
            "detection.yolo.class_map targets %s are not in this taxonomy; "
            "those detections are dropped",
            sorted(unused),
        )
    if uncovered:
        log.warning(
            "detection.backend yolo: no class_map entry produces taxonomy "
            "classes %s — they will never be detected",
            sorted(uncovered),
        )
    return YoloDetector(
        det.yolo.checkpoint,
        class_map=det.yolo.class_map,
        conf=det.yolo.conf,
        iou=det.yolo.iou,
        imgsz=det.yolo.imgsz,
        device=None,
    )


def _write_chunk_manifest(
    bag_id: str,
    chunk_id: str,
    config_path: Optional[str],
    cfg: ComponentConfig,
    device: str,
) -> None:
    """Record the models (``model_revisions()``), config and inputs behind this
    chunk's 2D labels. Best-effort: never fails a chunk that succeeded."""
    try:
        common_manifest.write(
            component="perception_2d",
            bag_id=bag_id,
            chunk_id=chunk_id,
            inputs={
                "frame_index": frame_index_path(bag_id, chunk_id),
                "poses": poses_path(bag_id, chunk_id),
            },
            outputs={
                "detections_2d": detections_2d_path(bag_id, chunk_id),
                "tracklets_2d": tracklets_2d_path(bag_id, chunk_id),
                "masks_2d": masks_2d_dir(bag_id, chunk_id),
                **(
                    {"masklet_embeddings": masklet_embeddings_path(bag_id, chunk_id)}
                    if cfg.embeddings.every_k_frames > 0
                    else {}
                ),
                "depth_2d": depth_2d_dir(bag_id, chunk_id),
                "depth_stats": depth_stats_path(bag_id, chunk_id),
                **(
                    {"mask_quality": mask_quality.directory_uri(bag_id, chunk_id)}
                    if cfg.segmentation.mask_quality
                    else {}
                ),
            },
            config_path=config_path,
            models=model_revisions(),
            extra={
                "effective_config": cfg.model_dump(mode="json"),
                "mask_propagation": propagation_provenance(
                    cfg.segmentation.backend, device
                ),
            },
            filename=common_manifest.component_manifest_name("perception_2d"),
        )
    except Exception:  # noqa: BLE001
        log.warning("failed to write manifest for chunk %s", chunk_id, exc_info=True)


def run(
    cfg: ComponentConfig,
    *,
    bag_id: str,
    chunk_id: Optional[str] = None,
    force: bool = False,
    config_path: Optional[str] = None,
) -> None:
    """Process one bag (or one chunk) end-to-end.

    Skips chunks whose output parquets already exist unless force=True.  Model
    loading is fail-loud: a chunk that needs a model the environment can't
    provide raises rather than emitting empty placeholder output.

    Args:
        config_path: the config behind ``cfg``, hashed into each manifest.
    """
    if cfg.discovery.backend == "fixed" and not os.path.exists(cfg.prompts_path):
        log.info(
            "prompts.yaml not found at %s — fixed concepts come from "
            "discovery.fixed_classes (or the hardcoded fallback).",
            cfg.prompts_path,
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
        bag_id,
        len(chunks),
        force,
    )
    log.info("=" * 70)

    selection = camera_calibration.resolve(bag_id, cfg.calibration_source)
    for chunk in chunks:
        cid = chunk["chunk_id"]
        camera_calibration.bind_outputs(
            bag_id,
            cid,
            "perception_2d",
            selection,
            force=force,
            existing_outputs=_chunk_complete(bag_id, cid)
            or os.path.isdir(local_path(depth_2d_dir(bag_id, cid)))
            or os.path.isdir(_partial_dir(bag_id, cid)),
        )
    device = _default_device()

    # Florence-2 and the detector load their weights lazily (in the tracking
    # pass); DA-V2 and SAM2 are built inside each chunk's passes.
    discovery: Optional[Florence2Discovery] = None
    fixed_concepts: list[Concept] = []
    if cfg.discovery.backend == "florence2":
        discovery = Florence2Discovery(model_id=cfg.discovery.model, device=None)
        log.info("perception_2d: open-vocabulary discovery (Florence-2) → detector")
    else:
        fixed_concepts = cfg.concept_prompts()
        log.info(
            "perception_2d: fixed-vocabulary detection (%d concepts)",
            len(fixed_concepts),
        )

    detector = build_detector(cfg, fixed_concepts)

    n_ok = n_skip = 0
    try:
        for chunk in chunks:
            cid = chunk["chunk_id"]
            if not force and _chunk_complete(bag_id, cid):
                log.info("chunk %s already complete — skipping (use force=True)", cid)
                n_skip += 1
                continue
            # Fail loud: a chunk failure (e.g. a missing model) propagates out of
            # run() rather than being swallowed into a log line.
            _process_chunk(
                cfg,
                bag_id,
                cid,
                discovery,
                detector,
                fixed_concepts,
                device,
                force=force,
            )
            _write_chunk_manifest(bag_id, cid, config_path, cfg, device)
            n_ok += 1
    finally:
        # Don't leave SAM2 parked in VRAM after the bag finishes.
        release_sam2_predictor()

    log.info(
        "perception_2d complete for bag %s: %d processed, %d skipped",
        bag_id,
        n_ok,
        n_skip,
    )
