"""perception_2d pipeline orchestrator.

Each chunk is processed in two passes so the two heavy GPU models are never
co-resident — critical on small-VRAM GPUs, where DA-V2-Large + SAM 3.1 + DINOv2
do not all fit (e.g. ~6 GB):

  PASS 1 — depth (only DA-V2 in VRAM), per camera/frame:
    1. DepthAnythingV2.infer() → relative_depth
    2. Load static LiDAR points for this sweep
    3. depth_align.build_anchor_pairs() → d_lidar, d_da
    4. depth_align.ransac_affine_fit() → affine params
    5. depth_align.apply_affine() → metric_depth
    6. Write depth_2d/<cam>/<frame>.npz (skipped on total fit failure)
    Then DA-V2 is unloaded and its VRAM freed before tracking starts.

  PASS 2 — tracking (only SAM 3.1 + DINOv2 in VRAM), per camera stream:
    Concept source (selected by discovery.backend):
      fixed     → concepts come straight from config (closed-set, e.g. COCO).
      florence2 → Florence-2 discovers open-vocab phrases across the camera
                  stream; they're pooled + canonicalised into the concept set.
    SAM 3.1's multiplex concept-video predictor detects + segments + tracks
    every instance of each concept across the stream → Masklets with persistent
    IDs, with DINOv2 appearance embeddings extracted every k frames. Then write
    detections_2d.parquet + tracklets_2d.parquet.

The split costs a second image decode (each pass loads the frames once) in
exchange for halving peak VRAM. SAM 3.1 is the only tracker; model loading is
fail-loud — a missing predictor / depth model / Florence-2 raises rather than
emitting degraded output, and the SAM 3.1 import is checked before the depth
pass so a missing install fails fast instead of after the whole depth pass.

Cross-camera identity merging is intentionally NOT done here.  perception_2d's
deliverable is per-camera masklets + masks + metric depth + DINOv2 appearance
embeddings.  Resolving the same physical object across cameras belongs in the
downstream `tracking` component (3D/4D, gated by the DINOv2 features persisted
here).  `global_object_id` is left null for `tracking` to populate.
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
from wato_perception_2d.fusion.depth_align import (
    apply_affine,
    build_anchor_pairs,
    ransac_affine_fit,
)
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
from wato_perception_2d.models._sam3_runtime import (
    get_sam3_predictor,
    release_sam3_predictor,
    sam3_importable,
)
from wato_perception_2d.models.depth import DepthAnythingV2
from wato_perception_2d.models.discovery import Florence2Discovery
from wato_perception_2d.models.sam3_concept_tracker import track_camera_concepts

log = logging.getLogger(__name__)

# A (text_prompt, canonical_class) concept fed to the SAM 3.1 concept tracker.
Concept = tuple[str, str]


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
    fit_params: dict = {
        "a": 1.0,
        "b": 0.0,
        "n_inliers": 0,
        "rmse_inliers_m": 0.0,
        "fit_status": 2,
    }

    rel_depth, _ = depth_model.infer(image)
    static_pts = load_static_lidar_points(bag_id, chunk_id, frame.sweep_id)

    if static_pts is not None and frame.valid_pose and frame.world_T_ego_flat:
        world_T_ego = unflatten_se3(frame.world_T_ego_flat)
        cam_T_world = invert_se3(calib.ego_T_cam) @ invert_se3(world_T_ego)
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
            min_anchors=cfg.depth.min_lidar_anchors,
            fallback=fallback,
        )
        fit_params["n_anchors"] = int(len(d_lidar))
        if fit_params["fit_status"] == 0:
            fallback_window.append((fit_params["a"], fit_params["b"]))

    if fit_params["fit_status"] != 2:
        metric_depth = apply_affine(
            rel_depth,
            fit_params["a"],
            fit_params["b"],
            out_dtype=cfg.depth.output_dtype,
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


def _free_cuda() -> None:
    """Return freed allocations to the CUDA driver (best-effort, no-op on CPU)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _process_chunk(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
    discovery: Optional[Florence2Discovery],
    fixed_concepts: list[Concept],
    device: str,
    *,
    force: bool = False,
) -> None:
    """Run one chunk as two VRAM-disjoint passes: depth, then tracking.

    DA-V2 (depth) and SAM 3.1 (tracking) are never built at the same time, so
    only one heavy model occupies the GPU at once.
    """
    frames = load_frame_index(bag_id, chunk_id)
    if not frames:
        log.warning("chunk %s: empty frame index — skipping", chunk_id)
        _write_empty(bag_id, chunk_id)
        return

    # Fail FAST: there are frames to track, so SAM 3.1 is required. Confirm it
    # can import before the (expensive) depth pass, rather than discovering a
    # missing install only after depth has already run.
    if not sam3_importable():
        raise RuntimeError(
            f"chunk {chunk_id}: SAM 3.1 (`sam3`) is not importable but "
            f"{len(frames)} frames need tracking. Install the `sam3` package "
            "(+ its deps, e.g. pycocotools) and fetch facebook/sam3.1."
        )

    calibration = load_calibration(bag_id)
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
        "chunk %s: %d frames across %d cameras; discovery=%s; depth=%s",
        chunk_id,
        len(frames),
        len(frames_by_cam),
        cfg.discovery.backend,
        "on" if cfg.depth.enabled else "off",
    )

    # PASS 1 — depth. DA-V2 is the only heavy model resident; it is freed at the
    # end of the pass, before SAM 3.1 is built.
    if cfg.depth.enabled:
        _run_depth_pass(cfg, bag_id, chunk_id, frames_by_cam, calibration)
        # Depth is done; drop the per-sweep LiDAR cache so it doesn't sit in RAM
        # through tracking.
        clear_lidar_caches(bag_id, chunk_id)

    # PASS 2 — tracking. SAM 3.1 (+ DINOv2) only; DA-V2's VRAM is already freed.
    all_masklets = _run_tracking_pass(
        cfg,
        bag_id,
        chunk_id,
        frames_by_cam,
        calibration,
        masks_base,
        discovery,
        fixed_concepts,
        device,
    )

    # NOTE: cross-camera identity merging is deliberately deferred to the
    # downstream `tracking` component (see module docstring). global_object_id
    # is left null here.
    _write_masklets(bag_id, chunk_id, all_masklets, cfg.synonym_to_class_map())
    log.info("chunk %s: wrote %d masklets", chunk_id, len(all_masklets))

    # Final parquets committed — the per-camera resume cache is now redundant.
    if os.path.isdir(partial_dir):
        shutil.rmtree(partial_dir, ignore_errors=True)


def _run_depth_pass(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
    frames_by_cam: dict[str, list[CameraFrameInfo]],
    calibration: dict[str, CalibrationInfo],
) -> None:
    """PASS 1: write metric depth for every frame, then free DA-V2's VRAM.

    DA-V2 is built here (not in run()) and unloaded in the finally block so it
    never co-resides with SAM 3.1.
    """
    depth_model = DepthAnythingV2(model_size=cfg.depth.model, device=None)
    try:
        for cam_id, cam_frames in frames_by_cam.items():
            calib = calibration.get(cam_id)
            if calib is None:
                continue  # no calibration → the tracking pass skips it too
            fallback_window: deque[tuple[float, float]] = deque(
                maxlen=cfg.depth.fallback_window
            )
            for frame in tqdm(
                cam_frames,
                desc=f"{chunk_id}/{cam_id} depth",
                unit="frame",
                leave=False,
            ):
                image = _load_image(frame.image_path)
                if image is None:
                    continue
                _run_depth_branch(
                    cfg,
                    bag_id,
                    chunk_id,
                    cam_id,
                    frame,
                    image,
                    calib,
                    depth_model,
                    fallback_window,
                )
    finally:
        # Drop DA-V2 from VRAM before the tracking pass builds SAM 3.1.
        depth_model.unload()
        del depth_model
        _free_cuda()


def _run_tracking_pass(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
    frames_by_cam: dict[str, list[CameraFrameInfo]],
    calibration: dict[str, CalibrationInfo],
    masks_base: str,
    discovery: Optional[Florence2Discovery],
    fixed_concepts: list[Concept],
    device: str,
) -> list[Masklet]:
    """PASS 2: SAM 3.1 concept-video tracking per camera → masklets.

    Images are re-loaded here (the depth pass already discarded them) so only
    one camera's frames are resident at a time.
    """
    predictor = get_sam3_predictor(
        version=cfg.segmentation.version, use_fa3=cfg.segmentation.use_fa3
    )
    if predictor is None:
        # sam3 imports (checked before the depth pass) but the predictor still
        # couldn't be built — e.g. the checkpoint isn't fetched. Fail loud.
        raise RuntimeError(
            f"chunk {chunk_id}: SAM 3.1 predictor could not be built. Fetch "
            "facebook/sam3.1 into the HF cache (scripts/fetch_models.py)."
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
                chunk_id,
                cam_id,
                len(cached),
            )
            all_masklets.extend(cached)
            continue

        log.info(
            "chunk %s / %s: tracking (%d frames)", chunk_id, cam_id, len(cam_frames)
        )

        valid_frames: list[CameraFrameInfo] = []
        valid_images: list[np.ndarray] = []
        for frame in cam_frames:
            image = _load_image(frame.image_path)
            if image is None:
                continue
            valid_frames.append(frame)
            valid_images.append(image)

        # Concept vocabulary for this camera: open-vocab (Florence-2) or fixed.
        if discovery is not None:
            concepts = _discover_concepts_florence2(discovery, valid_images, cfg)
        else:
            concepts = fixed_concepts

        cam_masklets = track_camera_concepts(
            predictor,
            valid_frames,
            valid_images,
            concepts,
            bag_id=bag_id,
            chunk_id=chunk_id,
            cam_id=cam_id,
            masks_2d_base_dir=masks_base,
            dino_model=cfg.embeddings.model,
            dino_every_k=cfg.embeddings.every_k_frames,
            device=device,
            output_prob_thresh=cfg.segmentation.output_prob_thresh,
        )
        log.info(
            "chunk %s / %s: %d masklets from %d frames",
            chunk_id,
            cam_id,
            len(cam_masklets),
            len(valid_frames),
        )
        _save_cam_partial(bag_id, chunk_id, cam_id, cam_masklets)
        all_masklets.extend(cam_masklets)

    return all_masklets


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

    Skips chunks whose output parquets already exist unless force=True.  Model
    loading is fail-loud: a chunk that needs a model the environment can't
    provide raises rather than emitting empty placeholder output.
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

    device = _default_device()

    # Concept source. The fixed list is computed once; the Florence-2 model (if
    # used) is built lazily on first propose(). The heavy GPU models (DA-V2 and
    # SAM 3.1) are built *inside* each chunk's passes, not here, so the two never
    # co-reside in VRAM.
    discovery: Optional[Florence2Discovery] = None
    fixed_concepts: list[Concept] = []
    if cfg.discovery.backend == "florence2":
        discovery = Florence2Discovery(model_id=cfg.discovery.model, device=None)
        log.info("perception_2d: open-vocabulary discovery (Florence-2)")
    else:
        fixed_concepts = cfg.concept_prompts()
        log.info(
            "perception_2d: fixed-vocabulary discovery (%d concepts)",
            len(fixed_concepts),
        )

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
                cfg, bag_id, cid, discovery, fixed_concepts, device, force=force
            )
            n_ok += 1
    finally:
        # Don't leave SAM 3.1 parked in VRAM after the bag finishes.
        release_sam3_predictor()

    log.info(
        "perception_2d complete for bag %s: %d processed, %d skipped",
        bag_id,
        n_ok,
        n_skip,
    )
