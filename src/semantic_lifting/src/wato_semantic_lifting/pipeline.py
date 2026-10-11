"""Semantic lifting pipeline orchestrator.

Per-chunk steps:
  For each LiDAR sweep:
    1. Temporal match: find nearest camera frame per camera within max_offset_s
    2. Load all LiDAR points for the sweep
    3. For each matched camera:
       a. Compute cam_T_world at the matched frame time (points are already
          world-registered upstream, so no sweep-pose / ego-motion term)
       b. Project world-frame points to image
       c. Load depth_2d artifact; skip the camera when the frame has none
          (perception_2d writes none when the depth fit failed)
       d. Run UniLiPs Eq.1 visibility test
       e. Look up the masklets present in that camera frame
       f. Assign masks to visible points (innermost-mask heuristic)
       g. Cast votes with score = det_score * depth_confidence (1.0 fallback)
    4. Accumulate cross-camera votes → final per-point labels
    5. Write lifted_labels/<sweep_id:06d>.npz
  Write lifted_stats.parquet (wato_common.artifacts.lifted_labels)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from wato_common.artifacts import depth, lifted_labels
from wato_common.artifacts.calibration import Calibration, CameraCalibration
from wato_common.artifacts.lidar_sweeps import ProcessedSweeps
from wato_common.artifacts.masklets import MaskletIndex
from wato_semantic_lifting.config import ComponentConfig
from wato_semantic_lifting.io import (
    load_chunks,
    load_frame_refs,
)
from wato_semantic_lifting.projection import assign_masks_to_points, project_and_clip
from wato_semantic_lifting.stats import build_stats_row
from wato_semantic_lifting.temporal_match import (
    compute_cam_T_world,
    match_sweep_to_frames,
)
from wato_semantic_lifting.visibility import visibility_test
from wato_semantic_lifting.voting import LabeledPoint, PointVote, accumulate_votes

log = logging.getLogger(__name__)


def _process_chunk(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
    calibration: dict[str, CameraCalibration],
) -> None:
    """Run semantic lifting for all sweeps in one chunk."""
    sweeps = ProcessedSweeps.load(bag_id, chunk_id)
    if not sweeps.rows:
        log.warning("semantic_lifting: chunk %s has no valid sweeps", chunk_id)
        lifted_labels.write_stats(bag_id, chunk_id, [])
        return

    frame_refs = load_frame_refs(bag_id, chunk_id)
    masklet_index = MaskletIndex.load(bag_id, chunk_id)
    stats_rows = []

    for sweep in sweeps.rows:
        log.debug("semantic_lifting: sweep %d", sweep.sweep_id)

        matched_cams = match_sweep_to_frames(
            sweep.reference_timestamp_ns,
            frame_refs,
            max_offset_s=cfg.temporal.max_offset_s,
        )

        try:
            pts_world = sweeps.world(sweep.sweep_id).xyz
        except FileNotFoundError:
            log.warning("semantic_lifting: sweep %d has no world NPZ", sweep.sweep_id)
            pts_world = None
        if pts_world is None or pts_world.shape[0] == 0:
            stats_rows.append(
                build_stats_row(
                    bag_id,
                    chunk_id,
                    sweep.sweep_id,
                    n_points_total=0,
                    labeled_points=[],
                    n_points_in_any_mask=0,
                    n_points_failed_visibility=0,
                    n_cameras_used=0,
                )
            )
            continue

        n_total = pts_world.shape[0]
        all_votes: list[PointVote] = []
        n_in_any_mask = 0
        n_failed_vis = 0
        n_cameras_used = 0

        for cam_id, frame_ref in matched_cams.items():
            calib = calibration.get(cam_id)
            if calib is None:
                continue

            depth_frame = depth.read_frame(
                bag_id, chunk_id, cam_id, frame_ref.camera_seq
            )
            if depth_frame is None:
                continue

            cam_T_world = compute_cam_T_world(
                world_T_ego_frame=frame_ref.world_T_ego,
                ego_T_cam=calib.ego_T_cam,
            )

            H, W = depth_frame.depth_m.shape
            uv, depth_cam, valid = project_and_clip(
                pts_world, calib.K, cam_T_world, (W, H)
            )

            if not valid.any():
                continue

            uv_valid = uv[valid]
            d_cam_valid = depth_cam[valid]
            valid_indices = np.where(valid)[0]

            vis = visibility_test(
                d_cam_valid,
                depth_frame.depth_m,
                uv_valid,
                tolerance_m=cfg.visibility.tolerance_m,
                neighborhood=cfg.visibility.neighborhood,
            )

            n_failed_vis += int((~vis).sum())

            uv_vis = uv_valid[vis]
            vis_indices = valid_indices[vis]

            if uv_vis.shape[0] == 0:
                continue

            frame_masklets = masklet_index.at(cam_id, frame_ref.camera_seq)
            if not frame_masklets:
                continue
            masks = [
                masklet_index.mask(m, frame_ref.camera_seq) for m in frame_masklets
            ]

            assignment = assign_masks_to_points(uv_vis, masks)

            for local_i, pt_idx in enumerate(vis_indices):
                mask_idx = int(assignment[local_i])
                if mask_idx < 0:
                    continue
                n_in_any_mask += 1
                masklet = frame_masklets[mask_idx]
                # Vote weight is the detector confidence of the mask that claimed
                # this point (design doc's ``det_score``). The depth artifact has
                # no per-pixel confidence to fold in, so det_score is the signal;
                # a small floor keeps a zero-scored mask able to break ties.
                score = max(masklet.det_score, 1e-3)
                all_votes.append(
                    PointVote(
                        point_idx=int(pt_idx),
                        instance_id=masklet.instance_id,
                        cls=masklet.cls,
                        score=score,
                        cam_id=cam_id,
                        mask_area=int(masks[mask_idx].sum()),
                    )
                )

            n_cameras_used += 1

        labeled = accumulate_votes(all_votes, n_total)

        _write_sweep_labels(bag_id, chunk_id, sweep.sweep_id, labeled)
        stats_rows.append(
            build_stats_row(
                bag_id,
                chunk_id,
                sweep.sweep_id,
                n_points_total=n_total,
                labeled_points=labeled,
                n_points_in_any_mask=n_in_any_mask,
                n_points_failed_visibility=n_failed_vis,
                n_cameras_used=n_cameras_used,
            )
        )

    lifted_labels.write_stats(bag_id, chunk_id, stats_rows)
    log.info(
        "semantic_lifting: chunk %s done — %d sweeps, %d total stats rows",
        chunk_id,
        len(sweeps.rows),
        len(stats_rows),
    )


def _write_sweep_labels(
    bag_id: str,
    chunk_id: str,
    sweep_id: int,
    labeled: list[LabeledPoint],
) -> None:
    if not labeled:
        lifted_labels.write_sweep(
            bag_id, chunk_id, sweep_id, lifted_labels.SweepLabels.empty()
        )
        return
    lifted_labels.write_sweep(
        bag_id,
        chunk_id,
        sweep_id,
        lifted_labels.SweepLabels(
            point_indices=np.array([p.point_idx for p in labeled], dtype=np.int32),
            instance_ids=np.array([p.instance_id for p in labeled], dtype=np.str_),
            classes=np.array([p.cls for p in labeled], dtype=np.str_),
            confidences=np.array([p.confidence for p in labeled], dtype=np.float32),
            n_supporting_cameras=np.array(
                [p.n_supporting_cameras for p in labeled], dtype=np.int32
            ),
            n_disagreeing_cameras=np.array(
                [p.n_disagreeing_cameras for p in labeled], dtype=np.int32
            ),
        ),
    )


def run(
    cfg: ComponentConfig,
    *,
    bag_id: str,
    chunk_id: Optional[str] = None,
    force: bool = False,
) -> None:
    """Process one bag (or one chunk) end-to-end."""
    chunks = load_chunks(bag_id)
    if not chunks:
        raise FileNotFoundError(
            f"No chunks found for bag {bag_id!r} — run ingest first."
        )

    if chunk_id:
        chunks = [c for c in chunks if c["chunk_id"] == chunk_id]
        if not chunks:
            raise ValueError(f"chunk_id {chunk_id!r} not found for bag {bag_id!r}")

    calib = Calibration.load(bag_id)
    calibration = {cam_id: calib.camera(cam_id) for cam_id in calib.cameras}

    n_ok = n_skip = 0
    for chunk in chunks:
        cid = chunk["chunk_id"]
        if not force and lifted_labels.exists(bag_id, cid):
            log.info("chunk %s already complete — skipping", cid)
            n_skip += 1
            continue
        try:
            _process_chunk(cfg, bag_id, cid, calibration)
            n_ok += 1
        except Exception:  # noqa: BLE001
            log.exception("semantic_lifting: chunk %s failed", cid)

    log.info(
        "semantic_lifting complete for bag %s: %d processed, %d skipped",
        bag_id,
        n_ok,
        n_skip,
    )
