"""Semantic lifting pipeline orchestrator.

Per-chunk steps:
  For each LiDAR sweep:
    1. Temporal match: find nearest camera frame per camera within max_offset_s
    2. Load all LiDAR points for the sweep
    3. For each matched camera:
       a. Compute time-corrected cam_T_lidar (accounts for ego motion)
       b. Project LiDAR points to image
       c. Load depth_2d artifact; skip camera if fit_status == 2
       d. Run UniLiPs Eq.1 visibility test
       e. Load masks from tracklets_2d artifact
       f. Assign masks to visible points (innermost-mask heuristic)
       g. Cast votes with score = sam3_score * depth_confidence (1.0 fallback)
    4. Accumulate cross-camera votes → final per-point labels
    5. Write lifted_labels/<sweep_id>.npz
  Write lifted_stats.parquet
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from wato_semantic_lifting.config import ComponentConfig
from wato_semantic_lifting.io import (
    CalibrationInfo,
    load_all_lidar_points,
    load_calibration,
    load_chunks,
    load_depth_map,
    load_frame_refs,
    load_masks_for_frame,
    load_sweeps,
    write_lifted_labels,
    write_lifted_stats,
)
from wato_semantic_lifting.projection import assign_masks_to_points, project_and_clip
from wato_semantic_lifting.stats import build_stats_row
from wato_semantic_lifting.temporal_match import CameraFrameRef, compute_cam_T_lidar, match_sweep_to_frames
from wato_semantic_lifting.visibility import visibility_test
from wato_semantic_lifting.voting import LabeledPoint, PointVote, accumulate_votes

log = logging.getLogger(__name__)


def _process_chunk(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
    calibration: dict[str, CalibrationInfo],
) -> None:
    """Run semantic lifting for all sweeps in one chunk."""
    sweeps = load_sweeps(bag_id, chunk_id)
    if not sweeps:
        log.warning("semantic_lifting: chunk %s has no valid sweeps", chunk_id)
        write_lifted_stats(bag_id, chunk_id, [])
        return

    frame_refs = load_frame_refs(bag_id, chunk_id)
    stats_rows = []

    for sweep in sweeps:
        log.debug("semantic_lifting: sweep %d", sweep.sweep_id)

        matched_cams = match_sweep_to_frames(
            sweep.timestamp_ns,
            frame_refs,
            max_offset_s=cfg.temporal.max_offset_s,
        )

        pts_world = load_all_lidar_points(sweep)
        if pts_world is None or pts_world.shape[0] == 0:
            stats_rows.append(
                build_stats_row(
                    bag_id, chunk_id, sweep.sweep_id,
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

            depth_info = load_depth_map(bag_id, chunk_id, cam_id, frame_ref.camera_seq)
            if depth_info is None:
                continue
            if cfg.depth.skip_on_fit_failure and depth_info["fit_status"] == 2:
                continue

            cam_T_lidar = compute_cam_T_lidar(
                world_T_ego_sweep=_approx_ego_at_sweep(sweep, frame_refs),
                world_T_ego_frame=frame_ref.world_T_ego,
                ego_T_lidar=calib.ego_T_lidar,
                ego_T_cam=calib.ego_T_cam,
            )

            H, W = depth_info["depth_m"].shape
            uv, depth_cam, valid = project_and_clip(
                pts_world, calib.K, cam_T_lidar, (W, H)
            )

            if not valid.any():
                continue

            uv_valid = uv[valid]
            d_cam_valid = depth_cam[valid]
            valid_indices = np.where(valid)[0]

            vis = visibility_test(
                d_cam_valid,
                depth_info["depth_m"],
                uv_valid,
                tolerance_m=cfg.visibility.tolerance_m,
                neighborhood=cfg.visibility.neighborhood,
            )

            n_failed_vis += int((~vis).sum())

            uv_vis = uv_valid[vis]
            d_cam_vis = d_cam_valid[vis]
            vis_indices = valid_indices[vis]

            if uv_vis.shape[0] == 0:
                continue

            masks, instance_ids, classes = load_masks_for_frame(
                bag_id, chunk_id, cam_id, frame_ref.camera_seq
            )
            if not masks:
                continue

            assignment = assign_masks_to_points(uv_vis, masks)

            for local_i, pt_idx in enumerate(vis_indices):
                mask_idx = int(assignment[local_i])
                if mask_idx < 0:
                    continue
                n_in_any_mask += 1
                score = float(d_cam_vis[local_i]) if d_cam_vis[local_i] > 0 else 1.0
                score = 1.0 / (1.0 + score / 50.0)  # crude depth confidence proxy
                all_votes.append(
                    PointVote(
                        point_idx=int(pt_idx),
                        instance_id=instance_ids[mask_idx],
                        cls=classes[mask_idx],
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
                bag_id, chunk_id, sweep.sweep_id,
                n_points_total=n_total,
                labeled_points=labeled,
                n_points_in_any_mask=n_in_any_mask,
                n_points_failed_visibility=n_failed_vis,
                n_cameras_used=n_cameras_used,
            )
        )

    write_lifted_stats(bag_id, chunk_id, stats_rows)
    log.info(
        "semantic_lifting: chunk %s done — %d sweeps, %d total stats rows",
        chunk_id, len(sweeps), len(stats_rows),
    )


def _approx_ego_at_sweep(sweep, frame_refs: list[CameraFrameRef]) -> np.ndarray:
    """Use the ego pose from the nearest frame ref as sweep-time ego pose.

    Falls back to identity when no frame ref is available.
    """
    if not frame_refs:
        return np.eye(4, dtype=np.float64)
    best = min(frame_refs, key=lambda f: abs(f.timestamp_ns - sweep.timestamp_ns))
    return best.world_T_ego


def _write_sweep_labels(
    bag_id: str,
    chunk_id: str,
    sweep_id: int,
    labeled: list[LabeledPoint],
) -> None:
    if not labeled:
        write_lifted_labels(
            bag_id, chunk_id, sweep_id,
            np.array([], dtype=np.int32),
            [], [],
            np.array([], dtype=np.float32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        )
        return

    write_lifted_labels(
        bag_id, chunk_id, sweep_id,
        point_indices=np.array([p.point_idx for p in labeled], dtype=np.int32),
        instance_ids=[p.instance_id for p in labeled],
        classes=[p.cls for p in labeled],
        confidences=np.array([p.confidence for p in labeled], dtype=np.float32),
        n_supporting=np.array([p.n_supporting_cameras for p in labeled], dtype=np.int32),
        n_disagreeing=np.array([p.n_disagreeing_cameras for p in labeled], dtype=np.int32),
    )


def run(
    cfg: ComponentConfig,
    *,
    bag_id: str,
    chunk_id: Optional[str] = None,
    force: bool = False,
) -> None:
    """Process one bag (or one chunk) end-to-end."""
    from wato_common.artifact_store import lifted_stats_path, local_path
    import os

    chunks = load_chunks(bag_id)
    if not chunks:
        raise FileNotFoundError(
            f"No chunks found for bag {bag_id!r} — run ingest first."
        )

    if chunk_id:
        chunks = [c for c in chunks if c["chunk_id"] == chunk_id]
        if not chunks:
            raise ValueError(f"chunk_id {chunk_id!r} not found for bag {bag_id!r}")

    calibration = load_calibration(bag_id)

    n_ok = n_skip = 0
    for chunk in chunks:
        cid = chunk["chunk_id"]
        stats_path = local_path(lifted_stats_path(bag_id, cid))
        if not force and os.path.exists(stats_path):
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
        bag_id, n_ok, n_skip,
    )
