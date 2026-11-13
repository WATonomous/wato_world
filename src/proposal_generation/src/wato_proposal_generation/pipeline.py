"""proposal_generation pipeline orchestrator.

Per chunk, frame by frame (lidar_preprocessing's frame_sync frames, regrouped
with ProcessedSweeps.frames on the configured canonical LiDAR):
  1. Load all non-ground frame points with their provisional semantic/motion
     labels and independent lifted candidates. Refine each candidate before
     resolving 3D ownership; scan_clusters
     builds the multiscale DBSCAN bank, largest_group retains legacy cleanup.
  2. lifted: coverage suppression, proposed exclusive 3D support, and box gates
     against accepted owners; commit ownership only after acceptance (lifted.py).
  3. lidar_mos: one box per gated Step F cluster, classed only by camera seed
     points retained by the surviving 3D owners (lidar_mos.py).
  4. fusion_detector (when enabled): one row per kept detection_3d box, with
     the points inside it (detector.py).
  5. fused: one box per object, merging a lidar_mos and a lifted box that
     share points, then the detector box that shares the object's points
     (fusion.py).
Then every row goes to proposals.parquet (wato_common.artifacts.proposals).

A missing or malformed upstream artifact stops the run rather than leaving a
partial proposals file behind.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from wato_common import manifest as common_manifest
from wato_common.artifact_store import (
    chunk_root,
    detections_3d_path,
    ground_path,
    lidar_proc_dir,
    lidar_proc_index_path,
    lifted_instances_path,
    lifted_stats_path,
    local_path,
    motion_clusters_path,
    proposals_path,
    proposal_refinement_path,
    semantic_lifting_dir,
)
from wato_common.artifacts import ground as ground_artifact
from wato_common.artifacts import calibration_refinement as camera_calibration
from wato_common.artifacts import (
    detections_3d,
    lifted_labels,
    motion_proposals,
    proposals,
)
from wato_common.artifacts.ground import GroundGrid
from wato_common.artifacts.lidar_sweeps import ProcessedSweeps
from wato_common.artifacts.proposals import Proposal
from wato_proposal_generation import detector, fusion, lidar_mos, lifted
from wato_proposal_generation import frame_points as frame_points_mod
from wato_proposal_generation.config import ComponentConfig
from wato_proposal_generation.io import load_chunks, write_refinement_audit

log = logging.getLogger(__name__)


def _check_inputs(bag_id: str, chunk_id: str, with_detector: bool) -> None:
    """Name the upstream step to run when a chunk's inputs are missing."""
    required = {
        "lidar_proc_index.parquet (run lidar_preprocessing)": lidar_proc_index_path(
            bag_id, chunk_id
        ),
        "ground.npz (run lidar_preprocessing)": ground_path(bag_id, chunk_id),
        "motion_clusters.parquet (run lidar_preprocessing's proposals step)": (
            motion_clusters_path(bag_id, chunk_id)
        ),
    }
    missing = [n for n, uri in required.items() if not os.path.exists(local_path(uri))]
    if not lifted_labels.exists(bag_id, chunk_id):
        missing.append("lifted labels (run semantic_lifting)")
    if with_detector and not detections_3d.exists(bag_id, chunk_id):
        missing.append("detections_3d.parquet (run detection_3d)")
    if missing:
        raise FileNotFoundError(
            f"chunk {chunk_id!r} of bag {bag_id!r} is missing {', '.join(missing)}"
        )


def _ground(bag_id: str, chunk_id: str) -> Optional[GroundGrid]:
    try:
        return ground_artifact.read(bag_id, chunk_id)
    except ground_artifact.GroundUnavailableError as exc:
        log.warning(
            "chunk %s: no ground grid (status=%r) — boxes keep their own bottoms",
            chunk_id,
            exc.status,
        )
        return None


def process_chunk(cfg: ComponentConfig, bag_id: str, chunk_id: str) -> dict[str, int]:
    """Write the chunk's proposals.parquet; returns rows per provenance."""
    _check_inputs(bag_id, chunk_id, cfg.fusion_detector.enabled)
    sweeps = ProcessedSweeps.load(bag_id, chunk_id)
    frames = sweeps.frames(cfg.frames.canonical_lidar)
    ground = _ground(bag_id, chunk_id)
    clusters = motion_proposals.read_clusters(bag_id, chunk_id)
    shapes = lidar_mos.track_shapes(clusters)
    gated = lidar_mos.gate(clusters, shapes, cfg.lidar_mos)
    clusters_at = lidar_mos.by_frame(gated, frames)
    gated_ids = np.array(sorted(c.cluster_id for c in gated), dtype=np.int64)
    instances = {
        i.instance_id: i for i in lifted_labels.read_instances(bag_id, chunk_id)
    }
    detections_at: dict[int, list] = {}
    if cfg.fusion_detector.enabled:
        _, dets = detections_3d.read(bag_id, chunk_id)
        for d in dets:
            detections_at.setdefault(d.sweep_id, []).append(d)

    rows: list[Proposal] = []
    audit: list[dict] = []
    for frame in frames:
        fp = frame_points_mod.load(bag_id, chunk_id, frame, sweeps, gated_ids)
        refined = lifted.refine_support(fp, instances, cfg.lifted)
        lifts = lifted.proposals(frame, fp, ground, instances, cfg.lifted, refined)
        semantic = lifted.semantic_view(fp, lifts)
        mos = lidar_mos.proposals(
            frame,
            clusters_at.get(frame.reference_sweep_id, []),
            semantic,
            ground,
            instances,
            shapes,
            cfg.lidar_mos,
        )
        by_id = {b.proposal.proposal_id: b.proposal for b in lifts}
        for iid, result in refined.items():
            p = by_id.get(f"lifted-{iid}-{frame.reference_sweep_id:06d}")
            shared = np.intersect1d(result.seed_indices, result.point_indices).size
            audit.append(
                dict(
                    sweep_id=frame.reference_sweep_id,
                    instance_id=iid,
                    support_method=cfg.lifted.support_method,
                    status=result.status,
                    best_radius_m=result.radius_m,
                    best_cluster_id=result.cluster_id,
                    best_iou=result.iou,
                    n_seed_points=int(result.seed_indices.size),
                    n_refined_points=int(result.point_indices.size),
                    n_added_points=int(result.point_indices.size - shared),
                    n_removed_points=int(result.seed_indices.size - shared),
                    emitted=p is not None,
                    n_box_points=p.lidar_point_count if p is not None else None,
                    n_seed_points_in_box=(
                        p.lift_seed_point_count if p is not None else None
                    ),
                )
            )
        dets = detector.proposals(
            frame,
            detections_at.get(frame.reference_sweep_id, []),
            fp,
            cfg.fusion_detector,
        )
        rows += [b.proposal for b in mos] + [b.proposal for b in lifts]
        rows += [b.proposal for b in dets]
        rows += fusion.fuse(
            frame,
            mos,
            lifts,
            fp,
            ground,
            cfg.fusion,
            detector=dets,
            add_unmatched_detector=cfg.fusion_detector.add_unmatched,
            unmatched_detector_min_score=cfg.fusion_detector.unmatched_min_score,
            unmatched_detector_min_points=cfg.fusion_detector.unmatched_min_points,
        )
    # proposals.parquet remains the completion marker: diagnostics must exist
    # before it is written, including for instances later dropped by box gates.
    write_refinement_audit(bag_id, chunk_id, audit)
    proposals.write(bag_id, chunk_id, rows)
    counts = fusion.counts(rows)
    log.info("chunk %s: %d frames, proposals %s", chunk_id, len(frames), counts)
    return counts


def _write_chunk_manifest(
    bag_id: str, chunk_id: str, config_path: Optional[str], with_detector: bool
) -> None:
    """Record the config and inputs behind this chunk's proposals.
    Best-effort: never fails a chunk that succeeded.

    The per-sweep files are too many to hash one by one; their producers'
    manifests (which hash those producers' own inputs and config) and indexes
    stand in for them."""
    root = chunk_root(bag_id, chunk_id)
    components = ["lidar_preprocessing", "semantic_lifting"]
    if with_detector:
        components.append("detection_3d")
    upstream = {
        f"manifest_{c}": f"{root}/{common_manifest.component_manifest_name(c)}"
        for c in components
    }
    detector_inputs = (
        {"detections_3d": detections_3d_path(bag_id, chunk_id)} if with_detector else {}
    )
    try:
        common_manifest.write(
            component="proposal_generation",
            bag_id=bag_id,
            chunk_id=chunk_id,
            inputs={
                "lidar_proc_index": lidar_proc_index_path(bag_id, chunk_id),
                "ground": ground_path(bag_id, chunk_id),
                "motion_clusters": motion_clusters_path(bag_id, chunk_id),
                "lifted_instances": lifted_instances_path(bag_id, chunk_id),
                "lifted_stats": lifted_stats_path(bag_id, chunk_id),
                **detector_inputs,
                **upstream,
            },
            outputs={
                "proposals": proposals_path(bag_id, chunk_id),
                "refinement_audit": proposal_refinement_path(bag_id, chunk_id),
            },
            config_path=config_path,
            extra={
                "unhashed_inputs": {
                    "world_sweeps_and_motion_proposals": lidar_proc_dir(
                        bag_id, chunk_id
                    ),
                    "lifted_labels": f"{semantic_lifting_dir(bag_id, chunk_id)}/lifted_labels",
                    "lifted_candidates": (
                        f"{semantic_lifting_dir(bag_id, chunk_id)}/lifted_candidates"
                    ),
                }
            },
            filename=common_manifest.component_manifest_name("proposal_generation"),
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

    Skips chunks whose proposals.parquet exists unless force=True.

    Args:
        config_path: the config behind ``cfg``, hashed into each manifest.
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

    n_ok = n_skip = 0
    for chunk in chunks:
        cid = chunk["chunk_id"]
        upstream_calibration = [
            f"{chunk_root(bag_id, cid)}/manifest_semantic_lifting.json"
        ]
        if cfg.fusion_detector.enabled:
            upstream_calibration.append(
                f"{chunk_root(bag_id, cid)}/manifest_detection_3d.json"
            )
        camera_calibration.bind_inherited_outputs(
            bag_id,
            cid,
            "proposal_generation",
            upstream_calibration,
            force=force,
            existing_outputs=proposals.exists(bag_id, cid),
        )
        if not force and proposals.exists(bag_id, cid):
            log.info("chunk %s already complete — skipping (use --force)", cid)
            n_skip += 1
            continue
        process_chunk(cfg, bag_id, cid)
        _write_chunk_manifest(bag_id, cid, config_path, cfg.fusion_detector.enabled)
        n_ok += 1
    log.info(
        "proposal_generation complete for bag %s: %d processed, %d skipped",
        bag_id,
        n_ok,
        n_skip,
    )
