"""proposal_generation pipeline orchestrator.

Per chunk, frame by frame (lidar_preprocessing's frame_sync frames, regrouped
with ProcessedSweeps.frames on the configured canonical LiDAR):
  1. Load the frame's points that a source can claim: semantic_lifting's
     labelled points and the gated Step F clusters' (frame_points.py).
  2. lidar_mos: one box per gated Step F cluster, classed by its lifted
     labels (lidar_mos.py).
  3. lifted: one box per lifted instance, around its largest group of points
     (lifted.py).
  4. fused: one box per object, merging a lidar_mos and a lifted box that
     share points (fusion.py).
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
    ground_path,
    lidar_proc_dir,
    lidar_proc_index_path,
    lifted_instances_path,
    lifted_stats_path,
    local_path,
    motion_clusters_path,
    proposals_path,
    semantic_lifting_dir,
)
from wato_common.artifacts import ground as ground_artifact
from wato_common.artifacts import lifted_labels, motion_proposals, proposals
from wato_common.artifacts.ground import GroundGrid
from wato_common.artifacts.lidar_sweeps import ProcessedSweeps
from wato_common.artifacts.proposals import Proposal
from wato_proposal_generation import fusion, lidar_mos, lifted
from wato_proposal_generation import frame_points as frame_points_mod
from wato_proposal_generation.config import ComponentConfig
from wato_proposal_generation.io import load_chunks

log = logging.getLogger(__name__)


def _check_inputs(bag_id: str, chunk_id: str) -> None:
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
    _check_inputs(bag_id, chunk_id)
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

    rows: list[Proposal] = []
    for frame in frames:
        fp = frame_points_mod.load(bag_id, chunk_id, frame, sweeps, gated_ids)
        mos = lidar_mos.proposals(
            frame,
            clusters_at.get(frame.reference_sweep_id, []),
            fp,
            ground,
            instances,
            shapes,
            cfg.lidar_mos,
        )
        lifts = lifted.proposals(frame, fp, ground, instances, cfg.lifted)
        rows += [b.proposal for b in mos] + [b.proposal for b in lifts]
        rows += fusion.fuse(frame, mos, lifts, fp, ground, cfg.fusion)
    proposals.write(bag_id, chunk_id, rows)
    counts = fusion.counts(rows)
    log.info("chunk %s: %d frames, proposals %s", chunk_id, len(frames), counts)
    return counts


def _write_chunk_manifest(
    bag_id: str, chunk_id: str, config_path: Optional[str]
) -> None:
    """Record the config and inputs behind this chunk's proposals.
    Best-effort: never fails a chunk that succeeded.

    The per-sweep files are too many to hash one by one; their producers'
    manifests (which hash those producers' own inputs and config) and indexes
    stand in for them."""
    root = chunk_root(bag_id, chunk_id)
    upstream = {
        f"manifest_{c}": f"{root}/{common_manifest.component_manifest_name(c)}"
        for c in ("lidar_preprocessing", "semantic_lifting")
    }
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
                **upstream,
            },
            outputs={"proposals": proposals_path(bag_id, chunk_id)},
            config_path=config_path,
            extra={
                "unhashed_inputs": {
                    "world_sweeps_and_motion_proposals": lidar_proc_dir(
                        bag_id, chunk_id
                    ),
                    "lifted_labels": f"{semantic_lifting_dir(bag_id, chunk_id)}/lifted_labels",
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
        if not force and proposals.exists(bag_id, cid):
            log.info("chunk %s already complete — skipping (use --force)", cid)
            n_skip += 1
            continue
        process_chunk(cfg, bag_id, cid)
        _write_chunk_manifest(bag_id, cid, config_path)
        n_ok += 1
    log.info(
        "proposal_generation complete for bag %s: %d processed, %d skipped",
        bag_id,
        n_ok,
        n_skip,
    )
