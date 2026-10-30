"""Semantic lifting pipeline orchestrator.

Per chunk, sweeps in time order:
  Pass 1 — for each sweep, for each camera frame ingest paired with it
  (frame_index):
    a. Skip the frame when it has no masklets, or no depth artifact
       (perception_2d writes none when its depth fit failed).
    b. Project the sweep's world-frame points with the camera's pose at the
       frame's own time (points are world-registered upstream, so no
       sweep-pose / ego-motion term).
    c. Look up the mask each point lands in (innermost mask wins).
    d. Keep the in-mask points that pass the visibility test (UniLiPs Eq. 1)
       against the depth map up close and the sweep's own LiDAR returns.
    e. Each such point gets one vote from this camera, weighted by the
       claiming masklet's det_score.
  Then merge one object's masklets across cameras into one instance, from the
  points they both claim (instances.py).
  Pass 2 — resolve each sweep's votes across cameras (voting.py) and write
  lifted_labels/<sweep_id:06d>.npz, then lifted_instances.parquet and
  lifted_stats.parquet (wato_common.artifacts.lifted_labels).

A failure in one chunk propagates: an upstream artifact that is missing or
malformed stops the run rather than leaving partial labels behind.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np

from wato_common import manifest as common_manifest
from wato_common.artifact_store import (
    calibration_path,
    chunk_root,
    depth_2d_dir,
    frame_index_path,
    lidar_proc_dir,
    lidar_proc_index_path,
    lifted_instances_path,
    lifted_stats_path,
    local_path,
    masks_2d_dir,
    poses_path,
    semantic_lifting_dir,
    tracklets_2d_path,
)
from wato_common.artifacts import ArtifactFormatError, lifted_labels
from wato_common.artifacts.calibration import Calibration
from wato_common.artifacts.lidar_sweeps import ProcessedSweeps, WorldSweep
from wato_common.artifacts.masklets import MaskletIndex, MaskletRecord
from wato_common.schemas import ProcessedSweepMeta
from wato_semantic_lifting.config import ComponentConfig
from wato_semantic_lifting.frames import FrameCache
from wato_semantic_lifting.instances import CoClaims, merge_masklets
from wato_semantic_lifting.io import CameraFrameRef, load_chunks, load_sweep_frames
from wato_semantic_lifting.projection import (
    compute_cam_T_world,
    pixel_indices,
    project_and_clip,
)
from wato_semantic_lifting.stats import build_stats_row
from wato_semantic_lifting.visibility import lidar_occluder, visible
from wato_semantic_lifting.voting import Resolved, Votes, resolve

log = logging.getLogger(__name__)

# A frame is paired with the sweeps nearest to it in time (~5 on WATO), so
# with sweeps processed in time order a few cached frames per camera suffice.
_FRAMES_CACHED_PER_CAMERA = 4
# The depth artifact has no per-pixel confidence, so a vote's weight is the
# masklet's det_score alone; the floor lets a zero-scored mask break ties.
_MIN_VOTE_SCORE = 1e-3


@dataclass(frozen=True)
class _SweepVotes:
    sweep_id: int
    n_points: int
    votes: Votes
    n_in_any_mask: int
    n_cameras_used: int


class _ChunkLifter:
    """Pass 1 for one chunk: every sweep's per-camera votes."""

    def __init__(
        self,
        cfg: ComponentConfig,
        bag_id: str,
        chunk_id: str,
        calibration: Calibration,
        masklets: MaskletIndex,
    ):
        self.cfg = cfg
        self.calibration = calibration
        self.cam_index = {c: i for i, c in enumerate(sorted(calibration.cameras))}
        self.vote_score = np.array(
            [max(r.det_score, _MIN_VOTE_SCORE) for r in masklets.records], np.float64
        )
        self.frames = FrameCache(
            bag_id,
            chunk_id,
            masklets,
            neighborhood=cfg.visibility.neighborhood,
            max_frames=_FRAMES_CACHED_PER_CAMERA * max(1, len(self.cam_index)),
        )

    def sweep(
        self, sweep: ProcessedSweepMeta, world: WorldSweep, refs: list[CameraFrameRef]
    ) -> _SweepVotes:
        if world.ground_mask is None:
            raise ArtifactFormatError(
                local_path(sweep.world_path),
                "no ground_mask: the LiDAR occlusion test needs it (ground must "
                "not hide what stands on it); lidar_preprocessing writes it "
                "with Patchwork++",
            )
        xyz = world.xyz
        can_occlude = ~world.ground_mask
        vis_cfg = self.cfg.visibility
        n = int(xyz.shape[0])
        in_mask = np.zeros(n, dtype=bool)
        parts: list[Votes] = []
        n_cameras = 0
        for ref in refs:
            if n == 0:
                break
            frame = self.frames.get(ref)
            if frame is None:
                continue
            cam = self.calibration.camera(ref.cam_id)
            W, H = frame.size
            uv, d_cam, valid = project_and_clip(
                xyz, cam.K, compute_cam_T_world(ref.world_T_ego, cam.ego_T_cam), (W, H)
            )
            n_cameras += 1
            idx = np.flatnonzero(valid)
            ui, vi = pixel_indices(uv[idx], W, H)
            label = frame.labels[vi, ui]
            hit = label >= 0
            if not hit.any():
                continue
            # Every projected return can occlude, not only the in-mask ones.
            lidar_occ = lidar_occluder(
                ui,
                vi,
                d_cam[idx],
                can_occlude[idx],
                (W, H),
                vis_cfg.lidar_cell_px,
                vis_cfg.lidar_neighborhood_cells,
            )[hit]
            idx, ui, vi, label = idx[hit], ui[hit], vi[hit], label[hit]
            in_mask[idx] = True
            vis = visible(d_cam[idx], frame.occluder_m[vi, ui], lidar_occ, vis_cfg)
            parts.append(
                Votes(
                    point_idx=idx[vis],
                    cam_idx=np.full(int(vis.sum()), self.cam_index[ref.cam_id]),
                    label_idx=label[vis].astype(np.int64),
                    score=self.vote_score[label[vis]],
                )
            )
        return _SweepVotes(
            sweep.sweep_id, n, Votes.concat(parts), int(in_mask.sum()), n_cameras
        )


def _write_sweep_labels(
    bag_id: str,
    chunk_id: str,
    sweep_id: int,
    resolved: Resolved,
    instance_ids: np.ndarray,
    classes: np.ndarray,
) -> None:
    if resolved.point_idx.size == 0:
        labels = lifted_labels.SweepLabels.empty()
    else:
        labels = lifted_labels.SweepLabels(
            point_indices=resolved.point_idx.astype(np.int32),
            instance_ids=instance_ids[resolved.instance],
            classes=classes[resolved.cls],
            confidences=resolved.confidence.astype(np.float32),
            n_supporting_cameras=resolved.n_supporting_cameras.astype(np.int32),
            n_disagreeing_cameras=resolved.n_disagreeing_cameras.astype(np.int32),
        )
    lifted_labels.write_sweep(bag_id, chunk_id, sweep_id, labels)


def _instances_table(
    records: list[MaskletRecord],
    label_instance: np.ndarray,
    per_sweep: list[Resolved],
) -> list[lifted_labels.LiftedInstance]:
    """One LiftedInstance per instance some point is labeled with."""
    n_points: Counter[int] = Counter()
    n_sweeps: Counter[int] = Counter()
    for resolved in per_sweep:
        inst, counts = np.unique(resolved.instance, return_counts=True)
        n_points.update(dict(zip(inst.tolist(), counts.tolist())))
        n_sweeps.update(inst.tolist())
    members: dict[int, list[int]] = {}
    for label, inst in enumerate(label_instance.tolist()):
        members.setdefault(inst, []).append(label)
    return [
        lifted_labels.LiftedInstance(
            instance_id=records[inst].masklet_id,
            cls=records[inst].cls,
            masklet_ids=tuple(records[m].masklet_id for m in members[inst]),
            cam_ids=tuple(records[m].cam_id for m in members[inst]),
            n_points_labeled=n_points[inst],
            n_sweeps=n_sweeps[inst],
        )
        for inst in sorted(n_points, key=lambda i: records[i].masklet_id)
    ]


def _process_chunk(
    cfg: ComponentConfig, bag_id: str, chunk_id: str, calibration: Calibration
) -> None:
    """Run semantic lifting for all sweeps in one chunk."""
    sweeps = ProcessedSweeps.load(bag_id, chunk_id)
    if not sweeps.rows:
        log.warning("semantic_lifting: chunk %s has no valid sweeps", chunk_id)
        lifted_labels.write_instances(bag_id, chunk_id, [])
        lifted_labels.write_stats(bag_id, chunk_id, [])
        return

    frames_by_sweep = load_sweep_frames(bag_id, chunk_id)
    masklets = MaskletIndex.load(bag_id, chunk_id)
    lifter = _ChunkLifter(cfg, bag_id, chunk_id, calibration, masklets)

    # Pass 1: every sweep's votes, and which masklets claim the same points.
    co_claims = CoClaims()
    collected = []
    for s in sorted(sweeps.rows, key=lambda r: (r.reference_timestamp_ns, r.sweep_id)):
        world = sweeps.world(s.sweep_id)
        sv = lifter.sweep(s, world, frames_by_sweep.get(s.sweep_id, []))
        co_claims.add(s.sweep_id, sv.votes, world.xyz)
        collected.append(sv)

    records = masklets.records
    label_instance = merge_masklets(
        records,
        co_claims,
        min_shared_points=cfg.instances.min_shared_points,
        min_shared_fraction=cfg.instances.min_shared_fraction,
        max_separation_m=cfg.instances.max_separation_m,
    )
    classes = np.array(sorted({r.cls for r in records}), dtype=np.str_)
    cls_index = {c: i for i, c in enumerate(classes.tolist())}
    label_cls = np.array([cls_index[r.cls] for r in records], np.int64)
    instance_ids = np.array([r.masklet_id for r in records], dtype=np.str_)

    # Pass 2: resolve and write.
    stats_rows = []
    per_sweep: list[Resolved] = []
    for sv in sorted(collected, key=lambda sv: sv.sweep_id):
        resolved = resolve(sv.votes, label_cls, label_instance)
        _write_sweep_labels(
            bag_id, chunk_id, sv.sweep_id, resolved, instance_ids, classes
        )
        per_sweep.append(resolved)
        stats_rows.append(
            build_stats_row(
                bag_id,
                chunk_id,
                sv.sweep_id,
                n_points_total=sv.n_points,
                resolved=resolved,
                n_points_in_any_mask=sv.n_in_any_mask,
                n_cameras_used=sv.n_cameras_used,
            )
        )
    instances = _instances_table(records, label_instance, per_sweep)
    lifted_labels.write_instances(bag_id, chunk_id, instances)
    lifted_labels.write_stats(bag_id, chunk_id, stats_rows)
    log.info(
        "semantic_lifting: chunk %s done — %d sweeps, %d labeled points, "
        "%d instances (%d spanning cameras), %d frames with masklets but no depth",
        chunk_id,
        len(stats_rows),
        sum(r.point_idx.size for r in per_sweep),
        len(instances),
        sum(len(i.masklet_ids) > 1 for i in instances),
        lifter.frames.n_no_depth,
    )


def _write_chunk_manifest(
    bag_id: str, chunk_id: str, config_path: Optional[str]
) -> None:
    """Record the config and inputs behind this chunk's lifted labels.
    Best-effort: never fails a chunk that succeeded.

    The world NPZs and mask PNGs are too many to hash one by one; their
    producers' manifests (which hash those producers' own inputs and config)
    and indexes stand in for them."""
    root = chunk_root(bag_id, chunk_id)
    upstream = {
        f"manifest_{c}": f"{root}/{common_manifest.component_manifest_name(c)}"
        for c in ("lidar_preprocessing", "perception_2d")
    }
    try:
        common_manifest.write(
            component="semantic_lifting",
            bag_id=bag_id,
            chunk_id=chunk_id,
            inputs={
                "calibration": calibration_path(bag_id),
                "frame_index": frame_index_path(bag_id, chunk_id),
                "poses": poses_path(bag_id, chunk_id),
                "lidar_proc_index": lidar_proc_index_path(bag_id, chunk_id),
                "tracklets_2d": tracklets_2d_path(bag_id, chunk_id),
                **upstream,
            },
            outputs={
                "lifted_labels": f"{semantic_lifting_dir(bag_id, chunk_id)}/lifted_labels",
                "lifted_instances": lifted_instances_path(bag_id, chunk_id),
                "lifted_stats": lifted_stats_path(bag_id, chunk_id),
            },
            config_path=config_path,
            extra={
                "unhashed_inputs": {
                    "world_sweeps": lidar_proc_dir(bag_id, chunk_id),
                    "masks_2d": masks_2d_dir(bag_id, chunk_id),
                    "depth_2d": depth_2d_dir(bag_id, chunk_id),
                }
            },
            filename=common_manifest.component_manifest_name("semantic_lifting"),
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

    Skips chunks whose lifted_stats.parquet exists unless force=True.

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

    calibration = Calibration.load(bag_id)

    n_ok = n_skip = 0
    for chunk in chunks:
        cid = chunk["chunk_id"]
        if not force and lifted_labels.exists(bag_id, cid):
            log.info("chunk %s already complete — skipping (use --force)", cid)
            n_skip += 1
            continue
        _process_chunk(cfg, bag_id, cid, calibration)
        _write_chunk_manifest(bag_id, cid, config_path)
        n_ok += 1

    log.info(
        "semantic_lifting complete for bag %s: %d processed, %d skipped",
        bag_id,
        n_ok,
        n_skip,
    )
