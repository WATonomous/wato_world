"""label_refinement pipeline orchestrator.

Per chunk:
  1. Each frame's non-ground world points, loaded once: every track frame's
     box counts its points, and a parked track gathers the points inside the
     rectangle enclosing all its boxes.
  2. Each track refined (refine.py): a parked vehicle from its gathered points
     (refined as a mover when there are too few), any other track from its
     boxes.
Then every track's refined boxes go to refined_labels.parquet
(wato_common.artifacts.refined_labels), bag-level, owned flags carried from
tracks.parquet.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import numpy as np

from wato_common import manifest as common_manifest
from wato_common.artifact_store import (
    chunk_root,
    refined_labels_path,
    track_summaries_path,
    tracks_path,
)
from wato_common.artifacts import ground as ground_artifact
from wato_common.artifacts import refined_labels, tracks
from wato_common.artifacts.ground import GroundGrid
from wato_common.artifacts.lidar_sweeps import ProcessedSweeps
from wato_common.artifacts.refined_labels import AGGREGATE, PERCENTILE, RefinedBox
from wato_common.artifacts.tracks import PARKED, TrackFrame, TrackSummary
from wato_common.pose_lookup import PoseLookup
from wato_common.schemas import Box3D
from wato_common.tracking import points_in_box
from wato_label_refinement import refine
from wato_label_refinement.config import ComponentConfig

log = logging.getLogger(__name__)

# Margin when counting a box's points: ~3σ of the scanners' range noise.
_COUNT_MARGIN_M = 0.1


def _box(f: TrackFrame) -> Box3D:
    return Box3D(cx=f.cx, cy=f.cy, cz=f.cz, w=f.w, l=f.l, h=f.h, heading=f.heading)


def _ground(bag_id: str, chunk_id: str) -> Optional[GroundGrid]:
    try:
        return ground_artifact.read(bag_id, chunk_id)
    except (FileNotFoundError, ground_artifact.GroundUnavailableError):
        log.warning("chunk %s: no ground grid — boxes keep their own bottoms", chunk_id)
        return None


def gather(
    bag_id: str,
    chunk_id: str,
    frames_by_track: dict[str, list[TrackFrame]],
    parked_regions: dict[str, Box3D],
    canonical_lidar: Optional[str],
) -> tuple[dict[tuple[str, int], int], dict[str, np.ndarray]]:
    """(points per (track, sweep_id) box, accumulated points per parked track)."""
    sweeps = ProcessedSweeps.load(bag_id, chunk_id)
    frame_of = {f.reference_sweep_id: f for f in sweeps.frames(canonical_lidar)}
    at_sweep: dict[int, list[TrackFrame]] = defaultdict(list)
    for fs in frames_by_track.values():
        for f in fs:
            at_sweep[f.sweep_id].append(f)
    counts: dict[tuple[str, int], int] = {}
    parts: dict[str, list[np.ndarray]] = defaultdict(list)
    for sid, tfs in at_sweep.items():
        frame = frame_of.get(sid)
        if frame is None:
            raise KeyError(
                f"chunk {chunk_id}: track frame at sweep {sid} is no frame of "
                f"canonical LiDAR {canonical_lidar!r}; frames.canonical_lidar "
                "differs from the one the proposals were made with"
            )
        pts = []
        for s in frame.sweep_ids:
            w = sweeps.world(s)
            pts.append(w.xyz[~w.ground_mask] if w.ground_mask is not None else w.xyz)
        xyz = np.concatenate(pts)
        for f in tfs:
            counts[(f.track_id, sid)] = int(
                points_in_box(xyz, _box(f), _COUNT_MARGIN_M).sum()
            )
            region = parked_regions.get(f.track_id)
            if region is not None:
                parts[f.track_id].append(xyz[points_in_box(xyz, region)])
    accumulated = {
        tid: np.concatenate(p) if p else np.empty((0, 3)) for tid, p in parts.items()
    }
    return counts, accumulated


def process_chunk(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
    frames: list[TrackFrame],
    summaries: list[TrackSummary],
) -> list[RefinedBox]:
    by_track: dict[str, list[TrackFrame]] = defaultdict(list)
    for f in frames:
        by_track[f.track_id].append(f)
    summary_of = {s.track_id: s for s in summaries}
    parked_regions = {
        tid: refine.enclosing_region(
            [_box(f) for f in fs],
            cfg.parked.region_margin_m,
            cfg.parked.region_max_centre_offset_m,
        )
        for tid, fs in by_track.items()
        if summary_of[tid].motion_state == PARKED
        and summary_of[tid].cls in cfg.parked.aggregate_classes
    }
    counts, accumulated = gather(
        bag_id, chunk_id, by_track, parked_regions, cfg.frames.canonical_lidar
    )
    ground = _ground(bag_id, chunk_id)
    poses = PoseLookup.load(bag_id, chunk_id)

    out: list[RefinedBox] = []
    n_aggregate = 0
    for tid, fs in by_track.items():
        s = summary_of[tid]
        parked_box = (
            refine.refine_parked(accumulated[tid], ground, cfg.parked)
            if tid in parked_regions
            else None
        )
        if parked_box is not None:
            n_aggregate += 1
            boxes = [parked_box] * len(fs)
            directed = [False] * len(fs)
            residual, source = None, AGGREGATE
        else:
            ego = np.array(
                [poses.at(f.reference_timestamp_ns).world_T_ego[:2, 3] for f in fs]
            )
            r = refine.refine_moving(
                np.array([f.reference_timestamp_ns for f in fs], dtype=np.int64),
                [_box(f) for f in fs],
                np.array([counts[(tid, f.sweep_id)] for f in fs]),
                ego,
                ground,
                cfg.moving,
                cls=s.cls,
            )
            boxes, directed = r.boxes, r.heading_directed
            residual, source = r.residual_smoothness, PERCENTILE
        confidence = float(np.median([f.score for f in fs]))
        for f, b, d in zip(fs, boxes, directed):
            out.append(
                RefinedBox(
                    track_id=tid,
                    chunk_id=chunk_id,
                    sweep_id=f.sweep_id,
                    reference_timestamp_ns=f.reference_timestamp_ns,
                    cx=b.cx,
                    cy=b.cy,
                    cz=b.cz,
                    w=b.w,
                    l=b.l,
                    h=b.h,
                    heading=b.heading,
                    heading_directed=d,
                    cls=s.cls,
                    motion_state=s.motion_state,
                    size_source=source,
                    confidence=confidence,
                    owned=f.owned,
                    residual_smoothness=residual,
                )
            )
    log.info(
        "chunk %s: %d tracks refined (%d parked from their accumulated points)",
        chunk_id,
        len(by_track),
        n_aggregate,
    )
    return out


def _write_chunk_manifest(
    bag_id: str, chunk_id: str, config_path: Optional[str]
) -> None:
    """Best-effort: never fails a run that succeeded."""
    root = chunk_root(bag_id, chunk_id)
    try:
        common_manifest.write(
            component="label_refinement",
            bag_id=bag_id,
            chunk_id=chunk_id,
            inputs={
                "tracks": tracks_path(bag_id),
                "track_summaries": track_summaries_path(bag_id),
                "manifest_lidar_preprocessing": (
                    f"{root}/{common_manifest.component_manifest_name('lidar_preprocessing')}"
                ),
            },
            outputs={"refined_labels": refined_labels_path(bag_id)},
            config_path=config_path,
            filename=common_manifest.component_manifest_name("label_refinement"),
        )
    except Exception:  # noqa: BLE001
        log.warning("failed to write manifest for chunk %s", chunk_id, exc_info=True)


def run(
    cfg: ComponentConfig,
    *,
    bag_id: str,
    force: bool = False,
    config_path: Optional[str] = None,
) -> None:
    """Refine every track of the bag. Skips when refined labels exist unless
    force=True."""
    if refined_labels.exists(bag_id) and not force:
        log.info("bag %s already refined — skipping (use --force)", bag_id)
        return
    if not tracks.exists(bag_id):
        raise FileNotFoundError(f"bag {bag_id!r} has no tracks.parquet (run tracking)")
    frames, summaries = tracks.read(bag_id)
    by_chunk: dict[str, list[TrackFrame]] = defaultdict(list)
    for f in frames:
        by_chunk[f.chunk_id].append(f)
    out: list[RefinedBox] = []
    for cid in sorted(by_chunk):
        out += process_chunk(
            cfg, bag_id, cid, by_chunk[cid], [s for s in summaries if s.chunk_id == cid]
        )
    refined_labels.write(bag_id, out)
    for cid in sorted(by_chunk):
        _write_chunk_manifest(bag_id, cid, config_path)
    log.info("label_refinement complete for bag %s: %d boxes", bag_id, len(out))
