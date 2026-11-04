"""label_refinement pipeline orchestrator.

1. Each track's frames, one per instant (tracks.per_instant: the owned frame
   where two chunks saw it), are what is refined; a track linked across
   chunks is refined whole.
2. SLF's views of each track of its classes (views.py): the images where its
   masklets have masks.
3. Per chunk, each such frame's non-ground world points, loaded once: every
   frame's box counts its points, a parked track gathers the points inside
   the rectangle enclosing all its boxes, and a frame SLF views keeps the
   returns on its observed box.
4. Each track refined (refine.py): a parked vehicle from its gathered points
   (refined as a mover when there are too few), any other track from its
   boxes, each frame against its own chunk's ground grid and poses. Then SLF
   (slf.py) fits the track's size to its views and points; where the fit
   passes its gates, the evaluated sizes and positions are written directly.
Then every tracks.parquet row gets its instant's refined box in
refined_labels.parquet (wato_common.artifacts.refined_labels), bag-level, with
owned carried from tracks.parquet.
"""

from __future__ import annotations

import logging
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from wato_common import manifest as common_manifest
from wato_common.artifact_store import (
    chunk_root,
    frame_index_path,
    refined_labels_path,
    track_summaries_path,
    tracks_path,
)
from wato_common.artifacts import ground as ground_artifact
from wato_common.artifacts import refined_labels, tracks
from wato_common.artifacts.calibration import Calibration
from wato_common.artifacts.ground import GroundGrid
from wato_common.artifacts.lidar_sweeps import ProcessedSweeps
from wato_common.artifacts.refined_labels import (
    AGGREGATE,
    PERCENTILE,
    SLF,
    RefinedBox,
)
from wato_common.artifacts.tracks import PARKED, TrackFrame, TrackSummary
from wato_common.pose_lookup import PoseLookup
from wato_common.schemas import Box3D
from wato_common.tracking import points_in_box
from wato_label_refinement import refine, views
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
    frames: list[TrackFrame],
    parked_regions: dict[str, Box3D],
    canonical_lidar: Optional[str],
    voxel_m: float,
    keep_points: Optional[set[tuple[str, int]]] = None,
    point_margin_m: float = 0.0,
) -> tuple[
    dict[tuple[str, int], int], dict[str, np.ndarray], dict[tuple[str, int], np.ndarray]
]:
    """For the chunk's frames: (points per (track, sweep_id) box, each parked
    track's points inside its region, one per voxel_m cell, and for each
    (track, sweep_id) in keep_points the returns on its observed box grown by
    point_margin_m)."""
    sweeps = ProcessedSweeps.load(bag_id, chunk_id)
    frame_of = {f.reference_sweep_id: f for f in sweeps.frames(canonical_lidar)}
    at_sweep: dict[int, list[TrackFrame]] = defaultdict(list)
    for f in frames:
        at_sweep[f.sweep_id].append(f)
    counts: dict[tuple[str, int], int] = {}
    parts: dict[str, list[np.ndarray]] = defaultdict(list)
    kept: dict[tuple[str, int], np.ndarray] = {}
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
            if keep_points and (f.track_id, sid) in keep_points:
                kept[(f.track_id, sid)] = views.object_points(
                    xyz, _box(f), point_margin_m
                )
    accumulated = {
        tid: refine.one_per_voxel(np.concatenate(p), voxel_m)
        for tid, p in parts.items()
    }
    return counts, accumulated, kept


@dataclass(frozen=True)
class Phase3:
    """One track refined training-free, one box per frame (one per instant)."""

    boxes: list[Box3D]
    directed: list[bool]
    residual_smoothness: Optional[float]
    source: str  # AGGREGATE or PERCENTILE
    velocity: np.ndarray  # (N, 2) m/s, zero for a parked box
    ego_xy: np.ndarray  # (N, 2)


def refine_track(
    cfg: ComponentConfig,
    frames: list[TrackFrame],
    summary: TrackSummary,
    counts: dict[tuple[str, str, int], int],
    points: Optional[np.ndarray],
    grounds: dict[str, Optional[GroundGrid]],
    poses: dict[str, PoseLookup],
) -> Phase3:
    """frames one per instant; points is the parked region's gathered cloud
    (None for a track not aggregated)."""
    ego = np.array(
        [
            poses[f.chunk_id].at(f.reference_timestamp_ns).world_T_ego[:2, 3]
            for f in frames
        ]
    )
    home = Counter(f.chunk_id for f in frames).most_common(1)[0][0]
    parked_box = (
        refine.refine_parked(points, grounds[home], cfg.parked)
        if points is not None
        else None
    )
    if parked_box is not None:
        n = len(frames)
        return Phase3(
            [parked_box] * n, [False] * n, None, AGGREGATE, np.zeros((n, 2)), ego
        )
    r = moving(cfg, frames, summary, counts, ego, grounds)
    return Phase3(
        r.boxes, r.heading_directed, r.residual_smoothness, PERCENTILE, r.velocity, ego
    )


def moving(cfg, frames, summary, counts, ego, grounds, size=None, offset=(0.0, 0.0)):
    return refine.refine_moving(
        np.array([f.reference_timestamp_ns for f in frames], dtype=np.int64),
        [_box(f) for f in frames],
        np.array([counts[(f.track_id, f.chunk_id, f.sweep_id)] for f in frames]),
        ego,
        [grounds[f.chunk_id] for f in frames],
        cfg.moving,
        cls=summary.cls,
        size=size,
        offset=offset,
    )


@dataclass(frozen=True)
class SlfOutcome:
    accepted: bool
    residual_silhouette: Optional[float]
    residual_lidar_fit: Optional[float]


def _value(x: float) -> Optional[float]:
    return None if math.isnan(x) else float(x)


def slf_problem(
    cfg: ComponentConfig,
    frames: list[TrackFrame],
    summary: TrackSummary,
    p3: Phase3,
    selected: list[views.View],
    points: dict[int, np.ndarray],
    chunks: dict[str, views.Chunk],
    calib: Calibration,
):
    """The track's SLF fit (views.problem), or None when fewer than
    slf.min_views of its views could be built."""
    prob = views.problem(
        frames[0].track_id,
        summary.cls,
        frames,
        p3.boxes,
        p3.velocity,
        p3.ego_xy,
        p3.source != AGGREGATE and summary.cls not in cfg.moving.centred_classes,
        selected,
        points,
        chunks,
        calib,
        cfg.slf,
    )
    return prob if prob.origin.shape[0] >= cfg.slf.min_views else None


def apply_slf(
    cfg: ComponentConfig,
    frames: list[TrackFrame],
    summary: TrackSummary,
    p3: Phase3,
    fit,
    counts: dict[tuple[str, str, int], int],
    grounds: dict[str, Optional[GroundGrid]],
) -> tuple[Phase3, SlfOutcome]:
    """Write SLF's evaluated geometry where it passes its gates; otherwise
    keep Phase 3 and its own residuals on the views."""
    gain = fit.start_residual_silhouette - fit.residual_silhouette
    need = cfg.slf.min_silhouette_gain_by_class.get(
        summary.cls, cfg.slf.min_silhouette_gain
    )
    lidar_ok = (
        math.isnan(fit.residual_lidar)
        or fit.residual_lidar <= cfg.slf.max_residual_lidar_m
    )
    if not (
        gain >= need
        and fit.residual_silhouette <= cfg.slf.max_residual_silhouette
        and lidar_ok
    ):
        return p3, SlfOutcome(
            False,
            _value(fit.start_residual_silhouette),
            _value(fit.start_residual_lidar),
        )
    length, width, height = (float(x) for x in fit.size)
    centres = np.asarray(fit.centres)
    if centres.shape != (len(frames), 3) or not np.isfinite(centres).all():
        raise ValueError(
            "SLF must return a finite evaluated centre for every track frame"
        )
    # Residuals describe these exact positions. Re-running moving() here
    # would smooth a different trajectory and re-sample ground the fitter
    # held fixed, invalidating both the optimization and its quality gates.
    boxes = [
        Box3D(
            cx=float(c[0]),
            cy=float(c[1]),
            cz=float(c[2]),
            w=width,
            l=length,
            h=height,
            heading=b.heading,
        )
        for b, c in zip(p3.boxes, centres)
    ]
    new = replace(p3, boxes=boxes, source=SLF)
    return new, SlfOutcome(
        True, _value(fit.residual_silhouette), _value(fit.residual_lidar)
    )


def rows(
    frames: list[TrackFrame],
    summary: TrackSummary,
    p3: Phase3,
    outcome: Optional[SlfOutcome],
) -> list[RefinedBox]:
    confidence = float(np.median([f.score for f in frames]))
    return [
        RefinedBox(
            track_id=f.track_id,
            chunk_id=f.chunk_id,
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
            cls=summary.cls,
            motion_state=summary.motion_state,
            size_source=p3.source,
            confidence=confidence,
            owned=f.owned,
            residual_silhouette=outcome.residual_silhouette if outcome else None,
            residual_lidar_fit=outcome.residual_lidar_fit if outcome else None,
            residual_smoothness=p3.residual_smoothness,
        )
        for f, b, d in zip(frames, p3.boxes, p3.directed)
    ]


def _write_chunk_manifest(
    bag_id: str, chunk_id: str, config_path: Optional[str], with_slf: bool
) -> None:
    """Best-effort: never fails a run that succeeded."""
    root = chunk_root(bag_id, chunk_id)
    slf_inputs = (
        {
            "frame_index": frame_index_path(bag_id, chunk_id),
            "manifest_perception_2d": (
                f"{root}/{common_manifest.component_manifest_name('perception_2d')}"
            ),
        }
        if with_slf
        else {}
    )
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
                **slf_inputs,
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
    rows_of: dict[str, list[TrackFrame]] = defaultdict(list)
    for f in frames:
        rows_of[f.track_id].append(f)
    line = {tid: tracks.per_instant(fs) for tid, fs in rows_of.items()}
    summary_of = {s.track_id: s for s in summaries}
    parked_regions = {
        tid: refine.enclosing_region(
            [_box(f) for f in fs],
            cfg.parked.region_margin_m,
            cfg.parked.region_max_centre_offset_m,
        )
        for tid, fs in line.items()
        if summary_of[tid].motion_state == PARKED
        and summary_of[tid].cls in cfg.parked.aggregate_classes
    }
    chunk_ids = sorted({f.chunk_id for fs in line.values() for f in fs})

    # SLF's views, chosen before the points are read: their frames keep theirs.
    slf_tracks = [
        tid
        for tid in line
        if cfg.slf.enabled and summary_of[tid].cls in cfg.slf.classes
    ]
    chunks: dict[str, views.Chunk] = {}
    calib = None
    selected: dict[str, list[views.View]] = {}
    if slf_tracks:
        calib = Calibration.load(bag_id)
        chunks = {cid: views.Chunk.load(bag_id, cid, frames) for cid in chunk_ids}
        for tid in slf_tracks:
            sel = views.select(
                views.candidates(line[tid], chunks, calib, cfg.slf), cfg.slf.max_views
            )
            if len(sel) >= cfg.slf.min_views:
                selected[tid] = sel
    keep = {
        (tid, line[tid][v.frame].chunk_id, line[tid][v.frame].sweep_id)
        for tid, sel in selected.items()
        for v in sel
    }

    counts: dict[tuple[str, str, int], int] = {}
    parts: dict[str, list[np.ndarray]] = defaultdict(list)
    kept: dict[tuple[str, str, int], np.ndarray] = {}
    for cid in chunk_ids:
        c, acc, pts = gather(
            bag_id,
            cid,
            [f for fs in line.values() for f in fs if f.chunk_id == cid],
            parked_regions,
            cfg.frames.canonical_lidar,
            cfg.parked.voxel_m,
            {(tid, sid) for tid, ch, sid in keep if ch == cid},
            cfg.slf.point_margin_m,
        )
        counts.update({(tid, cid, sid): n for (tid, sid), n in c.items()})
        kept.update({(tid, cid, sid): p for (tid, sid), p in pts.items()})
        for tid, p in acc.items():
            parts[tid].append(p)
    grounds = {cid: _ground(bag_id, cid) for cid in chunk_ids}
    poses = {cid: PoseLookup.load(bag_id, cid) for cid in chunk_ids}
    dev = None
    if selected:
        from wato_label_refinement import slf

        dev = slf.device(cfg.slf.device)
        log.info(
            "SLF on %d of %d tracks of its classes, on %s",
            len(selected),
            len(slf_tracks),
            dev,
        )

    p3 = {
        tid: refine_track(
            cfg,
            fs,
            summary_of[tid],
            counts,
            np.concatenate(parts[tid]) if tid in parked_regions else None,
            grounds,
            poses,
        )
        for tid, fs in line.items()
    }

    # SLF, slf.batch_tracks tracks per fit.
    t0 = time.monotonic()
    fits: dict = {}
    batch: list = []

    def flush():
        from wato_label_refinement import slf

        for (tid, _), f in zip(
            batch, slf.fit_many([b for _, b in batch], cfg.slf, dev)
        ):
            fits[tid] = f
        batch.clear()

    for tid, sel in selected.items():
        fs = line[tid]
        on = {
            k: kept[(tid, f.chunk_id, f.sweep_id)]
            for k, f in enumerate(fs)
            if (tid, f.chunk_id, f.sweep_id) in kept
        }
        prob = slf_problem(cfg, fs, summary_of[tid], p3[tid], sel, on, chunks, calib)
        if prob is not None:
            batch.append((tid, prob))
        if len(batch) >= cfg.slf.batch_tracks:
            flush()
    if batch:
        flush()

    out: list[RefinedBox] = []
    n_aggregate = n_linked = n_slf = 0
    for tid, fs in line.items():
        p, outcome = p3[tid], None
        if tid in fits:
            p, outcome = apply_slf(
                cfg, fs, summary_of[tid], p, fits[tid], counts, grounds
            )
            n_slf += outcome.accepted
        n_aggregate += p.source == AGGREGATE
        n_linked += len({f.chunk_id for f in rows_of[tid]}) > 1
        refined = rows(fs, summary_of[tid], p, outcome)
        at = {b.reference_timestamp_ns: b for b in refined}
        # A shared instant's second row (the other chunk's) gets the same box.
        for f in rows_of[tid]:
            b = at[f.reference_timestamp_ns]
            out.append(
                b
                if (b.chunk_id, b.sweep_id) == (f.chunk_id, f.sweep_id)
                else replace(b, chunk_id=f.chunk_id, sweep_id=f.sweep_id, owned=f.owned)
            )
    refined_labels.write(bag_id, out)
    for cid in chunk_ids:
        _write_chunk_manifest(bag_id, cid, config_path, bool(selected))
    log.info(
        "label_refinement complete for bag %s: %d tracks (%d across chunks, %d "
        "parked from their accumulated points, %d SLF fits of which %d kept, "
        "%.0f s), %d boxes",
        bag_id,
        len(line),
        n_linked,
        n_aggregate,
        len(fits),
        n_slf,
        time.monotonic() - t0,
        len(out),
    )
