"""tracking pipeline orchestrator.

1. Per chunk: its fused proposals as per-frame detections (detections.py),
   tracks over them (tracker.py), and the tracks that never exist at once
   fused where they are one object (relink.py), named "<chunk_id>-<n>". With
   `appearance` on, both also read perception_2d's masklet descriptors
   (appearance.py), under one threshold calibrated over the bag.
2. Each chunk's tracks linked to the next chunk's across the span they share
   (stitch.py). A chain of linked tracks is one track of the bag, named after
   its first.
3. Per track, over one observation per instant (the owning chunk's where both
   saw it): class, parked / moving / unknown, kept or dropped (summary.py). A
   track with no owned frame is dropped too: the chunks that label its
   instants didn't see it.
4. Every kept track's frames from every chunk, `owned` set on those in their
   chunk's own half of each shared span (wato_common.chunk_spans).
Then tracks.parquet + track_summaries.parquet (wato_common.artifacts.tracks),
bag-level.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Optional

from wato_common import chunk_spans
from wato_common import manifest as common_manifest
from wato_common.artifact_store import (
    chunk_root,
    chunks_index_path,
    masklet_embeddings_path,
    proposals_path,
    track_summaries_path,
    tracks_path,
)
from wato_common.artifacts import proposals, tracks
from wato_common.artifacts import calibration_refinement as camera_calibration
from wato_common.artifacts.tracks import TrackFrame, TrackSummary
from wato_tracking import appearance, detections, relink, stitch, summary, tracker
from wato_tracking.appearance import Calibration, MemberCost
from wato_tracking.config import ComponentConfig
from wato_tracking.io import load_chunks
from wato_tracking.tracker import Observation

log = logging.getLogger(__name__)


def bag_appearance(
    cfg: ComponentConfig, bag_id: str, chunk_ids: list[str]
) -> tuple[dict[str, MemberCost], Optional[Calibration]]:
    """Each chunk's appearance cost (config `appearance`), and the calibration
    behind its threshold; ({}, None) with appearance off."""
    if not cfg.appearance.enabled:
        return {}, None
    return appearance.for_bag(
        bag_id,
        chunk_ids,
        cfg.appearance.budget,
        cfg.appearance.max_cosine_distance,
    )


def track_chunk(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
    member_cost: Optional[MemberCost] = None,
) -> list[list[Observation]]:
    """The chunk's tracks, each its observations in time order. member_cost:
    the chunk's entry of bag_appearance, None without appearance."""
    det_frames = detections.frames(
        proposals.read(bag_id, chunk_id), cfg.strong, cfg.duplicates
    )
    out = tracker.run(
        det_frames,
        cfg.association,
        member_cost if cfg.appearance.association else None,
    )
    if cfg.relink.enabled:
        out = relink.relink(
            out,
            det_frames,
            cfg.relink,
            cfg.association.max_centre_dist_m,
            member_cost if cfg.appearance.relink else None,
        )
    log.info("chunk %s: %d frames, %d tracks", chunk_id, len(det_frames), len(out))
    return out


def _frame(
    track_id: str, chunk_id: str, o: Observation, window: tuple[float, float]
) -> TrackFrame:
    """The observation as a row; cls is set once the track's class is known."""
    p = o.det.proposal
    return TrackFrame(
        track_id=track_id,
        chunk_id=chunk_id,
        sweep_id=p.sweep_id,
        reference_timestamp_ns=p.reference_timestamp_ns,
        proposal_id=p.proposal_id,
        cx=p.cx,
        cy=p.cy,
        cz=p.cz,
        w=p.w,
        l=p.l,
        h=p.h,
        heading=p.heading,
        cls=None,
        score=p.score,
        owned=window[0] <= p.reference_timestamp_ns < window[1],
        supporting_cam_ids=p.supporting_cam_ids,
        supporting_masklet_ids=p.supporting_masklet_ids,
    )


def link_and_summarize(
    cfg: ComponentConfig,
    index: list[dict],
    per_chunk: dict[str, list[list[Observation]]],
) -> tuple[list[TrackFrame], list[TrackSummary]]:
    """Every chunk's tracks (per_chunk, by chunk_id) as the bag's tracks."""
    order = [r["chunk_id"] for r in sorted(index, key=lambda r: r["t_start_ns"])]
    windows = chunk_spans.owned_windows(index)
    obs_of: dict[str, tuple[str, list[Observation]]] = {}
    chunk_tracks: list[list[stitch.ChunkTrack]] = []
    for cid in order:
        ts = []
        for n, obs in enumerate(per_chunk[cid]):
            tid = f"{cid}-{n}"
            obs_of[tid] = (cid, obs)
            centres = {
                o.frame.t_ns: (o.det.proposal.cx, o.det.proposal.cy) for o in obs
            }
            ts.append(stitch.ChunkTrack(tid, cid, centres))
        chunk_tracks.append(ts)
    joined = stitch.chains(chunk_tracks, chunk_spans.shared_spans(index), cfg.stitch)

    frames: list[TrackFrame] = []
    summaries: list[TrackSummary] = []
    n_weak = n_unowned = n_linked = 0
    for chain in joined:
        tid = chain[0].track_id
        rows, obs_at = [], {}
        for member in chain:
            cid, obs = obs_of[member.track_id]
            for o in obs:
                rows.append(_frame(tid, cid, o, windows[cid]))
                obs_at[(cid, o.det.proposal.sweep_id)] = o
        if not any(f.owned for f in rows):
            n_unowned += 1
            continue
        line = [obs_at[(f.chunk_id, f.sweep_id)] for f in tracks.per_instant(rows)]
        s = summary.summarize(tid, line, cfg.motion)
        if not summary.keep(s, cfg.filter):
            n_weak += 1
            continue
        n_linked += len(chain) > 1
        summaries.append(s)
        frames += [dataclasses.replace(f, cls=s.cls) for f in rows]
    states = {
        st: sum(s.motion_state == st for s in summaries) for st in tracks.MOTION_STATES
    }
    log.info(
        "%d tracks kept (%s), %d of them linked across chunks; dropped: %d short "
        "weak, %d with no owned frame",
        len(summaries),
        ", ".join(f"{n} {st}" for st, n in states.items()),
        n_linked,
        n_weak,
        n_unowned,
    )
    return frames, summaries


def _write_chunk_manifest(
    bag_id: str,
    chunk_id: str,
    config_path: Optional[str],
    cfg: ComponentConfig,
    calibration: Optional[Calibration],
) -> None:
    """Best-effort: never fails a run that succeeded."""
    root = chunk_root(bag_id, chunk_id)
    try:
        common_manifest.write(
            component="tracking",
            bag_id=bag_id,
            chunk_id=chunk_id,
            inputs={
                "proposals": proposals_path(bag_id, chunk_id),
                "chunks_index": chunks_index_path(bag_id),
                "manifest_proposal_generation": (
                    f"{root}/{common_manifest.component_manifest_name('proposal_generation')}"
                ),
                **(
                    {"masklet_embeddings": masklet_embeddings_path(bag_id, chunk_id)}
                    if cfg.appearance.enabled
                    else {}
                ),
            },
            outputs={
                "tracks": tracks_path(bag_id),
                "track_summaries": track_summaries_path(bag_id),
            },
            config_path=config_path,
            extra=(
                {"appearance_calibration": dataclasses.asdict(calibration)}
                if calibration is not None
                else None
            ),
            filename=common_manifest.component_manifest_name("tracking"),
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
    """Track every chunk of the bag and link the tracks across chunks.

    Skips when the bag's tracks exist unless force=True.
    """
    index = load_chunks(bag_id)
    if not index:
        raise FileNotFoundError(
            f"No chunks found for bag {bag_id!r} — run ingest first."
        )
    camera_calibration.bind_inherited_outputs(
        bag_id,
        None,
        "tracking",
        [
            f"{chunk_root(bag_id, c['chunk_id'])}/manifest_proposal_generation.json"
            for c in index
        ],
        force=force,
        existing_outputs=tracks.exists(bag_id),
    )
    if tracks.exists(bag_id) and not force:
        log.info("bag %s already tracked — skipping (use --force)", bag_id)
        return
    chunk_ids = [r["chunk_id"] for r in index]
    missing = [c for c in chunk_ids if not proposals.exists(bag_id, c)]
    if missing:
        raise FileNotFoundError(
            f"bag {bag_id!r}: no proposals.parquet for chunks {missing} "
            "(run proposal_generation)"
        )
    costs, calibration = bag_appearance(cfg, bag_id, chunk_ids)
    per_chunk = {
        cid: track_chunk(cfg, bag_id, cid, costs.get(cid)) for cid in chunk_ids
    }
    frames, summaries = link_and_summarize(cfg, index, per_chunk)
    tracks.write(bag_id, frames, summaries)
    for cid in chunk_ids:
        _write_chunk_manifest(bag_id, cid, config_path, cfg, calibration)
    log.info(
        "tracking complete for bag %s: %d chunks, %d tracks",
        bag_id,
        len(chunk_ids),
        len(summaries),
    )
