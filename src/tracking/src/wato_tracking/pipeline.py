"""tracking pipeline orchestrator.

Per chunk:
  1. The chunk's fused proposals as per-frame detections (detections.py).
  2. Tracks over them (tracker.py).
  3. Per track: class, parked / moving / unknown, kept or dropped (summary.py).
  4. Each kept track's frames, with `owned` set on the frames in the chunk's
     own half of each 6 s overlap.
Then every chunk's tracks go to tracks.parquet + track_summaries.parquet
(wato_common.artifacts.tracks), bag-level. Track ids are chunk-local
("<chunk_id>-<n>") until tracks are stitched across chunks.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from wato_common import manifest as common_manifest
from wato_common.artifact_store import (
    chunk_root,
    chunks_index_path,
    proposals_path,
    track_summaries_path,
    tracks_path,
)
from wato_common.artifacts import proposals, tracks
from wato_common.artifacts.tracks import TrackFrame, TrackSummary
from wato_tracking import detections, summary, tracker
from wato_tracking.config import ComponentConfig
from wato_tracking.io import load_chunks

log = logging.getLogger(__name__)


def ownership(chunks: list[dict]) -> dict[str, tuple[float, float]]:
    """chunk_id -> [lo, hi) of the instants it labels: each shared span is
    split at its midpoint, (t_end(k) + t_start(k+1)) / 2."""
    rows = sorted(chunks, key=lambda r: r["t_start_ns"])
    out = {}
    for i, r in enumerate(rows):
        lo = (rows[i - 1]["t_end_ns"] + r["t_start_ns"]) / 2 if i > 0 else -math.inf
        hi = (
            (r["t_end_ns"] + rows[i + 1]["t_start_ns"]) / 2
            if i + 1 < len(rows)
            else math.inf
        )
        out[r["chunk_id"]] = (lo, hi)
    return out


def process_chunk(
    cfg: ComponentConfig, bag_id: str, chunk_id: str, owned: tuple[float, float]
) -> tuple[list[TrackFrame], list[TrackSummary]]:
    rows = proposals.read(bag_id, chunk_id)
    det_frames = detections.frames(rows, cfg.strong)
    frames_out: list[TrackFrame] = []
    summaries: list[TrackSummary] = []
    n_dropped = 0
    for obs in tracker.run(det_frames, cfg.association):
        s = summary.summarize(f"{chunk_id}-{len(summaries)}", chunk_id, obs, cfg.motion)
        if not summary.keep(s, cfg.filter):
            n_dropped += 1
            continue
        summaries.append(s)
        for o in obs:
            p = o.det.proposal
            frames_out.append(
                TrackFrame(
                    track_id=s.track_id,
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
                    cls=s.cls,
                    score=p.score,
                    owned=owned[0] <= p.reference_timestamp_ns < owned[1],
                    supporting_cam_ids=p.supporting_cam_ids,
                    supporting_masklet_ids=p.supporting_masklet_ids,
                )
            )
    states = {
        st: sum(s.motion_state == st for s in summaries) for st in tracks.MOTION_STATES
    }
    log.info(
        "chunk %s: %d frames, %d tracks kept (%s), %d short weak tracks dropped",
        chunk_id,
        len(det_frames),
        len(summaries),
        ", ".join(f"{n} {st}" for st, n in states.items()),
        n_dropped,
    )
    return frames_out, summaries


def _write_chunk_manifest(
    bag_id: str, chunk_id: str, config_path: Optional[str]
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
            },
            outputs={
                "tracks": tracks_path(bag_id),
                "track_summaries": track_summaries_path(bag_id),
            },
            config_path=config_path,
            filename=common_manifest.component_manifest_name("tracking"),
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
    """Track every chunk of the bag (or one chunk, replacing only its tracks).

    Skips when the bag's tracks exist unless force=True.
    """
    chunks = load_chunks(bag_id)
    if not chunks:
        raise FileNotFoundError(
            f"No chunks found for bag {bag_id!r} — run ingest first."
        )
    windows = ownership(chunks)
    todo = [c["chunk_id"] for c in chunks]
    if chunk_id:
        if chunk_id not in windows:
            raise ValueError(f"chunk_id {chunk_id!r} not found for bag {bag_id!r}")
        todo = [chunk_id]
    if tracks.exists(bag_id) and not force:
        log.info("bag %s already tracked — skipping (use --force)", bag_id)
        return
    missing = [c for c in todo if not proposals.exists(bag_id, c)]
    if missing:
        raise FileNotFoundError(
            f"bag {bag_id!r}: no proposals.parquet for chunks {missing} "
            "(run proposal_generation)"
        )

    frames: list[TrackFrame] = []
    summaries: list[TrackSummary] = []
    if chunk_id and tracks.exists(bag_id):
        old_frames, old_summaries = tracks.read(bag_id)
        frames = [f for f in old_frames if f.chunk_id != chunk_id]
        summaries = [s for s in old_summaries if s.chunk_id != chunk_id]
    for cid in todo:
        f, s = process_chunk(cfg, bag_id, cid, windows[cid])
        frames += f
        summaries += s
    tracks.write(bag_id, frames, summaries)
    for cid in todo:
        _write_chunk_manifest(bag_id, cid, config_path)
    log.info(
        "tracking complete for bag %s: %d chunks, %d tracks",
        bag_id,
        len(todo),
        len(summaries),
    )
