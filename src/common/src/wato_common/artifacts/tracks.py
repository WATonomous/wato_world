"""tracking output: tracks over the fused proposals, and what each track is.

Written by tracking; read by label_refinement and evaluation.

On disk, per bag:

  tracks.parquet
      one TrackRow per (track, frame) (schemas.TRACK_SCHEMA);
      supporting_cam_ids and supporting_masklet_ids are JSON list strings on
      disk, tuples once read.
  track_summaries.parquet
      one TrackSummaryRow per track (schemas.TRACK_SUMMARY_SCHEMA).

A track can span chunks: its rows come from every chunk it was linked across,
so an instant of a shared span can have one row per chunk (schemas.TrackRow).
`per_instant` gives one frame per instant, the owned one where there is one.

Invariants, checked on write and on read: a track has one summary and at
least one frame; its frames have distinct (chunk_id, sweep_id), at most one
owned frame per reference_timestamp_ns, and the summary's n_frames (distinct
instants) and first / last timestamps match them; motion_state is one of
MOTION_STATES.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from typing import Optional

from wato_common.artifact_store import local_path, track_summaries_path, tracks_path
from wato_common.artifacts import ArtifactFormatError
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import (
    TRACK_SCHEMA,
    TRACK_SUMMARY_SCHEMA,
    TrackRow,
    TrackSummaryRow,
    decode_str_list,
    encode_str_list,
)

MOVING = "moving"
PARKED = "parked"
UNKNOWN = "unknown"
MOTION_STATES = (MOVING, PARKED, UNKNOWN)

_LIST_FIELDS = ("supporting_cam_ids", "supporting_masklet_ids")


@dataclass(frozen=True)
class TrackFrame:
    """A TrackRow without its bag id, list columns decoded. Field meanings:
    schemas.TrackRow."""

    track_id: str
    chunk_id: str
    sweep_id: int
    reference_timestamp_ns: int
    proposal_id: str
    cx: float
    cy: float
    cz: float
    w: float
    l: float  # noqa: E741 — domain term: length, matches parquet w/l/h column triplet
    h: float
    heading: float
    cls: Optional[str]
    score: float
    owned: bool
    supporting_cam_ids: tuple[str, ...] = ()
    supporting_masklet_ids: tuple[str, ...] = ()
    dino_feature_path: Optional[str] = None


@dataclass(frozen=True)
class TrackSummary:
    """A TrackSummaryRow without its bag id. Field meanings:
    schemas.TrackSummaryRow."""

    track_id: str
    n_frames: int
    first_timestamp_ns: int
    last_timestamp_ns: int
    motion_state: str
    cls: Optional[str]
    n_strong: int
    frac_two_source: float
    frac_lidar_mos: float
    net_displacement_m: float
    centre_var_m2: float
    median_points: float


def _problem(frames: list[TrackFrame], summaries: list[TrackSummary]) -> Optional[str]:
    """The first broken invariant (module docstring), or None."""
    by_id: dict[str, TrackSummary] = {}
    for s in summaries:
        if s.track_id in by_id:
            return f"track {s.track_id!r} has two summaries"
        if s.motion_state not in MOTION_STATES:
            return f"track {s.track_id!r}: motion_state {s.motion_state!r} not in {MOTION_STATES}"
        by_id[s.track_id] = s
    per_track: dict[str, list[TrackFrame]] = defaultdict(list)
    for f in frames:
        if f.track_id not in by_id:
            return f"frame of track {f.track_id!r} at sweep {f.sweep_id} has no summary"
        per_track[f.track_id].append(f)
    for tid, s in by_id.items():
        fs = per_track.get(tid, [])
        where = f"track {tid!r}"
        if not fs:
            return f"{where} has no frames"
        if len({(f.chunk_id, f.sweep_id) for f in fs}) != len(fs):
            return f"{where} has two frames at one (chunk_id, sweep_id)"
        owned = [f.reference_timestamp_ns for f in fs if f.owned]
        if len(set(owned)) != len(owned):
            return f"{where} has two owned frames at one instant"
        t = [f.reference_timestamp_ns for f in fs]
        if s.n_frames != len(set(t)):
            return f"{where}: n_frames {s.n_frames} but {len(set(t))} instants"
        if (s.first_timestamp_ns, s.last_timestamp_ns) != (min(t), max(t)):
            return f"{where}: first/last timestamps don't match its frames"
    return None


def write(bag_id: str, frames: list[TrackFrame], summaries: list[TrackSummary]) -> None:
    problem = _problem(frames, summaries)
    if problem is not None:
        raise ValueError(f"{local_path(tracks_path(bag_id))}: {problem}")
    rows = []
    for f in frames:
        row = asdict(f)
        for name in _LIST_FIELDS:
            row[name] = encode_str_list(list(row[name]))
        rows.append(TrackRow(bag_id=bag_id, **row).model_dump())
    write_table(rows, TRACK_SCHEMA, tracks_path(bag_id))
    write_table(
        [TrackSummaryRow(bag_id=bag_id, **asdict(s)).model_dump() for s in summaries],
        TRACK_SUMMARY_SCHEMA,
        track_summaries_path(bag_id),
    )


def read(bag_id: str) -> tuple[list[TrackFrame], list[TrackSummary]]:
    """Every track frame (file order) and every track summary of the bag."""
    frames = []
    for row in read_rows(tracks_path(bag_id)):
        values = {f.name: row[f.name] for f in fields(TrackFrame)}
        for name in _LIST_FIELDS:
            values[name] = tuple(decode_str_list(values[name]))
        frames.append(TrackFrame(**values))
    summaries = [
        TrackSummary(**{f.name: row[f.name] for f in fields(TrackSummary)})
        for row in read_rows(track_summaries_path(bag_id))
    ]
    problem = _problem(frames, summaries)
    if problem is not None:
        raise ArtifactFormatError(local_path(tracks_path(bag_id)), problem)
    return frames, summaries


def per_instant(frames: list[TrackFrame]) -> list[TrackFrame]:
    """One track's frames, one per instant in time order: the owned frame
    where the track has one, else the one chunk's that saw it then."""
    best: dict[int, TrackFrame] = {}
    for f in frames:
        t = f.reference_timestamp_ns
        if t not in best or (f.owned and not best[t].owned):
            best[t] = f
    return [best[t] for t in sorted(best)]


def exists(bag_id: str) -> bool:
    return os.path.exists(local_path(tracks_path(bag_id))) and os.path.exists(
        local_path(track_summaries_path(bag_id))
    )
