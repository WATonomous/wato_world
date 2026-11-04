"""label_refinement output: each track's refined box in every frame.

Written by label_refinement; read by student_training and evaluation.

On disk, per bag:

  refined_labels.parquet
      one RefinedTrackRow per tracks.parquet row (schemas.REFINED_TRACK_SCHEMA).

Invariants, checked on write and on read: a (track_id, chunk_id, sweep_id)
appears once, and a track has at most one owned row per instant; every row of
a track has the same size (w, l, h), all positive; motion_state is one of
tracks.MOTION_STATES and size_source one of SIZE_SOURCES.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from typing import Optional

from wato_common.artifact_store import local_path, refined_labels_path
from wato_common.artifacts import ArtifactFormatError
from wato_common.artifacts.tracks import MOTION_STATES
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import REFINED_TRACK_SCHEMA, RefinedTrackRow

AGGREGATE = "aggregate"
PERCENTILE = "percentile"
SLF = "slf"
SIZE_SOURCES = (AGGREGATE, PERCENTILE, SLF)


@dataclass(frozen=True)
class RefinedBox:
    """A RefinedTrackRow without its bag id. Field meanings:
    schemas.RefinedTrackRow."""

    track_id: str
    chunk_id: str
    sweep_id: int
    reference_timestamp_ns: int
    cx: float
    cy: float
    cz: float
    w: float
    l: float  # noqa: E741 — domain term: length, matches parquet w/l/h column triplet
    h: float
    heading: float
    heading_directed: bool
    cls: Optional[str]
    motion_state: str
    size_source: str
    confidence: float
    owned: bool
    residual_silhouette: Optional[float] = None
    residual_lidar_fit: Optional[float] = None
    residual_smoothness: Optional[float] = None


def _problem(boxes: list[RefinedBox]) -> Optional[str]:
    """The first broken invariant (module docstring), or None."""
    seen: set[tuple[str, str, int]] = set()
    owned: set[tuple[str, int]] = set()
    size: dict[str, tuple[float, float, float]] = {}
    for b in boxes:
        where = f"track {b.track_id!r} at chunk {b.chunk_id} sweep {b.sweep_id}"
        if (b.track_id, b.chunk_id, b.sweep_id) in seen:
            return f"{where} appears twice"
        seen.add((b.track_id, b.chunk_id, b.sweep_id))
        if b.owned:
            if (b.track_id, b.reference_timestamp_ns) in owned:
                return f"{where}: a second owned row at one instant"
            owned.add((b.track_id, b.reference_timestamp_ns))
        if b.motion_state not in MOTION_STATES:
            return f"{where}: motion_state {b.motion_state!r} not in {MOTION_STATES}"
        if b.size_source not in SIZE_SOURCES:
            return f"{where}: size_source {b.size_source!r} not in {SIZE_SOURCES}"
        if min(b.w, b.l, b.h) <= 0.0:
            return f"{where}: non-positive extent ({b.w}, {b.l}, {b.h})"
        if size.setdefault(b.track_id, (b.w, b.l, b.h)) != (b.w, b.l, b.h):
            return f"{where}: its size differs from the track's other rows"
    return None


def write(bag_id: str, boxes: list[RefinedBox]) -> None:
    uri = refined_labels_path(bag_id)
    problem = _problem(boxes)
    if problem is not None:
        raise ValueError(f"{local_path(uri)}: {problem}")
    rows = [RefinedTrackRow(bag_id=bag_id, **asdict(b)).model_dump() for b in boxes]
    write_table(rows, REFINED_TRACK_SCHEMA, uri)


def read(bag_id: str) -> list[RefinedBox]:
    """Every refined box of the bag, in file order."""
    uri = refined_labels_path(bag_id)
    out = [
        RefinedBox(**{f.name: row[f.name] for f in fields(RefinedBox)})
        for row in read_rows(uri)
    ]
    problem = _problem(out)
    if problem is not None:
        raise ArtifactFormatError(local_path(uri), problem)
    return out


def exists(bag_id: str) -> bool:
    return os.path.exists(local_path(refined_labels_path(bag_id)))
