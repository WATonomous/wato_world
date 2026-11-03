"""Ground-truth 3D boxes, imported to score the pipeline.

Written by the importers (src/ingest/scripts/import_nuscenes_gt.py for
nuScenes keyframe annotations; hand labels later); read by evaluation scripts
and wato_common.eval. No pipeline component reads it, so nothing the pipeline
labels can depend on it.

On disk, per bag:

  gt_boxes.parquet
      one GtBoxRow per (instance, instant) (schemas.GT_BOX_SCHEMA).

Invariants, checked on write and on read: source is one of SOURCES;
motion_state is one of MOTION_STATES or None; every extent is positive; an
(source, instance_id, timestamp_ns) appears once.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from typing import Optional

from wato_common.artifact_store import gt_boxes_path, local_path
from wato_common.artifacts import ArtifactFormatError
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import GT_BOX_SCHEMA, GtBoxRow

SOURCES = ("nuscenes_markers", "sustechpoints")
MOVING = "moving"
PARKED = "parked"
MOTION_STATES = (MOVING, PARKED)


@dataclass(frozen=True)
class GtBox:
    """A GtBoxRow without its bag id. Field meanings: schemas.GtBoxRow."""

    source: str
    timestamp_ns: int
    instance_id: str
    cx: float
    cy: float
    cz: float
    w: float
    l: float  # noqa: E741 — domain term: length, matches parquet w/l/h column triplet
    h: float
    heading: float
    cls: Optional[str]
    source_class: str
    motion_state: Optional[str]


def _problem(boxes: list[GtBox]) -> Optional[str]:
    """The first broken invariant (module docstring), or None."""
    seen: set[tuple[str, str, int]] = set()
    for b in boxes:
        where = f"box {b.instance_id!r} at {b.timestamp_ns}"
        if b.source not in SOURCES:
            return f"{where}: source {b.source!r} not in {SOURCES}"
        if b.motion_state is not None and b.motion_state not in MOTION_STATES:
            return f"{where}: motion_state {b.motion_state!r} not in {MOTION_STATES}"
        if min(b.w, b.l, b.h) <= 0.0:
            return f"{where}: non-positive extent ({b.w}, {b.l}, {b.h})"
        key = (b.source, b.instance_id, b.timestamp_ns)
        if key in seen:
            return f"{where}: appears twice for source {b.source!r}"
        seen.add(key)
    return None


def write(bag_id: str, boxes: list[GtBox]) -> None:
    uri = gt_boxes_path(bag_id)
    problem = _problem(boxes)
    if problem is not None:
        raise ValueError(f"{local_path(uri)}: {problem}")
    rows = [GtBoxRow(bag_id=bag_id, **asdict(b)).model_dump() for b in boxes]
    write_table(rows, GT_BOX_SCHEMA, uri)


def read(bag_id: str) -> list[GtBox]:
    """Every ground-truth box of the bag, in file order."""
    uri = gt_boxes_path(bag_id)
    out = [
        GtBox(**{f.name: row[f.name] for f in fields(GtBox)}) for row in read_rows(uri)
    ]
    problem = _problem(out)
    if problem is not None:
        raise ArtifactFormatError(local_path(uri), problem)
    return out


def exists(bag_id: str) -> bool:
    return os.path.exists(local_path(gt_boxes_path(bag_id)))
