"""proposal_generation output: 3D box proposals, per source and fused.

Written by proposal_generation; read by tracking (the fused rows) and by
evaluation of each source (all rows) once they exist.

On disk, per chunk:

  proposals.parquet
      one ProposalRow per proposal (schemas.PROPOSAL_SCHEMA); sweep_ids,
      supporting_cam_ids, supporting_masklet_ids and member_proposal_ids are
      JSON list strings on disk, tuples once read.

Invariants, checked on write and on read: proposal_ids are unique; provenance
is one of PROVENANCES; sweep_id is among sweep_ids; a fused row lists at least
one member, every member is a non-fused row of the same file and frame (same
sweep_id), and only fused rows list members.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from typing import Optional

from wato_common.artifact_store import local_path, proposals_path
from wato_common.artifacts import ArtifactFormatError
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import (
    PROPOSAL_SCHEMA,
    ProposalRow,
    decode_int_list,
    decode_str_list,
    encode_int_list,
    encode_str_list,
)

FUSED = "fused"
SOURCES = ("lidar_mos", "lifted", "slf", "lidar_detector")
PROVENANCES = SOURCES + (FUSED,)

_LIST_FIELDS = {
    "sweep_ids": (encode_int_list, decode_int_list),
    "supporting_cam_ids": (encode_str_list, decode_str_list),
    "supporting_masklet_ids": (encode_str_list, decode_str_list),
    "member_proposal_ids": (encode_str_list, decode_str_list),
}


@dataclass(frozen=True)
class Proposal:
    """A ProposalRow without bag/chunk ids, list columns decoded.

    Field meanings are documented on schemas.ProposalRow.
    """

    proposal_id: str
    sweep_id: int
    frame_id: Optional[int]
    sweep_ids: tuple[int, ...]
    reference_timestamp_ns: int
    cx: float
    cy: float
    cz: float
    w: float
    l: float  # noqa: E741 — domain term: length, matches parquet w/l/h column triplet
    h: float
    heading: float
    cls: Optional[str]
    score: float
    provenance: str
    lidar_point_count: Optional[int] = None
    supporting_cam_ids: tuple[str, ...] = ()
    supporting_masklet_ids: tuple[str, ...] = ()
    member_proposal_ids: tuple[str, ...] = ()
    motion_score: Optional[float] = None
    n_sources: Optional[int] = None
    frac_persistent: Optional[float] = None
    track_life: Optional[int] = None
    lift_confidence: Optional[float] = None


def _problem(proposals: list[Proposal]) -> Optional[str]:
    """The first broken invariant (module docstring), or None."""
    by_id: dict[str, Proposal] = {}
    for p in proposals:
        if p.proposal_id in by_id:
            return f"proposal_id {p.proposal_id!r} appears twice"
        by_id[p.proposal_id] = p
    for p in proposals:
        where = f"proposal {p.proposal_id!r}"
        if p.provenance not in PROVENANCES:
            return f"{where}: provenance {p.provenance!r} not in {PROVENANCES}"
        if p.sweep_id not in p.sweep_ids:
            return f"{where}: sweep_id {p.sweep_id} not in sweep_ids {p.sweep_ids}"
        if p.provenance != FUSED:
            if p.member_proposal_ids:
                return f"{where}: only fused rows list member_proposal_ids"
            continue
        if not p.member_proposal_ids:
            return f"{where}: fused row lists no member_proposal_ids"
        for m in p.member_proposal_ids:
            member = by_id.get(m)
            if member is None:
                return f"{where}: member {m!r} is not in the file"
            if member.provenance == FUSED:
                return f"{where}: member {m!r} is itself fused"
            if member.sweep_id != p.sweep_id:
                return (
                    f"{where}: member {m!r} is from sweep {member.sweep_id}, "
                    f"not {p.sweep_id}"
                )
    return None


def write(bag_id: str, chunk_id: str, proposals: list[Proposal]) -> None:
    uri = proposals_path(bag_id, chunk_id)
    problem = _problem(proposals)
    if problem is not None:
        raise ValueError(f"{local_path(uri)}: {problem}")
    rows = []
    for p in proposals:
        row = asdict(p)
        for name, (encode, _) in _LIST_FIELDS.items():
            row[name] = encode(list(row[name]))
        rows.append(ProposalRow(bag_id=bag_id, chunk_id=chunk_id, **row).model_dump())
    write_table(rows, PROPOSAL_SCHEMA, uri)


def read(bag_id: str, chunk_id: str) -> list[Proposal]:
    """Every proposal in the chunk, source and fused, in file order."""
    uri = proposals_path(bag_id, chunk_id)
    out = []
    for row in read_rows(uri):
        values = {f.name: row[f.name] for f in fields(Proposal)}
        for name, (_, decode) in _LIST_FIELDS.items():
            values[name] = tuple(decode(values[name]))
        out.append(Proposal(**values))
    problem = _problem(out)
    if problem is not None:
        raise ArtifactFormatError(local_path(uri), problem)
    return out


def fused(proposals: list[Proposal]) -> list[Proposal]:
    """The rows tracking consumes."""
    return [p for p in proposals if p.provenance == FUSED]


def exists(bag_id: str, chunk_id: str) -> bool:
    return os.path.exists(local_path(proposals_path(bag_id, chunk_id)))
