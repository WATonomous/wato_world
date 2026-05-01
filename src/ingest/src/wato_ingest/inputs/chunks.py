"""Compute virtual chunk boundaries for a bag.

We do not physically split the rosbag — chunks are just (t_start, t_end) ranges
that downstream tooling can read independently from the original bag.  Each
chunk overlaps the next by `chunk_overlap_seconds` so tracks crossing the seam
can be stitched in tracking.
"""

from __future__ import annotations

from wato_common.artifact_store import chunks_index_path
from wato_common.io.parquet_io import write_table
from wato_common.io.rosbag_reader import summarize
from wato_common.schemas import CHUNK_SCHEMA, ChunkRow
from wato_ingest.config import IngestConfig


def chunk_id_for(index: int) -> str:
    return f"{index:04d}"


def compute_chunks(
    bag_path: str,
    bag_id: str,
    cfg: IngestConfig,
) -> list[ChunkRow]:
    summary = summarize(bag_path, storage_id=cfg.storage_id)
    return _compute_chunks(
        bag_id=bag_id,
        starting_time_ns=summary.starting_time_ns,
        duration_ns=summary.duration_ns,
        chunk_seconds=cfg.chunk_seconds,
        overlap_seconds=cfg.chunk_overlap_seconds,
    )


def _compute_chunks(
    *,
    bag_id: str,
    starting_time_ns: int,
    duration_ns: int,
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[ChunkRow]:
    """Pure logic, exposed for unit testing without a real bag."""
    if duration_ns <= 0:
        return []
    chunk_ns = int(chunk_seconds * 1e9)
    overlap_ns = int(overlap_seconds * 1e9)
    step_ns = max(chunk_ns - overlap_ns, 1)

    end_ns = starting_time_ns + duration_ns
    chunks: list[ChunkRow] = []
    idx = 0
    t = starting_time_ns
    while t < end_ns:
        t_start = t
        t_end = min(t + chunk_ns, end_ns)
        # Overlap with previous: 2 s before t_start (clamped to bag start).
        ov_start = max(starting_time_ns, t_start - overlap_ns)
        # Overlap with next: 2 s after t_end (clamped to bag end).
        ov_end = min(end_ns, t_end + overlap_ns)
        chunks.append(ChunkRow(
            bag_id=bag_id,
            chunk_id=chunk_id_for(idx),
            t_start_ns=t_start,
            t_end_ns=t_end,
            t_overlap_start_ns=ov_start,
            t_overlap_end_ns=ov_end,
        ))
        if t_end >= end_ns:
            break
        idx += 1
        t += step_ns
    return chunks


def write_chunk_index(bag_id: str, chunks: list[ChunkRow]) -> str:
    uri = chunks_index_path(bag_id)
    write_table([c.model_dump() for c in chunks], CHUNK_SCHEMA, uri)
    return uri
