"""The spans neighbouring chunks share, and which chunk labels each instant.

Ingest decodes each chunk over its nominal window widened by
chunk_overlap_seconds on both sides, so chunks k and k+1 both cover
[t_overlap_start_ns(k+1), t_overlap_end_ns(k)] (6 s at ingest's defaults).
Both label that span independently. Each instant belongs to one chunk:
chunk k before the span's midpoint, (t_end_ns(k) + t_start_ns(k+1)) / 2, chunk
k+1 from it. An owned instant is then at least half the span from its chunk's
nearest data edge (bag start and end aside), clear of the edge effects
src/tracking/README.md lists. tracking marks its frames by it (`owned`) and
links tracks across the span; evaluation scores each instant once, in its
owner.

Both functions take chunks/index.parquet rows (schemas.ChunkRow as dicts),
in any order, and read the bounds from them, not from constants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SharedSpan:
    earlier: str  # chunk_id of chunk k
    later: str  # chunk_id of chunk k+1
    start_ns: int  # t_overlap_start_ns(k+1)
    end_ns: int  # t_overlap_end_ns(k)

    def __contains__(self, t_ns: int) -> bool:
        return self.start_ns <= t_ns <= self.end_ns


def _in_order(chunks: list[dict]) -> list[dict]:
    return sorted(chunks, key=lambda r: r["t_start_ns"])


def owned_windows(chunks: list[dict]) -> dict[str, tuple[float, float]]:
    """chunk_id -> [lo, hi) of the instants it owns."""
    rows = _in_order(chunks)
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


def shared_spans(chunks: list[dict]) -> list[SharedSpan]:
    """Each pair of neighbouring chunks' shared span, in time order."""
    rows = _in_order(chunks)
    return [
        SharedSpan(
            a["chunk_id"], b["chunk_id"], b["t_overlap_start_ns"], a["t_overlap_end_ns"]
        )
        for a, b in zip(rows, rows[1:])
    ]
