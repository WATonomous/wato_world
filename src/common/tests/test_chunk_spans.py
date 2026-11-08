"""wato_common.chunk_spans: shared spans and the instants each chunk owns."""

from __future__ import annotations

import math

from wato_common.chunk_spans import SharedSpan, owned_windows, shared_spans

# Ingest's defaults (30 s windows, 2 s overlap), listed out of order.
CHUNKS = [
    dict(
        chunk_id="0001",
        t_start_ns=28,
        t_end_ns=58,
        t_overlap_start_ns=26,
        t_overlap_end_ns=60,
    ),
    dict(
        chunk_id="0000",
        t_start_ns=0,
        t_end_ns=30,
        t_overlap_start_ns=0,
        t_overlap_end_ns=32,
    ),
]


def test_each_shared_span_is_split_at_its_midpoint():
    w = owned_windows(CHUNKS)
    assert w == {"0000": (-math.inf, 29.0), "0001": (29.0, math.inf)}


def test_neighbours_share_three_overlaps():
    (span,) = shared_spans(CHUNKS)
    assert span == SharedSpan("0000", "0001", 26, 32)
    assert 26 in span and 32 in span and 33 not in span


def test_a_single_chunk_owns_everything_and_shares_nothing():
    w = owned_windows(CHUNKS[1:])
    assert w == {"0000": (-math.inf, math.inf)}
    assert shared_spans(CHUNKS[1:]) == []
