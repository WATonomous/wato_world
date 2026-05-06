"""Tests for the chunk-range computation (no rosbag involved)."""

from __future__ import annotations

from wato_ingest.inputs.chunks import _compute_chunks


def test_single_chunk_when_duration_is_short():
    chunks = _compute_chunks(
        bag_id="b",
        starting_time_ns=1_000_000_000,
        duration_ns=10 * 1_000_000_000,  # 10 s
        chunk_seconds=30.0,
        overlap_seconds=2.0,
    )
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "0000"
    assert chunks[0].t_start_ns == 1_000_000_000
    # Chunk capped at bag end.
    assert chunks[0].t_end_ns == 1_000_000_000 + 10 * 1_000_000_000


def test_chunks_step_is_chunk_minus_overlap():
    chunks = _compute_chunks(
        bag_id="b",
        starting_time_ns=0,
        duration_ns=120 * 1_000_000_000,  # 120 s
        chunk_seconds=30.0,
        overlap_seconds=2.0,
    )
    # 30 - 2 = 28s step → chunks start at 0, 28, 56, 84, 112; last one would
    # hit 140 but the bag ends at 120 so we stop there.
    starts = [c.t_start_ns / 1e9 for c in chunks]
    assert starts == [0.0, 28.0, 56.0, 84.0, 112.0]
    assert chunks[-1].t_end_ns == 120 * 1_000_000_000


def test_overlap_window_is_clamped_to_bag_bounds():
    chunks = _compute_chunks(
        bag_id="b",
        starting_time_ns=0,
        duration_ns=60 * 1_000_000_000,
        chunk_seconds=30.0,
        overlap_seconds=2.0,
    )
    # First chunk: overlap_start cannot go below bag start.
    assert chunks[0].t_overlap_start_ns == 0
    # Last chunk: overlap_end cannot go past bag end.
    assert chunks[-1].t_overlap_end_ns == 60 * 1_000_000_000


def test_zero_duration_yields_no_chunks():
    assert (
        _compute_chunks(
            bag_id="b",
            starting_time_ns=0,
            duration_ns=0,
            chunk_seconds=30.0,
            overlap_seconds=2.0,
        )
        == []
    )
