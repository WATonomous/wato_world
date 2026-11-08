"""stitch: linking chunk k's tracks to chunk k+1's across their shared span."""

from __future__ import annotations

import pytest

from wato_common.chunk_spans import SharedSpan
from wato_tracking.config import StitchConfig
from wato_tracking.stitch import ChunkTrack, agreement, chains, links

SPAN = SharedSpan("0000", "0001", 10, 20)


def _track(tid, xy_at):
    return ChunkTrack(tid, tid.split("-")[0], dict(xy_at))


def _line(tid, ts, x0, y=0.0, v=0.0):
    return _track(tid, {t: (x0 + v * t, y) for t in ts})


def test_agreement_is_over_the_shorter_tracks_span_instants():
    a = _line("0000-0", range(0, 21), 0.0)  # 11 instants in the span
    b = _line("0001-0", range(14, 30), 0.0)  # 7 instants in the span
    assert agreement(a, b, SPAN, 1.0) == (1.0, 7)
    # Apart by 2 m at every shared instant: no agreement.
    assert agreement(a, _line("0001-1", range(14, 30), 2.0), SPAN, 1.0) == (0.0, 0)
    # No instant in the span: nothing to compare.
    assert agreement(a, _line("0001-2", range(25, 30), 0.0), SPAN, 1.0) == (0.0, 0)


def test_links_are_one_to_one():
    a = [_line("0000-0", range(0, 21), 0.0), _line("0000-1", range(0, 21), 0.6)]
    # b0 sits between a0 and a1 (within 1 m of both); b1 only near a1.
    b = [_line("0001-0", range(10, 30), 0.3), _line("0001-1", range(10, 30), 1.4)]
    assert sorted(links(a, b, SPAN, StitchConfig())) == [(0, 0), (1, 1)]


def test_the_long_track_wins_over_a_fragment_that_also_agrees():
    a = [_line("0000-0", range(0, 21), 0.0)]
    # A 4-frame fragment at the span's start, and the object's own track.
    b = [_line("0001-0", range(10, 14), 0.0), _line("0001-1", range(11, 30), 0.0)]
    assert links(a, b, SPAN, StitchConfig()) == [(0, 1)]


def test_too_few_agreeing_instants_make_no_link():
    a = [_line("0000-0", range(0, 21), 0.0)]
    b = [_line("0001-0", range(19, 30), 0.0)]  # 2 shared instants
    assert links(a, b, SPAN, StitchConfig()) == []
    assert links(a, b, SPAN, StitchConfig(min_agreeing_frames=2)) == [(0, 0)]


def test_a_mover_is_followed_through_three_chunks():
    spans = [SPAN, SharedSpan("0001", "0002", 30, 40)]
    per_chunk = [
        [_line("0000-0", range(0, 21), 0.0, v=1.0), _line("0000-1", range(0, 5), 50.0)],
        [_line("0001-0", range(10, 41), 0.0, v=1.0)],
        [_line("0002-0", range(30, 60), 0.0, v=1.0)],
    ]
    out = chains(per_chunk, spans, StitchConfig())
    assert sorted([t.track_id for t in c] for c in out) == [
        ["0000-0", "0001-0", "0002-0"],
        ["0000-1"],
    ]


def test_chains_need_one_span_per_seam():
    with pytest.raises(ValueError, match="shared spans"):
        chains([[], []], [], StitchConfig())
