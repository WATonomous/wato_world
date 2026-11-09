"""wato_common.eval.mot: TrackEval's HOTA association terms, CLEAR and IDF1."""

from __future__ import annotations

import numpy as np
import pytest

from wato_common.eval.mot import MotFrame, clear, distance_similarity, hota, identity


def _frames(gt_xy, tracks):
    """One object "g" at gt_xy[k] per frame k; tracks: per frame, a list of
    (tracker id, x) at y = 0, matched within 2 m."""
    out = []
    for k, x in enumerate(gt_xy):
        ids = tuple(i for i, _ in tracks[k])
        xy = np.array([[tx, 0.0] for _, tx in tracks[k]]).reshape(-1, 2)
        out.append(MotFrame(("g",), ids, distance_similarity([[x, 0.0]], xy, 2.0)))
    return out


def test_distance_similarity_is_half_at_the_match_distance():
    s = distance_similarity([[0.0, 0.0]], [[0.0, 0.0], [2.0, 0.0], [5.0, 0.0]], 2.0)
    assert s.tolist() == [[1.0, 0.5, 0.0]]


def test_one_object_one_track_is_perfect():
    f = _frames([0, 1, 2, 3], [[("a", 0)], [("a", 1)], [("a", 2)], [("a", 3)]])
    h, c, i = hota(f), clear(f), identity(f)
    assert (h["AssA"], h["AssRe"], h["AssPr"], h["DetA"]) == (1.0, 1.0, 1.0, 1.0)
    assert (c["IDSW"], c["Frag"], c["MOTA"]) == (0, 0, 1.0)
    assert i["IDF1"] == 1.0


def test_one_object_split_over_two_tracks_halves_association_recall():
    f = _frames([0, 1, 2, 3], [[("a", 0)], [("a", 1)], [("b", 2)], [("b", 3)]])
    h, c, i = hota(f), clear(f), identity(f)
    assert h["AssRe"] == pytest.approx(0.5)
    assert h["AssPr"] == pytest.approx(1.0)
    assert h["AssA"] == pytest.approx(0.5)
    # A continuous hand-over is one switch and no fragmentation.
    assert (c["IDSW"], c["Frag"]) == (1, 0)
    # The best mapping covers 2 of 4 ground-truth and 2 of 4 tracker boxes.
    assert (i["IDTP"], i["IDFN"], i["IDFP"]) == (2, 2, 2)
    assert i["IDF1"] == pytest.approx(0.5)


def test_an_untracked_frame_is_a_fragmentation_not_a_switch():
    # Frame 2: the track's box is 5 m off, so the object is untracked there.
    f = _frames(
        [0, 1, 2, 3, 4],
        [[("a", 0)], [("a", 1)], [("a", 7)], [("a", 3)], [("a", 4)]],
    )
    c = clear(f)
    assert (c["IDSW"], c["Frag"], c["FN"], c["FP"]) == (0, 1, 1, 1)


def test_one_track_over_two_objects_lowers_association_precision():
    sim = distance_similarity
    f = [
        MotFrame(("g1",), ("a",), sim([[0, 0]], [[0, 0]], 2.0)),
        MotFrame(("g1",), ("a",), sim([[1, 0]], [[1, 0]], 2.0)),
        MotFrame(("g2",), ("a",), sim([[2, 0]], [[2, 0]], 2.0)),
        MotFrame(("g2",), ("a",), sim([[3, 0]], [[3, 0]], 2.0)),
    ]
    h = hota(f)
    assert h["AssRe"] == pytest.approx(1.0)
    assert h["AssPr"] == pytest.approx(0.5)


def test_clear_keeps_the_previous_pair_over_a_closer_box():
    # Frame 1: track b is nearer the object, but a matched it last frame and
    # is still within the threshold, so there is no switch.
    f = _frames([0, 1], [[("a", 0)], [("a", 1.5), ("b", 1.0)]])
    c = clear(f)
    assert (c["IDSW"], c["TP"], c["FP"]) == (0, 2, 1)


def test_empty_frames_count_misses_and_false_positives():
    f = [
        MotFrame(("g",), (), np.zeros((1, 0))),
        MotFrame((), ("a",), np.zeros((0, 1))),
    ]
    h, c = hota(f), clear(f)
    assert (h["TP"], h["FN"], h["FP"]) == (0, 1, 1)
    assert (c["TP"], c["FN"], c["FP"]) == (0, 1, 1)
