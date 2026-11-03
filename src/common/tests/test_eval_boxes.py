"""wato_common.eval.boxes: matching predictions to ground truth at one instant."""

from __future__ import annotations

from wato_common.eval.boxes import Scored, Tally, band_of, match_frame, tally_frame
from wato_common.schemas import Box3D


def _box(cx, cy, l=4.0, w=1.8, h=1.5) -> Box3D:  # noqa: E741 — box length
    return Box3D(cx=cx, cy=cy, cz=0.75, w=w, l=l, h=h, heading=0.0)


def test_higher_score_claims_the_ground_truth_first():
    gts = [Scored(_box(0, 0), cls="car")]
    preds = [Scored(_box(0.3, 0), score=0.2), Scored(_box(0.9, 0), score=0.9)]
    m = match_frame(preds, gts, max_dist_m=1.0)
    assert m.pairs == ((1, 0),)
    assert m.false_positives == (0,)
    assert m.misses == ()


def test_out_of_taxonomy_ground_truth_is_ignored_not_scored():
    gts = [Scored(_box(0, 0), cls=None), Scored(_box(10, 0), cls="car")]
    preds = [Scored(_box(0.2, 0))]
    m = match_frame(preds, gts, max_dist_m=1.0)
    assert m.pairs == ()
    assert m.ignored_preds == (0,)
    assert m.false_positives == ()
    assert m.misses == (1,)


def test_tally_counts_bands_and_errors():
    tallies: dict[str, Tally] = {}
    gts = [
        Scored(_box(0, 0), cls="car", range_m=10.0),
        Scored(_box(40, 0), cls="pedestrian", range_m=40.0),
    ]
    preds = [
        Scored(_box(0.5, 0, l=2.0), cls="car", range_m=10.0),
        Scored(_box(80, 0), range_m=80.0),
    ]
    tally_frame(tallies, preds, gts, max_dist_m=1.0)
    s = tallies["all"].summary()
    assert (s["n_gt"], s["n_pred"], s["hits"], s["false_positives"]) == (2, 2, 1, 1)
    assert s["recall"] == 0.5 and s["precision"] == 0.5
    assert s["median_dl_m"] == -2.0 and s["class_accuracy"] == 1.0
    assert tallies["0-15"].summary()["recall"] == 1.0
    assert tallies["30-50"].summary()["recall"] == 0.0
    assert tallies["50+"].summary()["false_positives"] == 1


def test_band_names():
    assert band_of(0.0) == "0-15" and band_of(29.9) == "15-30" and band_of(99) == "50+"
