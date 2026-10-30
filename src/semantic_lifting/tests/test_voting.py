"""Tests for voting.py — cross-camera vote resolution."""

from __future__ import annotations

import numpy as np
import pytest

from wato_semantic_lifting.voting import Votes, resolve

# Labels (masklets) used below: index -> class index, instance index.
CAR, PED = 0, 1


def _votes(*rows: tuple[int, int, int, float]) -> Votes:
    """rows of (point, camera, label, score)."""
    a = np.array(rows, dtype=float).reshape(-1, 4)
    return Votes(
        point_idx=a[:, 0].astype(np.int64),
        cam_idx=a[:, 1].astype(np.int64),
        label_idx=a[:, 2].astype(np.int64),
        score=a[:, 3],
    )


def _resolve(votes: Votes, label_cls, label_instance=None):
    label_cls = np.asarray(label_cls, np.int64)
    if label_instance is None:
        label_instance = np.arange(len(label_cls))
    return resolve(votes, label_cls, np.asarray(label_instance, np.int64))


def test_single_vote_wins_with_confidence_one():
    r = _resolve(_votes((0, 0, 0, 0.9)), [CAR])
    assert r.point_idx.tolist() == [0]
    assert r.instance.tolist() == [0] and r.cls.tolist() == [CAR]
    assert r.confidence.tolist() == [1.0]
    assert r.n_supporting_cameras.tolist() == [1]
    assert r.n_disagreeing_cameras.tolist() == [0]


def test_two_cameras_on_the_same_class_agree():
    """Each camera's masklet is its own instance until they're merged, but two
    cameras calling a point "car" agree: full confidence, both support it."""
    r = _resolve(_votes((0, 0, 0, 0.6), (0, 1, 1, 0.8)), [CAR, CAR])
    assert r.cls.tolist() == [CAR]
    assert r.instance.tolist() == [1]  # the higher-scoring masklet
    assert r.confidence.tolist() == [1.0]
    assert r.n_supporting_cameras.tolist() == [2]
    assert r.n_disagreeing_cameras.tolist() == [0]


def test_merged_masklets_vote_as_one_instance():
    r = _resolve(_votes((0, 0, 0, 0.6), (0, 1, 1, 0.8)), [CAR, CAR], [5, 5])
    assert r.instance.tolist() == [5]
    assert r.n_supporting_cameras.tolist() == [2]


def test_cameras_on_different_classes_disagree():
    r = _resolve(_votes((0, 0, 0, 0.9), (0, 1, 1, 0.3)), [CAR, PED])
    assert r.cls.tolist() == [CAR]
    assert r.confidence[0] == pytest.approx(0.9 / 1.2)
    assert r.n_supporting_cameras.tolist() == [1]
    assert r.n_disagreeing_cameras.tolist() == [1]


def test_class_score_is_summed_across_cameras():
    """Two weak "car" votes outweigh one stronger "pedestrian" vote; the
    disagreeing count is in cameras, not in competing labels."""
    votes = _votes((0, 0, 0, 0.35), (0, 1, 1, 0.35), (0, 2, 2, 0.5))
    r = _resolve(votes, [CAR, CAR, PED])
    assert r.cls.tolist() == [CAR]
    assert r.confidence[0] == pytest.approx(0.7 / 1.2)
    assert r.n_supporting_cameras.tolist() == [2]
    assert r.n_disagreeing_cameras.tolist() == [1]


def test_runner_up_is_the_strongest_other_class():
    votes = _votes((0, 0, 0, 0.8), (0, 1, 1, 0.3), (0, 2, 2, 0.1))
    r = _resolve(votes, [CAR, PED, 2])
    assert r.confidence[0] == pytest.approx(0.8 / 1.1)
    assert r.n_disagreeing_cameras.tolist() == [2]


def test_points_resolve_independently_and_in_point_order():
    votes = _votes((5, 0, 1, 0.6), (2, 0, 0, 0.9), (2, 1, 1, 0.2))
    r = _resolve(votes, [CAR, PED])
    assert r.point_idx.tolist() == [2, 5]
    assert r.cls.tolist() == [CAR, PED]
    assert r.confidence[1] == pytest.approx(1.0)


def test_ties_break_deterministically_to_the_lower_index():
    r = _resolve(_votes((0, 0, 1, 0.5), (0, 1, 0, 0.5)), [PED, CAR])
    assert r.cls.tolist() == [CAR]  # class index 0 beats 1 at equal score
    r = _resolve(_votes((0, 0, 1, 0.5), (0, 1, 0, 0.5)), [CAR, CAR])
    assert r.instance.tolist() == [0]


def test_empty_votes_resolve_to_nothing():
    r = _resolve(_votes(), [CAR])
    assert r.point_idx.shape == (0,) and r.confidence.shape == (0,)


def test_a_camera_voting_twice_for_one_point_is_rejected():
    with pytest.raises(ValueError, match="one vote per camera"):
        _resolve(_votes((0, 0, 0, 0.5), (0, 0, 1, 0.5)), [CAR, CAR])
