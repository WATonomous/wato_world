"""Tests for voting.py — cross-camera vote aggregation."""

from __future__ import annotations

import pytest

from wato_semantic_lifting.voting import LabeledPoint, PointVote, accumulate_votes


def _vote(pt_idx: int, instance: str, cls: str, score: float, cam: str) -> PointVote:
    return PointVote(
        point_idx=pt_idx,
        instance_id=instance,
        cls=cls,
        score=score,
        cam_id=cam,
        mask_area=100,
    )


def test_single_vote_wins_with_confidence_one():
    votes = [_vote(0, "obj1", "car", 0.9, "cam_front")]
    result = accumulate_votes(votes, n_points=1)
    assert len(result) == 1
    assert result[0].instance_id == "obj1"
    assert result[0].confidence == pytest.approx(1.0)
    assert result[0].n_supporting_cameras == 1
    assert result[0].n_disagreeing_cameras == 0


def test_two_cameras_agree_high_confidence():
    """Two cameras vote for the same instance → confidence 1.0, 2 supporting cameras."""
    votes = [
        _vote(0, "obj1", "car", 0.8, "cam_front"),
        _vote(0, "obj1", "car", 0.9, "cam_back"),
    ]
    result = accumulate_votes(votes, n_points=1)
    assert len(result) == 1
    assert result[0].instance_id == "obj1"
    assert result[0].confidence == pytest.approx(1.0)
    assert result[0].n_supporting_cameras == 2
    assert result[0].n_disagreeing_cameras == 0


def test_two_cameras_disagree_winner_by_score():
    """Winner is the instance with higher cumulative score."""
    votes = [
        _vote(0, "obj1", "car", 0.9, "cam_front"),
        _vote(0, "obj2", "truck", 0.3, "cam_back"),
    ]
    result = accumulate_votes(votes, n_points=1)
    assert result[0].instance_id == "obj1"
    assert result[0].n_disagreeing_cameras == 1
    # confidence = 0.9 / (0.9 + 0.3) = 0.75
    assert result[0].confidence == pytest.approx(0.75, abs=0.01)


def test_multiple_points_independently_resolved():
    votes = [
        _vote(0, "A", "car", 1.0, "cam"),
        _vote(1, "B", "truck", 1.0, "cam"),
    ]
    result = accumulate_votes(votes, n_points=2)
    assert len(result) == 2
    by_idx = {r.point_idx: r for r in result}
    assert by_idx[0].instance_id == "A"
    assert by_idx[1].instance_id == "B"


def test_empty_votes_returns_empty():
    result = accumulate_votes([], n_points=10)
    assert result == []


def test_point_with_no_vote_not_in_result():
    """Points that received no vote are absent from the result."""
    votes = [_vote(5, "obj1", "car", 0.8, "cam")]
    result = accumulate_votes(votes, n_points=10)
    assert len(result) == 1
    assert result[0].point_idx == 5
