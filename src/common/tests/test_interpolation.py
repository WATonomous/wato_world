"""Tests for wato_common.geometry.interpolation: interpolate_pose (PoseLookup)
and batch_interpolate_poses (deskew). SLERP of the rotation is covered
through PoseLookup in test_pose_lookup.py."""

from __future__ import annotations

import numpy as np
import pytest

from wato_common.geometry import PoseSample, batch_interpolate_poses, interpolate_pose


def _make_sample(ts_ns: int, tx: float) -> PoseSample:
    return PoseSample(
        timestamp_ns=ts_ns,
        translation=np.array([tx, 0.0, 0.0]),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
    )


SAMPLES = [
    _make_sample(0, 0.0),
    _make_sample(1_000_000_000, 1.0),
    _make_sample(2_000_000_000, 2.0),
]


def test_interpolate_translation_is_linear():
    T, err = interpolate_pose(SAMPLES, 500_000_000)
    np.testing.assert_allclose(T[:3, 3], [0.5, 0.0, 0.0], atol=1e-9)
    assert err == 500_000_000


def test_interpolate_clamps_to_endpoints_when_target_out_of_range():
    T, err = interpolate_pose(SAMPLES, -1_000_000_000)
    np.testing.assert_allclose(T[:3, 3], [0.0, 0.0, 0.0])
    assert err == 1_000_000_000
    T, err = interpolate_pose(SAMPLES, 5_000_000_000)
    np.testing.assert_allclose(T[:3, 3], [2.0, 0.0, 0.0])
    assert err == 3_000_000_000


def test_single_sample_returns_that_sample():
    T, err = interpolate_pose([_make_sample(123, 7.0)], 200)
    np.testing.assert_allclose(T[:3, 3], [7.0, 0.0, 0.0])
    assert err == 77


def test_batch_matches_scalar_inside_and_outside_the_span():
    ts = np.array(
        [-1_000_000, 0, 500_000_000, 1_000_000_000, 2_000_000_000, 5_000_000_000],
        dtype=np.int64,
    )
    batch = batch_interpolate_poses(SAMPLES, ts)
    assert batch.shape == (len(ts), 4, 4)
    for i, t in enumerate(ts):
        scalar, _ = interpolate_pose(SAMPLES, int(t))
        np.testing.assert_allclose(batch[i], scalar, atol=1e-10)


def test_batch_single_sample():
    ts = np.array([0, 500_000_000, 1_000_000_000], dtype=np.int64)
    batch = batch_interpolate_poses([_make_sample(500_000_000, 3.0)], ts)
    assert batch.shape == (3, 4, 4)
    np.testing.assert_allclose(batch[:, :3, 3], [[3.0, 0.0, 0.0]] * 3, atol=1e-10)


def test_batch_empty_timestamps():
    batch = batch_interpolate_poses(SAMPLES, np.empty(0, dtype=np.int64))
    assert batch.shape == (0, 4, 4)


def test_empty_samples_raise():
    with pytest.raises(ValueError, match="empty"):
        interpolate_pose([], 0)
    with pytest.raises(ValueError, match="empty"):
        batch_interpolate_poses([], np.array([0], dtype=np.int64))
