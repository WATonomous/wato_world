"""Tests for batch_interpolate_poses."""

from __future__ import annotations

import numpy as np
import pytest

from wato_common.geometry import PoseSample, batch_interpolate_poses, interpolate_pose
from wato_common.geometry.transforms import make_se3


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


def test_matches_scalar_at_sample_points():
    ts = np.array([0, 1_000_000_000, 2_000_000_000], dtype=np.int64)
    batch = batch_interpolate_poses(SAMPLES, ts)
    for i, t in enumerate(ts):
        scalar, _ = interpolate_pose(SAMPLES, int(t))
        np.testing.assert_allclose(batch[i], scalar, atol=1e-10)


def test_matches_scalar_at_midpoint():
    ts = np.array([500_000_000], dtype=np.int64)
    batch = batch_interpolate_poses(SAMPLES, ts)
    scalar, _ = interpolate_pose(SAMPLES, 500_000_000)
    np.testing.assert_allclose(batch[0], scalar, atol=1e-10)


def test_clamps_before_first_sample():
    ts = np.array([-1_000_000], dtype=np.int64)
    batch = batch_interpolate_poses(SAMPLES, ts)
    scalar, _ = interpolate_pose(SAMPLES, -1_000_000)
    np.testing.assert_allclose(batch[0], scalar, atol=1e-10)


def test_clamps_after_last_sample():
    ts = np.array([5_000_000_000], dtype=np.int64)
    batch = batch_interpolate_poses(SAMPLES, ts)
    scalar, _ = interpolate_pose(SAMPLES, 5_000_000_000)
    np.testing.assert_allclose(batch[0], scalar, atol=1e-10)


def test_single_sample():
    single = [_make_sample(500_000_000, 3.0)]
    ts = np.array([0, 500_000_000, 1_000_000_000], dtype=np.int64)
    batch = batch_interpolate_poses(single, ts)
    assert batch.shape == (3, 4, 4)
    for i in range(3):
        np.testing.assert_allclose(batch[i, :3, 3], [3.0, 0.0, 0.0], atol=1e-10)


def test_output_shape():
    ts = np.arange(10, dtype=np.int64) * 100_000_000
    batch = batch_interpolate_poses(SAMPLES, ts)
    assert batch.shape == (10, 4, 4)


def test_empty_timestamps():
    ts = np.empty(0, dtype=np.int64)
    batch = batch_interpolate_poses(SAMPLES, ts)
    assert batch.shape == (0, 4, 4)


def test_raises_on_empty_samples():
    with pytest.raises(ValueError, match="empty"):
        batch_interpolate_poses([], np.array([0], dtype=np.int64))
