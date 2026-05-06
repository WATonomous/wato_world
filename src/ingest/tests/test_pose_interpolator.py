"""Tests for pose SLERP + linear interpolation."""

from __future__ import annotations

import numpy as np
import pytest

from wato_common.geometry import PoseSample, interpolate_pose, slerp


def _identity_quat() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 1.0])


def _yaw_quat(yaw_rad: float) -> np.ndarray:
    return np.array([0.0, 0.0, np.sin(yaw_rad / 2), np.cos(yaw_rad / 2)])


def test_slerp_identity_to_identity_is_identity():
    q = slerp(_identity_quat(), _identity_quat(), 0.5)
    np.testing.assert_allclose(q, _identity_quat(), atol=1e-9)


def test_slerp_halfway_between_0_and_pi_yaw():
    q = slerp(_identity_quat(), _yaw_quat(np.pi / 2), 0.5)
    expected = _yaw_quat(np.pi / 4)
    # Quaternions q and -q represent the same rotation; allow either sign.
    assert np.allclose(q, expected, atol=1e-9) or np.allclose(q, -expected, atol=1e-9)


def test_interpolate_translation_is_linear():
    samples = [
        PoseSample(0, np.array([0.0, 0.0, 0.0]), _identity_quat()),
        PoseSample(1_000_000_000, np.array([10.0, 0.0, 0.0]), _identity_quat()),
    ]
    T, err = interpolate_pose(samples, 500_000_000)
    np.testing.assert_allclose(T[:3, 3], [5.0, 0.0, 0.0], atol=1e-9)
    assert err == 500_000_000


def test_interpolate_clamps_to_endpoints_when_target_out_of_range():
    samples = [
        PoseSample(1_000_000_000, np.array([1.0, 0.0, 0.0]), _identity_quat()),
        PoseSample(2_000_000_000, np.array([2.0, 0.0, 0.0]), _identity_quat()),
    ]
    # Before first sample → returns first.
    T, err = interpolate_pose(samples, 0)
    np.testing.assert_allclose(T[:3, 3], [1.0, 0.0, 0.0])
    assert err == 1_000_000_000
    # After last sample → returns last.
    T, err = interpolate_pose(samples, 5_000_000_000)
    np.testing.assert_allclose(T[:3, 3], [2.0, 0.0, 0.0])
    assert err == 3_000_000_000


def test_empty_pose_samples_raises():
    with pytest.raises(ValueError):
        interpolate_pose([], 0)


def test_single_sample_returns_that_sample():
    samples = [PoseSample(123, np.array([7.0, 8.0, 9.0]), _identity_quat())]
    T, err = interpolate_pose(samples, 200)
    np.testing.assert_allclose(T[:3, 3], [7.0, 8.0, 9.0])
    assert err == 77
