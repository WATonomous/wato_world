"""Tests for Step B's point-level cleanup (classify/dynamic_filter.py)."""

from __future__ import annotations

import numpy as np
import pytest

from wato_lidar_preprocessing.classify.dynamic_filter import (
    cluster_keep,
    filter_candidates,
    persistence_keep,
    sweep_rate_hz,
)
from wato_lidar_preprocessing.config import DynamicFilterParams


def _blob(center, n: int, spread: float = 0.1) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.asarray(center) + rng.uniform(-spread, spread, size=(n, 3))


def test_persistence_keeps_movers_drops_dwellers():
    """A voxel occupied across many sweeps is structure; one a point crosses
    once is a mover."""
    dweller = np.tile([[5.0, 5.0, 1.0]], (30, 1))
    mover = np.column_stack([np.arange(30) * 2.0, np.zeros(30), np.ones(30)])
    xyz = np.concatenate([dweller, mover])
    sweep = np.concatenate([np.arange(30), np.arange(30)])

    keep = persistence_keep(xyz, sweep, voxel_m=0.5, max_sweeps=20)

    assert not keep[:30].any()
    assert keep[30:].all()


def test_persistence_counts_sweeps_not_points():
    """Many points in one voxel in one sweep is not persistence."""
    xyz = np.tile([[5.0, 5.0, 1.0]], (100, 1))
    keep = persistence_keep(xyz, np.zeros(100), voxel_m=0.5, max_sweeps=20)
    assert keep.all()


def test_persistence_disabled_is_noop():
    xyz = np.tile([[5.0, 5.0, 1.0]], (30, 1))
    assert persistence_keep(xyz, np.arange(30), 0.5, 0).all()


def test_cluster_keeps_blob_drops_isolated_point():
    blob = _blob([10.0, 0.0, 1.0], 20)
    lone = np.array([[30.0, 30.0, 1.0]])
    keep = cluster_keep(np.concatenate([blob, lone]), np.zeros(21), 0.5, 5)
    assert keep[:20].all()
    assert not keep[20]


def test_cluster_connects_diagonal_voxels():
    """Voxels touching at a corner are one cluster (26-connectivity)."""
    xyz = np.array(
        [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [1.25, 1.25, 1.25]], dtype=float
    )
    assert cluster_keep(xyz, np.zeros(3), 0.5, 3).all()


def test_cluster_is_per_frame():
    """The same spot hit once per frame is speckle, however many frames: only
    points seen together form a cluster."""
    xyz = np.tile([[10.0, 0.0, 1.0]], (20, 1))
    assert not cluster_keep(xyz, np.arange(20), 0.5, 5).any()
    assert cluster_keep(xyz, np.zeros(20), 0.5, 5).all()


def test_cluster_disabled_is_noop():
    assert cluster_keep(np.zeros((1, 3)), np.zeros(1), 0.5, 0).all()


def test_sweep_rate_counts_every_scanner():
    # Three scanners at 20 Hz: 60 sweeps per second.
    stamps = [int(i * 1e9 / 60) for i in range(121)]
    assert sweep_rate_hz(stamps) == pytest.approx(60.0, rel=1e-6)
    assert sweep_rate_hz([5]) == 0.0
    assert sweep_rate_hz([5, 5]) == 0.0


def test_filter_turns_duration_into_sweeps_and_splits_counts():
    """persistence_max_s scales with the sweep rate, and drops are attributed
    to the first test that removes a point."""
    params = DynamicFilterParams(min_cluster_pts=5, persistence_max_s=0.5)
    n = 40
    # A blob present in 40 sweeps: passes the cluster test, fails persistence
    # at 60 Hz (0.5 s = 30 sweeps).
    dwell = np.tile(_blob([0.0, 0.0, 1.0], 6), (n, 1))
    d_sweep = np.repeat(np.arange(n), 6)
    lone = np.array([[50.0, 50.0, 1.0]])
    xyz = np.concatenate([dwell, lone])
    sweep = np.concatenate([d_sweep, [0]])
    frame = sweep.copy()

    res = filter_candidates(params, xyz, sweep, frame, rate_hz=60.0)
    assert not res.keep.any()
    assert res.n_cluster_dropped == 1
    assert res.n_persistence_dropped == 6 * n

    # At 100 Hz, 0.5 s is 50 sweeps: 40 sweeps of dwell no longer persist.
    res = filter_candidates(params, xyz, sweep, frame, rate_hz=100.0)
    assert res.keep[:-1].all()
    assert res.n_persistence_dropped == 0


def test_filter_skips_persistence_without_a_usable_rate():
    """A one-sweep chunk has no rate; a sweep count below 2 would drop every
    point, so the test is skipped rather than applied."""
    params = DynamicFilterParams(min_cluster_pts=0, persistence_max_s=0.5)
    xyz = _blob([0.0, 0.0, 1.0], 6)
    assert filter_candidates(params, xyz, np.zeros(6), np.zeros(6), 0.0).keep.all()
    assert filter_candidates(params, xyz, np.zeros(6), np.zeros(6), 2.0).keep.all()


def test_filter_disabled_is_noop():
    params = DynamicFilterParams(min_cluster_pts=0, persistence_max_s=0.0)
    xyz = _blob([0.0, 0.0, 1.0], 6)
    res = filter_candidates(params, xyz, np.zeros(6), np.zeros(6), 60.0)
    assert res.keep.all()
    assert res.n_cluster_dropped == res.n_persistence_dropped == 0


def test_filter_handles_no_candidates():
    res = filter_candidates(
        DynamicFilterParams(),
        np.empty((0, 3)),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        60.0,
    )
    assert res.keep.shape == (0,)
