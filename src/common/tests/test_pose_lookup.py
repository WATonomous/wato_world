"""Tests for wato_common.pose_lookup: the stretch rules and the lookup."""

from __future__ import annotations

import numpy as np
import pytest

from wato_common.artifact_store import poses_path
from wato_common.geometry import PoseSample
from wato_common.io.parquet_io import write_table
from wato_common.pose_lookup import PoseLookup, interval_drop_reasons
from wato_common.schemas import POSES_SCHEMA

MS = 1_000_000


def _samples(points: list[tuple[int, float]]) -> list[PoseSample]:
    return [
        PoseSample(t * MS, np.array([x, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))
        for t, x in points
    ]


def _lookup(samples: list[PoseSample]) -> PoseLookup:
    """Marked with the shipped ingest defaults (250 ms, 30 m/s)."""
    return PoseLookup(
        samples,
        interval_drop_reasons(samples, max_bracket_ns=250 * MS, max_speed_mps=30.0),
    )


def _at(samples, t_ms: int):
    return _lookup(samples).at(t_ms * MS)


# ---------------------------------------------------------------------------
# Stretch rules
# ---------------------------------------------------------------------------
def test_short_plausible_stretch_is_valid():
    p = _at(_samples([(0, 0.0), (100, 1.0)]), 50)
    assert p.valid and p.drop_reason is None
    np.testing.assert_allclose(p.world_T_ego[:3, 3], [0.5, 0.0, 0.0])


def test_long_stretch_is_a_gap():
    p = _at(_samples([(0, 0.0), (900, 5.0)]), 450)
    assert not p.valid and p.drop_reason == "pose_gap_900ms"


def test_fast_stretch_is_a_jump():
    # 4.45 m in 100 ms = 44.5 m/s: the largest step in ring_road_corrected_part1's
    # slam/odometry.
    p = _at(_samples([(0, 0.0), (100, 4.45)]), 50)
    assert not p.valid and p.drop_reason == "pose_jump_44mps"


def test_small_step_over_jittery_stamps_is_not_a_jump():
    # NovAtel: a normal 0.15 m step stamped 0.3 ms after the previous fix
    # reads as 500 m/s raw; measured over one sweep period (50 ms) it's 3 m/s.
    s = [
        PoseSample(0, np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])),
        PoseSample(300_000, np.array([0.15, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0])),
    ]
    assert _lookup(s).at(150_000).valid


def test_large_step_over_a_short_stretch_is_a_jump():
    # 2 m in 1 ms: beyond max_speed × 50 ms = 1.5 m even at the floor.
    s = [
        PoseSample(0, np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])),
        PoseSample(1 * MS, np.array([2.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0])),
    ]
    assert _lookup(s).at(500_000).drop_reason == "pose_jump_40mps"


def test_last_sample_has_no_stretch():
    reasons = interval_drop_reasons(
        _samples([(0, 0.0), (100, 1.0), (2_000, 2.0)]),
        max_bracket_ns=250 * MS,
        max_speed_mps=30.0,
    )
    assert reasons == [None, "pose_gap_1900ms", None]
    assert interval_drop_reasons([], max_bracket_ns=1, max_speed_mps=1.0) == []


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------
def test_no_extrapolation_outside_the_sampled_span():
    s = _samples([(100, 0.0), (200, 1.0)])
    assert _at(s, 50).drop_reason == "outside_pose_span"
    assert _at(s, 250).drop_reason == "outside_pose_span"


def test_exactly_on_a_sample_is_valid_even_next_to_a_gap():
    p = _at(_samples([(0, 0.0), (100, 1.0), (2_000, 20.0)]), 100)
    assert p.valid and p.interp_error_ns == 0.0
    np.testing.assert_allclose(p.world_T_ego[:3, 3], [1.0, 0.0, 0.0])


def test_no_samples():
    p = PoseLookup([]).at(0)
    assert not p.valid and p.drop_reason == "no_pose_samples"


def test_each_stretch_is_judged_on_its_own():
    # A camera frame 40 ms after a valid sweep can land in the next stretch.
    lookup = _lookup(_samples([(0, 0.0), (100, 1.0), (1_100, 11.0)]))
    assert lookup.at(80 * MS).valid
    assert lookup.at(120 * MS).drop_reason == "pose_gap_1000ms"


def test_pose_is_interpolated_at_the_queried_time():
    lookup = _lookup(_samples([(0, 0.0), (100, 1.0), (200, 3.0)]))
    sweep, camera = lookup.at(90 * MS), lookup.at(130 * MS)
    np.testing.assert_allclose(sweep.world_T_ego[:3, 3], [0.9, 0.0, 0.0])
    np.testing.assert_allclose(camera.world_T_ego[:3, 3], [1.6, 0.0, 0.0])
    assert camera.pose_timestamp_ns == 100 * MS  # nearest sample
    assert camera.interp_error_ns == 30 * MS


def test_rotation_is_slerped():
    yaw = np.pi / 2
    s = [
        PoseSample(0, np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])),
        PoseSample(
            100 * MS,
            np.zeros(3),
            np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]),
        ),
    ]
    R = _lookup(s).at(50 * MS).world_T_ego[:3, :3]
    np.testing.assert_allclose(np.arctan2(R[1, 0], R[0, 0]), yaw / 2, atol=1e-9)


def test_unsorted_samples_and_reasons_stay_paired():
    s = _samples([(0, 0.0), (100, 1.0), (1_000, 2.0)])
    reasons = [None, "pose_gap_900ms", None]
    lookup = PoseLookup([s[2], s[0], s[1]], [reasons[2], reasons[0], reasons[1]])
    assert lookup.at(50 * MS).valid
    assert lookup.at(500 * MS).drop_reason == "pose_gap_900ms"


def test_reason_count_must_match_samples():
    with pytest.raises(ValueError):
        PoseLookup(_samples([(0, 0.0), (100, 1.0)]), [None])


# ---------------------------------------------------------------------------
# Loading poses.parquet
# ---------------------------------------------------------------------------
def _row(t_ms: int, x: float, **extra) -> dict:
    return {
        "bag_id": "b",
        "chunk_id": "0000",
        "timestamp_ns": t_ms * MS,
        "x": x,
        "y": 0.0,
        "z": 0.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
        "world_T_ego_flat": np.eye(4).flatten().tolist(),
        "source": "/odom",
        "valid": True,
        **extra,
    }


@pytest.fixture
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")
    return tmp_path


def test_load_reads_interval_reasons(artifact_root):
    rows = [
        _row(0, 0.0, interval_drop_reason=None),
        _row(100, 1.0, interval_drop_reason="pose_gap_900ms"),
        _row(1_000, 2.0, interval_drop_reason=None),
        _row(1_050, 9.0, valid=False, interval_drop_reason=None),
    ]
    write_table(rows, POSES_SCHEMA, poses_path("b", "0000"))
    lookup = PoseLookup.load("b", "0000")
    assert len(lookup) == 3  # the valid=False row is skipped
    assert lookup.at(50 * MS).valid
    assert lookup.at(500 * MS).drop_reason == "pose_gap_900ms"


def test_load_without_interval_column_trusts_every_stretch(artifact_root, caplog):
    import pyarrow as pa

    old_schema = pa.schema(
        [f for f in POSES_SCHEMA if f.name != "interval_drop_reason"]
    )
    write_table([_row(0, 0.0), _row(1_000, 1.0)], old_schema, poses_path("b", "0000"))
    lookup = PoseLookup.load("b", "0000")
    assert lookup.at(500 * MS).valid
    assert "re-run ingest" in caplog.text
