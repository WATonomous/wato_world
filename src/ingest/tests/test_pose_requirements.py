"""Pose requirements: dense + smooth pose stream, enforced at ingest.

Covers the chunk-level density abort, the ego-frame check, the held-position
filter (NovAtel repeats), the stretch marks written to poses.parquet, and the
shipped configs.  The stretch rules themselves (and the lookup that applies
them) are tested in src/common/tests/test_pose_lookup.py.  No rosbag:
`poses.messages` is monkeypatched with fake Odometry.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pydantic
import pytest
import yaml

from wato_common.artifact_store import local_path, poses_path
from wato_common.io.parquet_io import read_rows
from wato_ingest.config import IngestConfig, PoseRequirements, load_config
from wato_ingest.decoders import poses

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
REQ = PoseRequirements(min_dense_fraction=0.8, max_bracket_ms=250, max_speed_mps=30)
MS = 1_000_000


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
def _odom(t_ns: int, x: float, *, speed: float = 0.0, child: str = "base_link"):
    """Minimal nav_msgs/Odometry stand-in (only the fields ingest reads)."""
    Odometry = type("Odometry", (), {})
    msg = Odometry()
    msg.header = SimpleNamespace(
        stamp=SimpleNamespace(sec=t_ns // 1_000_000_000, nanosec=t_ns % 1_000_000_000)
    )
    msg.child_frame_id = child
    msg.pose = SimpleNamespace(
        pose=SimpleNamespace(
            position=SimpleNamespace(x=x, y=0.0, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )
    )
    msg.twist = SimpleNamespace(
        twist=SimpleNamespace(linear=SimpleNamespace(x=speed, y=0.0, z=0.0))
    )
    return msg


def _pose_stamped(t_ns: int, x: float, *, with_covariance: bool = False):
    """geometry_msgs/PoseStamped (or PoseWithCovarianceStamped) stand-in."""
    name = "PoseWithCovarianceStamped" if with_covariance else "PoseStamped"
    msg = type(name, (), {})()
    msg.header = SimpleNamespace(
        stamp=SimpleNamespace(sec=t_ns // 1_000_000_000, nanosec=t_ns % 1_000_000_000)
    )
    pose = SimpleNamespace(
        position=SimpleNamespace(x=x, y=0.0, z=0.0),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    msg.pose = SimpleNamespace(pose=pose) if with_covariance else pose
    return msg


def _cfg(**overrides) -> IngestConfig:
    cfg = load_config(str(CONFIG_DIR / "ingest.yaml"))
    return cfg.model_copy(update=overrides)


@pytest.fixture
def fake_bag(monkeypatch, tmp_path):
    """Route poses.extract's bag reads to a list of fake messages and its
    artifact writes into tmp_path.  Returns (set_messages, calls)."""
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")
    state = {"msgs": [], "calls": []}

    @contextmanager
    def fake_messages(bag_path, *, storage_id, topics, t_start_ns, t_end_ns):
        state["calls"].append((t_start_ns, t_end_ns))
        yield iter(
            (topics[0], m, 0)
            for m in state["msgs"]
            if t_start_ns
            <= m.header.stamp.sec * 10**9 + m.header.stamp.nanosec
            <= t_end_ns
        )

    monkeypatch.setattr(poses, "messages", fake_messages)
    return state


def _extract(cfg: IngestConfig, t_end_ns: int = 2_000 * MS):
    return poses.extract("bag", "b", "0000", t_start_ns=0, t_end_ns=t_end_ns, cfg=cfg)


# ---------------------------------------------------------------------------
# Chunk-level density (hard failure)
# ---------------------------------------------------------------------------
def test_dense_stream_passes_and_reports_coverage():
    ts = [i * 100 * MS for i in range(20)]
    assert poses.check_density(ts, REQ, topic="t", chunk_id="c") == pytest.approx(1.0)


def test_keyframe_rate_stream_aborts():
    # eidos main: one pose per 5 m ≈ 900 ms → no trusted bracket anywhere.
    ts = [i * 900 * MS for i in range(30)]
    with pytest.raises(poses.PoseRequirementError, match="too sparse"):
        poses.check_density(
            ts, REQ, topic="/world_modeling/slam/odometry", chunk_id="0000"
        )


def test_bursty_stream_aborts_despite_tiny_median_spacing():
    # may_30's unconverged NovAtel: bursts of fixes 2 ms apart, then seconds
    # of nothing.  Median spacing is tiny; coverage exposes it.
    ts = []
    for burst in range(10):
        ts += [burst * 3_000 * MS + i * 2 * MS for i in range(20)]
    assert np.median(np.diff(ts)) == 2 * MS
    with pytest.raises(poses.PoseRequirementError, match="too sparse"):
        poses.check_density(ts, REQ, topic="t", chunk_id="c")


def test_fewer_than_two_samples_is_not_judged():
    assert poses.check_density([], REQ, topic="t", chunk_id="c") is None
    assert poses.check_density([5], REQ, topic="t", chunk_id="c") is None


def test_one_dropout_in_a_dense_stream_does_not_abort():
    # A single 2 s hole in 12 s is a per-sweep problem, not a sparse stream.
    ts = [i * 100 * MS for i in range(50)] + [
        7_000 * MS + i * 100 * MS for i in range(50)
    ]
    frac = poses.check_density(ts, REQ, topic="t", chunk_id="c")
    assert frac == pytest.approx((4_900 + 4_900) / 11_900)


def test_dense_fraction_counts_only_short_brackets():
    # 100 ms, 100 ms, then one 800 ms gap: 200 of 1000 ms is covered.
    ts = [0, 100 * MS, 200 * MS, 1_000 * MS]
    assert poses.dense_fraction(ts, 250) == pytest.approx(0.2)


def test_extract_aborts_before_writing_on_sparse_stream(fake_bag):
    cfg = _cfg(ego_frame="base_link")
    fake_bag["msgs"] = [_odom(i * 900 * MS, i * 5.0) for i in range(3)]
    with pytest.raises(poses.PoseRequirementError):
        _extract(cfg)
    assert not os.path.exists(local_path(poses_path("b", "0000")))


def test_extract_pads_window_for_bracket_and_publish_latency(fake_bag):
    cfg = _cfg(ego_frame="base_link")
    fake_bag["msgs"] = [_odom(t * MS, 0.0) for t in range(-1_500, 3_600, 100)]
    res = _extract(cfg)
    # max_bracket_ms (250) + the 1 s publish-latency allowance on each side.
    assert fake_bag["calls"] == [(-1_250 * MS, 3_250 * MS)]
    ts = [r["timestamp_ns"] for r in read_rows(poses_path("b", "0000"))]
    # Samples outside the chunk window are kept so edge sweeps and camera
    # frames are still bracketed.
    assert min(ts) == -1_200 * MS and max(ts) == 3_200 * MS
    assert res.dense_fraction == pytest.approx(1.0)


def test_extract_marks_untrusted_stretches(fake_bag):
    cfg = _cfg(ego_frame="base_link")
    # 1 m/s at 100 ms spacing, one 400 ms dropout (200 → 600 ms), and a 3.3 m
    # jump between 2.9 s and 3.0 s (3.4 m in 100 ms).
    times = [t for t in range(0, 6_001, 100) if t not in (300, 400, 500)]
    fake_bag["msgs"] = [
        _odom(t * MS, t / 1e3 + (3.3 if t >= 3_000 else 0.0)) for t in times
    ]
    _extract(cfg, t_end_ns=6_000 * MS)
    rows = read_rows(poses_path("b", "0000"))
    reasons = {r["timestamp_ns"] // MS: r["interval_drop_reason"] for r in rows}
    assert reasons[200] == "pose_gap_400ms"
    assert reasons[2_900] == "pose_jump_34mps"
    assert reasons[6_000] is None  # last sample: no stretch after it
    assert sum(r is not None for r in reasons.values()) == 2


# ---------------------------------------------------------------------------
# Ego frame
# ---------------------------------------------------------------------------
def test_child_frame_matching_ego_frame_passes():
    poses.check_child_frame({"base_link"}, "base_link", "t")


def test_child_frame_mismatch_aborts():
    # /novatel/oem7/odom (child base_link) with the eidos profile's
    # base_footprint: every point would be 1.76 m off on the WATO rig.
    with pytest.raises(poses.PoseRequirementError, match="ego_frame"):
        poses.check_child_frame({"base_link"}, "base_footprint", "/novatel/oem7/odom")


def test_empty_child_frame_only_warns():
    poses.check_child_frame({""}, "base_link", "t")


def test_extract_rejects_wrong_child_frame(fake_bag):
    cfg = _cfg(ego_frame="base_footprint")
    fake_bag["msgs"] = [_odom(i * 100 * MS, 0.0, child="base_link") for i in range(10)]
    with pytest.raises(poses.PoseRequirementError, match="child_frame_id"):
        _extract(cfg)


# ---------------------------------------------------------------------------
# Held positions (NovAtel: 100 Hz messages, 50 Hz position)
# ---------------------------------------------------------------------------
def _row(t_ms: int, x: float) -> dict:
    return {"timestamp_ns": t_ms * MS, "x": x, "y": 0.0, "z": 0.0}


def test_repeated_position_while_moving_is_dropped():
    samples = [
        (_row(0, 0.0), 7.0),
        (_row(10, 0.0), 7.0),  # repeat of the 0 ms fix under a newer stamp
        (_row(20, 0.14), 7.0),
        (_row(30, 0.14), 7.0),  # repeat
        (_row(40, 0.28), 7.0),
    ]
    kept, dropped = poses._drop_held_positions(samples)
    assert dropped == 2
    assert [r["timestamp_ns"] // MS for r in kept] == [0, 20, 40]


def test_repeated_position_while_stationary_is_kept():
    samples = [(_row(t, 1.0), 0.0) for t in range(0, 100, 10)]
    kept, dropped = poses._drop_held_positions(samples)
    assert dropped == 0 and len(kept) == 10


def test_duplicate_stamps_keep_first():
    out = poses._dedupe_and_sort(
        [(_row(10, 2.0), 0.0), (_row(0, 0.0), 0.0), (_row(10, 9.0), 0.0)]
    )
    assert [(r["timestamp_ns"] // MS, r["x"]) for r, _ in out] == [(0, 0.0), (10, 2.0)]


# ---------------------------------------------------------------------------
# Pose message types
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("with_covariance", [False, True])
def test_pose_stamped_topics_are_extracted(fake_bag, with_covariance):
    # A Pose message names no child frame, so there is nothing to check
    # against ego_frame — the config is taken on trust.
    fake_bag["msgs"] = [
        _pose_stamped(t * 20 * MS, t * 0.2, with_covariance=with_covariance)
        for t in range(10)
    ]
    result = poses.extract(
        "bag", "b", "0000", t_start_ns=0, t_end_ns=200 * MS, cfg=_cfg()
    )
    rows = read_rows(poses_path("b", "0000"))
    assert result.rows_written == 10
    assert [r["x"] for r in rows] == pytest.approx([t * 0.2 for t in range(10)])


def test_other_message_types_are_not_poses():
    assert poses.parse_pose_msg(type("Imu", (), {})()) is None


# ---------------------------------------------------------------------------
# Shipped configs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["ingest.yaml", "ingest.wato.yaml"])
def test_shipped_profiles_load(name):
    cfg = load_config(str(CONFIG_DIR / name))
    assert 0 < cfg.pose_requirements.min_dense_fraction <= 1
    assert cfg.storage_id == ""  # detected from the bag, never per dataset


def test_old_max_pose_gap_key_is_rejected():
    data = yaml.safe_load((CONFIG_DIR / "ingest.yaml").read_text())
    data["max_pose_gap_ms"] = 200.0
    with pytest.raises(pydantic.ValidationError):
        IngestConfig.model_validate(data)
