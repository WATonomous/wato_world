"""validate-bag / trim-bag analysis: header parsing, gaps, window, pairing, tf."""

import struct
from pathlib import Path

import numpy as np
import pytest

from wato_ingest.config import load_config
from wato_ingest.inputs import bag_check
from wato_ingest.inputs.bag_check import BagScan, TopicScan

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
MS = 1_000_000


def _cdr_header(sec: int, nanosec: int, frame_id: str, little: bool = True) -> bytes:
    fmt = "<iII" if little else ">iII"
    enc = b"\x00\x01\x00\x00" if little else b"\x00\x00\x00\x00"
    s = frame_id.encode() + b"\x00"
    return enc + struct.pack(fmt, sec, nanosec, len(s)) + s + b"\xde\xad"


def _topic(name, type_, log_ms, header_ms=None, frame_ids=()):
    return TopicScan(
        name=name,
        type=type_,
        log_ns=np.asarray(log_ms, dtype=np.int64) * MS,
        header_ns=None if header_ms is None else np.asarray(header_ms, np.int64) * MS,
        frame_ids=set(frame_ids),
        header_note="" if header_ms is not None else "no header",
    )


def _bag(*topics, tf=None):
    log = np.concatenate([t.log_ns for t in topics if len(t.log_ns)])
    return BagScan(
        path="bag.mcap",
        start_ns=int(log.min()),
        end_ns=int(log.max()),
        topics={t.name: t for t in topics},
        tf_static=tf or {},
    )


# ---------------------------------------------------------------------------
@pytest.mark.parametrize("little", [True, False])
def test_parse_cdr_header(little):
    stamp, frame_id = bag_check.parse_cdr_header(
        _cdr_header(1773091077, 687519938, "camera_pano_nn", little)
    )
    assert stamp == 1773091077_687519938
    assert frame_id == "camera_pano_nn"


def test_parse_cdr_header_rejects_non_cdr():
    with pytest.raises(ValueError, match="not a plain CDR"):
        bag_check.parse_cdr_header(b"\x00\x07\x00\x00" + b"\x00" * 16)


# ---------------------------------------------------------------------------
def test_gap_is_more_than_factor_times_median_period():
    # 100 ms period; one message missing between 300 and 500 (a 200 ms step).
    t = _topic("/lidar", "sensor_msgs/msg/PointCloud2", [0, 100, 200, 300, 500, 600])
    r = bag_check.topic_report(t, _bag(t), gap_factor=1.5, static=False)
    assert r.median_period_ns == 100 * MS
    assert [(g.start_ns, g.end_ns) for g in r.gaps] == [(300 * MS, 500 * MS)]
    # At 2.5x the same single drop no longer counts.
    assert bag_check.topic_report(t, _bag(t), 2.5, static=False).gaps == []


def test_missing_start_and_end_against_the_bag_span():
    lidar = _topic("/lidar", "x/msg/A", np.arange(0, 1001, 100))
    cam = _topic("/cam", "x/msg/B", np.arange(400, 801, 100))
    bag = _bag(lidar, cam)
    r = bag_check.topic_report(cam, bag, 1.5, static=False)
    assert r.missing_start_ns == 400 * MS
    assert r.missing_end_ns == 200 * MS
    # Within one period of the bag's ends is not "missing".
    assert bag_check.topic_report(lidar, bag, 1.5, static=False).missing_start_ns == 0


def test_static_topics_are_not_judged():
    tf = _topic("/tf_static", "tf2_msgs/msg/TFMessage", [0, 5000])
    lidar = _topic("/lidar", "x/msg/A", np.arange(0, 10001, 100))
    r = bag_check.topic_report(tf, _bag(tf, lidar), 1.5, static=True)
    assert r.gaps == [] and r.missing_end_ns == 0 and r.median_period_ns is None


def test_header_skew():
    t = _topic("/cam", "x/msg/A", [100, 200, 300], header_ms=[90, 150, 300])
    r = bag_check.topic_report(t, _bag(t), 1.5, static=False)
    assert r.skew_median_ns == -10 * MS
    assert r.skew_max_abs_ns == 50 * MS
    assert (r.header_first_ns, r.header_last_ns) == (90 * MS, 300 * MS)


# ---------------------------------------------------------------------------
def test_common_window_is_latest_start_to_earliest_end():
    lidar = _topic("/lidar", "x/msg/A", np.arange(0, 1001, 50))
    cam = _topic("/cam", "x/msg/B", np.arange(400, 1000, 80))  # last 960
    gps = _topic("/gps", "x/msg/C", np.arange(10, 990, 100))
    w = bag_check.common_window(_bag(lidar, cam, gps), set())
    assert (w.start_ns, w.end_ns) == (400 * MS, 910 * MS)
    assert w.started_last == ["/cam"]
    assert w.ended_first == ["/gps"]


def test_one_shot_topic_empties_the_window_unless_static():
    lidar = _topic("/lidar", "x/msg/A", np.arange(0, 1001, 50))
    once = _topic("/slam/odometry", "x/msg/B", [20])
    w = bag_check.common_window(_bag(lidar, once), set())
    assert w.start_ns is None and "--static-topic" in w.reason
    w = bag_check.common_window(_bag(lidar, once), {"/slam/odometry"})
    assert (w.start_ns, w.end_ns) == (0, 1000 * MS)


def test_topic_with_no_messages_has_no_window():
    lidar = _topic("/lidar", "x/msg/A", [0, 100])
    empty = _topic("/cam", "x/msg/B", [])
    w = bag_check.common_window(_bag(lidar, empty), set())
    assert w.start_ns is None and "/cam" in w.reason


# ---------------------------------------------------------------------------
def test_camera_pairing_by_namespace_and_stamp():
    img = _topic("/cam_a/image_rect", "sensor_msgs/msg/Image", [0, 1, 2], [10, 20, 30])
    info = _topic(
        "/cam_a/camera_info", "sensor_msgs/msg/CameraInfo", [0, 1, 2], [10, 30, 40]
    )
    orphan = _topic("/cam_b/image_rect", "sensor_msgs/msg/CompressedImage", [0], [10])
    pairs, unpaired = bag_check.camera_pairings(_bag(img, info, orphan))
    assert unpaired == ["/cam_b/image_rect"]
    (p,) = pairs
    assert (p.image_topic, p.info_topic) == ("/cam_a/image_rect", "/cam_a/camera_info")
    assert (p.images_without_info, p.infos_without_image) == (1, 1)


def test_unresolved_sensor_frames():
    lidar = _topic(
        "/lidar", "sensor_msgs/msg/PointCloud2", [0], [0], frame_ids=["lidar_cc"]
    )
    cam = _topic("/cam/image", "sensor_msgs/msg/Image", [0], [0], frame_ids=["cam_x"])
    odom = _topic("/odom", "nav_msgs/msg/Odometry", [0], [0], frame_ids=["map"])
    bag = _bag(lidar, cam, odom, tf={("base_link", "lidar_cc"): np.eye(4)})
    # Odometry's frame isn't a sensor frame, so it isn't required.
    assert bag_check.unresolved_frames(bag, "base_link") == {"cam_x": ["/cam/image"]}


# ---------------------------------------------------------------------------
def test_report_flags_profile_mismatch_and_renders():
    cfg = load_config(str(CONFIG_DIR / "ingest.wato.yaml"))
    lidar = _topic(
        "/lidar_cc/velodyne_points",
        "sensor_msgs/msg/PointCloud2",
        np.arange(0, 1001, 50),
        np.arange(0, 1001, 50),
        frame_ids=["lidar_cc"],
    )
    tf = _topic("/tf_static", "tf2_msgs/msg/TFMessage", [0])
    bag = _bag(lidar, tf, tf={("base_footprint", "lidar_cc"): np.eye(4)})
    r = bag_check.build_report(bag, cfg, static_topics={"/tf_static"}, gap_factor=1.5)
    assert r.window.start_ns == 0
    assert not r.unresolved_frames
    assert any(f.startswith("ingest profile:") for f in r.failures)
    text = bag_check.render(r)
    assert "COMMON WINDOW" in text and "FAIL ingest profile" in text


def test_report_rejects_unknown_static_topic():
    cfg = load_config(str(CONFIG_DIR / "ingest.wato.yaml"))
    lidar = _topic("/lidar", "x/msg/A", [0, 100])
    with pytest.raises(ValueError, match="/nope"):
        bag_check.build_report(
            _bag(lidar), cfg, static_topics={"/nope"}, gap_factor=1.5
        )


def test_only_profile_sensor_frames_fail_the_extrinsics_check():
    cfg = load_config(str(CONFIG_DIR / "ingest.wato.yaml"))
    lidar = _topic(
        "/lidar_cc/velodyne_points",
        "sensor_msgs/msg/PointCloud2",
        [0, 50, 100],
        [0, 50, 100],
        frame_ids=["lidar_cc"],
    )
    world_cloud = _topic(
        "/world_modeling/liso/submap",
        "sensor_msgs/msg/PointCloud2",
        [0, 50, 100],
        [0, 50, 100],
        frame_ids=["map"],
    )
    bag = _bag(lidar, world_cloud, tf={("base_footprint", "lidar_cc"): np.eye(4)})
    r = bag_check.build_report(bag, cfg, static_topics=set(), gap_factor=1.5)
    assert r.unresolved_frames == {}
    assert r.unresolved_other_frames == {"map": ["/world_modeling/liso/submap"]}
    assert not any(f.startswith("no tf_static chain") for f in r.failures)

    bag.tf_static = {}
    r = bag_check.build_report(bag, cfg, static_topics=set(), gap_factor=1.5)
    assert r.unresolved_frames == {"lidar_cc": ["/lidar_cc/velodyne_points"]}
