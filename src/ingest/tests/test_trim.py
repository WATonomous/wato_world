"""trim-bag: window cut via the mcap CLI, static topics copied whole."""

import shutil

import numpy as np
import pytest

from wato_ingest.inputs import trim
from wato_ingest.inputs.bag_check import BagScan, TopicScan

MS = 1_000_000


def test_window_and_static_commands(tmp_path):
    src, w, s, out = (tmp_path / n for n in ("in.mcap", "w", "s", "out.mcap"))
    cmds = trim.mcap_commands(
        src, w, s, out,
        start_ns=100, end_ns=200,
        static_topics=["/slam/odometry", "/tf_static"],
        compression="none",
    )  # fmt: skip
    window, static, merge = cmds
    assert window[:5] == ["mcap", "filter", str(src), "-o", str(w)]
    # --end is exclusive, so the message at end_ns needs end_ns + 1.
    assert window[window.index("--start") + 1] == "100"
    assert window[window.index("--end") + 1] == "201"
    assert window.count("--exclude-topic-regex") == 2
    # Anchored: excluding /tf_static must not touch /tf.
    assert "^/tf_static$" in window
    assert "--start" not in static and "--end" not in static
    assert static.count("--include-topic-regex") == 2
    assert merge[:6] == ["mcap", "merge", str(w), str(s), "-o", str(out)]
    assert all(c[c.index("--compression") + 1] == "none" for c in cmds)


def test_no_static_topics_filters_straight_to_output(tmp_path):
    (cmd,) = trim.mcap_commands(
        tmp_path / "in.mcap", tmp_path / "w", tmp_path / "s", tmp_path / "out.mcap",
        start_ns=0, end_ns=1, static_topics=[], compression="zstd",
    )  # fmt: skip
    assert cmd[4] == str(tmp_path / "out.mcap")
    assert "--exclude-topic-regex" not in cmd


def test_expected_counts():
    def t(name, ms):
        return TopicScan(name, "x/msg/A", np.asarray(ms, np.int64) * MS)

    bag = BagScan(
        "b",
        0,
        900 * MS,
        {"/a": t("/a", [0, 100, 200, 300]), "/tf_static": t("/tf_static", [0, 900])},
    )
    got = trim.expected_counts(bag, 100 * MS, 200 * MS, {"/tf_static"})
    assert got == {"/a": 2, "/tf_static": 2}


def test_resolve_and_default_output(tmp_path):
    f = tmp_path / "rec_0_with_eidos.mcap"
    f.touch()
    assert trim.resolve_mcap_file(str(f)) == f
    assert trim.default_output(f, "/data/bags/trimmed").name == (
        "rec_0_with_eidos_trimmed.mcap"
    )
    d = tmp_path / "bagdir"
    d.mkdir()
    (d / "bagdir_0.mcap").touch()
    (d / "metadata.yaml").touch()
    assert trim.default_output(trim.resolve_mcap_file(str(d)), "o").name == (
        "bagdir_trimmed.mcap"
    )
    (d / "bagdir_1.mcap").touch()
    with pytest.raises(trim.TrimError, match="exactly one"):
        trim.resolve_mcap_file(str(d))
    db3 = tmp_path / "x.db3"
    db3.touch()
    with pytest.raises(trim.TrimError, match="MCAP"):
        trim.resolve_mcap_file(str(db3))


# ---------------------------------------------------------------------------
# End to end: needs rosbag2 (ingest container) and the mcap CLI.
# ---------------------------------------------------------------------------
def _write_bag(path, messages):
    rosbag2_py = pytest.importorskip("rosbag2_py")
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import Imu
    from std_msgs.msg import String

    w = rosbag2_py.SequentialWriter()
    w.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    types = {"/imu": "sensor_msgs/msg/Imu", "/cam": "sensor_msgs/msg/Imu"}
    types |= {"/tf_static": "std_msgs/msg/String", "/once": "std_msgs/msg/String"}
    for i, (name, ty) in enumerate(types.items()):
        w.create_topic(rosbag2_py.TopicMetadata(i, name, ty, "cdr"))
    for topic, t_ms in sorted(messages, key=lambda m: m[1]):
        if types[topic].endswith("Imu"):
            msg = Imu()
            msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(t_ms * MS, 10**9)
            msg.header.frame_id = topic.strip("/")
        else:
            msg = String(data=topic)
        w.write(topic, serialize_message(msg), t_ms * MS)
    del w


@pytest.mark.skipif(shutil.which("mcap") is None, reason="mcap CLI not installed")
def test_trim_end_to_end(tmp_path):
    msgs = [("/imu", t) for t in range(1000, 3001, 100)]  # 1.0-3.0 s
    msgs += [("/cam", t) for t in range(1500, 2601, 100)]  # starts late, ends early
    msgs += [("/tf_static", 1000), ("/tf_static", 2900), ("/once", 1050)]
    _write_bag(tmp_path / "rec", msgs)

    out = tmp_path / "trimmed" / "rec_trimmed.mcap"
    r = trim.trim(
        str(tmp_path / "rec"), str(out), static_topics=["/tf_static", "/once"]
    )
    assert (r.window.start_ns, r.window.end_ns) == (1500 * MS, 2600 * MS)
    assert r.counts["/imu"] == (21, 12)
    assert r.counts["/cam"] == (12, 12)
    assert r.counts["/tf_static"] == (2, 2) and r.counts["/once"] == (1, 1)
    assert out.exists()
    assert not list(out.parent.glob(".*"))  # temp parts cleaned up

    with pytest.raises(trim.TrimError, match="exists"):
        trim.trim(
            str(tmp_path / "rec"), str(out), static_topics=["/tf_static", "/once"]
        )
    with pytest.raises(trim.TrimError, match="no common window"):
        trim.trim(str(tmp_path / "rec"), str(tmp_path / "x.mcap"), static_topics=[])
