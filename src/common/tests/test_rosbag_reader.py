"""Tests for rosbag_reader.messages' time window, with a fake rosbag2 reader."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from wato_common.io import rosbag_reader

S = 1_000_000_000


class _FakeReader:
    """Serves (topic, raw, record_ts) in the given order and counts reads."""

    def __init__(self, stamps: list[int]):
        self._msgs = [("/t", f"m{i}".encode(), ts) for i, ts in enumerate(stamps)]
        self.reads = 0

    def set_filter(self, _f):
        pass

    def get_all_topics_and_types(self):
        return [SimpleNamespace(name="/t", type="pkg/msg/T")]

    def has_next(self):
        return self.reads < len(self._msgs)

    def read_next(self):
        self.reads += 1
        return self._msgs[self.reads - 1]


@pytest.fixture
def fake_bag(monkeypatch):
    state = {}

    def open_reader(_bag, storage_id=""):
        return state["reader"]

    monkeypatch.setattr(rosbag_reader, "open_reader", open_reader)
    monkeypatch.setattr(
        rosbag_reader, "_load_rosbag2", lambda: SimpleNamespace(StorageFilter=dict)
    )
    monkeypatch.setattr(
        rosbag_reader,
        "_load_rclpy_serialization",
        lambda: (lambda raw, _cls: raw.decode(), lambda _type: object),
    )

    def load(stamps):
        state["reader"] = _FakeReader(stamps)
        return state["reader"]

    return load


def _read(t_start, t_end):
    with rosbag_reader.messages(
        "bag", topics=["/t"], t_start_ns=t_start, t_end_ns=t_end
    ) as it:
        return [ts for _topic, _msg, ts in it]


def test_window_is_inclusive(fake_bag):
    fake_bag([0, 1 * S, 2 * S, 3 * S])
    assert _read(1 * S, 2 * S) == [1 * S, 2 * S]


def test_stops_reading_once_past_the_window(fake_bag):
    stamps = [i * S // 10 for i in range(1_000)]  # 100 s at 10 Hz
    reader = fake_bag(stamps)
    got = _read(10 * S, 20 * S)
    assert got == [t for t in stamps if 10 * S <= t <= 20 * S]
    # Reads stop at the first message beyond t_end + 1 s, not at the bag end.
    assert reader.reads == stamps.index(21 * S + S // 10) + 1


def test_slightly_out_of_order_message_inside_the_margin_is_kept(fake_bag):
    fake_bag([0, 2 * S + S // 2, 2 * S, 4 * S])
    assert _read(0, 2 * S) == [0, 2 * S]


def test_no_end_reads_the_whole_bag(fake_bag):
    reader = fake_bag([0, 1 * S, 2 * S])
    assert _read(None, None) == [0, 1 * S, 2 * S]
    assert reader.reads == 3
