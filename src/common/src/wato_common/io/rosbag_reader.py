"""Lazy rosbag2 reader.

ROS deps (`rosbag2_py`, `rclpy`, `rosidl_runtime_py`) are imported on first
use so this module can be imported on hosts without ROS installed (for tests
that exercise everything else).  Anything that touches the bag itself must be
called inside a container with ROS 2 Jazzy on PYTHONPATH.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass
class TopicInfo:
    name: str
    type: str
    serialization_format: str
    message_count: int = 0


@dataclass
class BagSummary:
    storage_path: str
    storage_id: str
    topics: list[TopicInfo]
    duration_ns: int
    starting_time_ns: int
    message_count: int


def _load_rosbag2():
    try:
        import rosbag2_py  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "rosbag2_py is not importable.  Run inside the ingest "
            "container (which has ROS 2 Jazzy on PYTHONPATH)."
        ) from e
    return rosbag2_py


def _load_rclpy_serialization():
    try:
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as e:
        raise RuntimeError(
            "rclpy.serialization / rosidl_runtime_py not importable.  Run "
            "inside the ingest container."
        ) from e
    return deserialize_message, get_message


def open_reader(bag_path: str, storage_id: str = "sqlite3"):
    """Open a rosbag2 reader.  Returns a SequentialReader configured for the bag."""
    rosbag2_py = _load_rosbag2()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    return reader


def summarize(bag_path: str, storage_id: str = "sqlite3") -> BagSummary:
    """Read the bag's metadata (topics, durations, message counts) without scanning."""
    rosbag2_py = _load_rosbag2()
    info = rosbag2_py.Info().read_metadata(bag_path, storage_id)
    topics = [
        TopicInfo(
            name=t.topic_metadata.name,
            type=t.topic_metadata.type,
            serialization_format=t.topic_metadata.serialization_format,
            message_count=int(getattr(t, "message_count", 0)),
        )
        for t in info.topics_with_message_count
    ]
    return BagSummary(
        storage_path=bag_path,
        storage_id=storage_id,
        topics=topics,
        duration_ns=int(info.duration.nanoseconds),
        starting_time_ns=int(info.starting_time.nanoseconds),
        message_count=int(info.message_count),
    )


@contextmanager
def messages(
    bag_path: str,
    *,
    storage_id: str = "sqlite3",
    topics: list[str] | None = None,
    t_start_ns: int | None = None,
    t_end_ns: int | None = None,
) -> Iterator[Iterator[tuple[str, object, int]]]:
    """Context manager yielding (topic, deserialized_msg, timestamp_ns) tuples.

    Filters by topic and time range.  Looks up the message type per topic from
    the bag's metadata, so the caller does not need to know types up front.
    """
    rosbag2_py = _load_rosbag2()
    deserialize_message, get_message = _load_rclpy_serialization()

    reader = open_reader(bag_path, storage_id=storage_id)

    if topics:
        f = rosbag2_py.StorageFilter(topics=topics)
        reader.set_filter(f)

    type_by_topic = {t.name: t.type for t in reader.get_all_topics_and_types()}
    msg_class_cache: dict[str, type] = {}

    def _iter():
        while reader.has_next():
            topic, raw, ts = reader.read_next()
            if t_start_ns is not None and ts < t_start_ns:
                continue
            if t_end_ns is not None and ts > t_end_ns:
                continue
            cls = msg_class_cache.get(topic)
            if cls is None:
                cls = get_message(type_by_topic[topic])
                msg_class_cache[topic] = cls
            yield topic, deserialize_message(raw, cls), int(ts)

    # Reader is held by the _iter closure; CPython refcounting releases it
    # when the contextmanager's frame and the closure both go out of scope.
    yield _iter()
