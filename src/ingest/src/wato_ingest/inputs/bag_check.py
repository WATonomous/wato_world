"""What a bag is missing, and the window in which every topic is recording.

`scan` makes one pass over the bag and records, per topic, every message's log
(record) time and — where the message type is installed and starts with a
std_msgs/Header — its header stamp and frame_id, read straight from the CDR
bytes so multi-megabyte images are never deserialized.  Everything after it is
a pure function over the resulting `BagScan`, unit-tested without ROS.

Used by `validate-bag` (the report) and `trim-bag` (the window).  Times are log
times unless a name says header; gaps, coverage and the window are all judged
on log time, which every topic has and which is what an MCAP cut operates on.

"Static" topics (named per run with --static-topic: /tf_static, one-shot
topics) can't span a window, so they're excluded from gap, coverage and window
judgments and listed separately.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from wato_ingest.config import IngestConfig
from wato_ingest.inputs import topics as topic_validation
from wato_ingest.inputs.calibration import _resolve_chain, _transform_se3

IMAGE_TYPES = topic_validation.IMAGE_TYPES
CAMERA_INFO_TYPE = "sensor_msgs/msg/CameraInfo"
# Types whose header.frame_id is a sensor frame that needs an extrinsic.
SENSOR_FRAME_TYPES = IMAGE_TYPES | {CAMERA_INFO_TYPE, "sensor_msgs/msg/PointCloud2"}


@dataclass
class TopicScan:
    name: str
    type: str
    log_ns: np.ndarray  # int64, ascending
    # int64, aligned with log_ns; None when the type has no readable header.
    header_ns: np.ndarray | None = None
    frame_ids: set[str] = field(default_factory=set)
    # Why header_ns is None ("" when it isn't).
    header_note: str = ""


@dataclass
class BagScan:
    path: str
    start_ns: int  # earliest log time over every topic
    end_ns: int  # latest log time over every topic
    topics: dict[str, TopicScan]
    # (parent, child) -> 4x4 from the profile's tf_static topic; empty if absent.
    tf_static: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------


def parse_cdr_header(raw: bytes) -> tuple[int, str]:
    """(stamp_ns, frame_id) of a CDR message whose first field is std_msgs/Header.

    Layout: 4-byte encapsulation header (0x00 0x01 = little-endian CDR,
    0x00 0x00 = big-endian), then int32 sec, uint32 nanosec, uint32 string
    length (counting the trailing NUL), then the frame_id bytes.
    """
    if len(raw) < 16 or raw[0] != 0 or raw[1] not in (0, 1):
        raise ValueError(f"not a plain CDR message (encapsulation {raw[:4]!r})")
    fmt = "<iII" if raw[1] == 1 else ">iII"
    sec, nanosec, n = struct.unpack_from(fmt, raw, 4)
    frame_id = raw[16 : 16 + n - 1].decode("utf-8") if n else ""
    return sec * 1_000_000_000 + nanosec, frame_id


def _header_support(type_str: str) -> str:
    """ "" if the type is installed and starts with a std_msgs/Header, else why not."""
    from rosidl_runtime_py.utilities import get_message

    try:
        cls = get_message(type_str)
    except (ImportError, AttributeError, ValueError):
        return "type not installed in this image"
    first = next(iter(cls.get_fields_and_field_types().items()), None)
    if first is None or first[1] != "std_msgs/Header":
        return "no header"
    return ""


# ---------------------------------------------------------------------------
# Bag-reading pass.  Uses the rosbag2 reader; not unit-testable directly.
# ---------------------------------------------------------------------------


def scan(
    bag_path: str,
    *,
    storage_id: str = "",
    headers: bool = True,
    tf_static_topic: str | None = None,
) -> BagScan:
    """One pass over every message of the bag.

    `headers=False` records log times only (all `trim-bag` needs).
    `tf_static_topic` is deserialized to collect static transforms.
    """
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    from wato_common.io.rosbag_reader import open_reader

    reader = open_reader(bag_path, storage_id=storage_id)
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    notes = {
        t: (_header_support(ty) if headers else "not read") for t, ty in types.items()
    }
    log: dict[str, list[int]] = {t: [] for t in types}
    hdr: dict[str, list[int]] = {t: [] for t in types if notes[t] == ""}
    frames: dict[str, set[str]] = {t: set() for t in types}
    tf_cls = None
    if tf_static_topic is not None and tf_static_topic in types:
        tf_cls = get_message(types[tf_static_topic])
    tf_static: dict[tuple[str, str], np.ndarray] = {}

    while reader.has_next():
        topic, raw, ts = reader.read_next()
        log[topic].append(int(ts))
        if topic in hdr:
            stamp, frame_id = parse_cdr_header(raw)
            hdr[topic].append(stamp)
            frames[topic].add(frame_id)
        if tf_cls is not None and topic == tf_static_topic:
            for t in deserialize_message(raw, tf_cls).transforms:
                key = (t.header.frame_id, t.child_frame_id)
                tf_static.setdefault(key, _transform_se3(t.transform))

    out: dict[str, TopicScan] = {}
    for name, ty in types.items():
        log_ns = np.asarray(log[name], dtype=np.int64)
        order = np.argsort(log_ns, kind="stable")
        header_ns = None
        if name in hdr:
            header_ns = np.asarray(hdr[name], dtype=np.int64)[order]
        out[name] = TopicScan(
            name=name,
            type=topic_validation.normalize_type(ty),
            log_ns=log_ns[order],
            header_ns=header_ns,
            frame_ids=frames[name],
            header_note=notes[name],
        )
    nonempty = [t.log_ns for t in out.values() if len(t.log_ns)]
    if not nonempty:
        raise ValueError(f"{bag_path} has no messages")
    return BagScan(
        path=bag_path,
        start_ns=int(min(a[0] for a in nonempty)),
        end_ns=int(max(a[-1] for a in nonempty)),
        topics=out,
        tf_static=tf_static,
    )


# ---------------------------------------------------------------------------
# Per-topic coverage and gaps
# ---------------------------------------------------------------------------


@dataclass
class Gap:
    start_ns: int  # log time of the message before the gap
    end_ns: int  # log time of the message after it

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


@dataclass
class TopicReport:
    topic: str
    type: str
    count: int
    first_ns: int | None
    last_ns: int | None
    static: bool
    median_period_ns: float | None = None
    # Time from the bag's first / to its last message with nothing on this
    # topic; reported only when longer than gap_factor x its median period.
    missing_start_ns: int = 0
    missing_end_ns: int = 0
    gaps: list[Gap] = field(default_factory=list)
    # Header stamps (None + header_note when the type has no readable header).
    header_first_ns: int | None = None
    header_last_ns: int | None = None
    skew_median_ns: float | None = None  # median(header - log)
    skew_max_abs_ns: int | None = None
    header_note: str = ""


def topic_report(
    t: TopicScan, bag: BagScan, gap_factor: float, static: bool
) -> TopicReport:
    n = len(t.log_ns)
    r = TopicReport(
        topic=t.name,
        type=t.type,
        count=n,
        first_ns=int(t.log_ns[0]) if n else None,
        last_ns=int(t.log_ns[-1]) if n else None,
        static=static,
        header_note=t.header_note,
    )
    if t.header_ns is not None and n:
        skew = t.header_ns - t.log_ns
        r.header_first_ns = int(t.header_ns.min())
        r.header_last_ns = int(t.header_ns.max())
        r.skew_median_ns = float(np.median(skew))
        r.skew_max_abs_ns = int(np.abs(skew).max())
    if static or n < 2:
        return r
    dt = np.diff(t.log_ns)
    period = float(np.median(dt))
    r.median_period_ns = period
    limit = gap_factor * period
    r.gaps = [
        Gap(int(t.log_ns[i]), int(t.log_ns[i + 1])) for i in np.flatnonzero(dt > limit)
    ]
    lead = int(t.log_ns[0]) - bag.start_ns
    tail = bag.end_ns - int(t.log_ns[-1])
    r.missing_start_ns = lead if lead > limit else 0
    r.missing_end_ns = tail if tail > limit else 0
    return r


# ---------------------------------------------------------------------------
# Common window
# ---------------------------------------------------------------------------


@dataclass
class Window:
    start_ns: int | None  # None when the topics never all record at once
    end_ns: int | None
    # The non-static topic(s) that start last / end first — they set the cut.
    started_last: list[str]
    ended_first: list[str]
    reason: str = ""


def common_window(bag: BagScan, static_topics: set[str]) -> Window:
    """[latest first message, earliest last message] over the non-static topics."""
    live = {
        name: t
        for name, t in bag.topics.items()
        if name not in static_topics and len(t.log_ns)
    }
    empty = sorted(
        n for n, t in bag.topics.items() if n not in static_topics and not len(t.log_ns)
    )
    if empty:
        return Window(None, None, [], [], reason=f"topics with no messages: {empty}")
    if not live:
        return Window(None, None, [], [], reason="every topic is static")
    start = max(int(t.log_ns[0]) for t in live.values())
    end = min(int(t.log_ns[-1]) for t in live.values())
    started_last = sorted(n for n, t in live.items() if int(t.log_ns[0]) == start)
    ended_first = sorted(n for n, t in live.items() if int(t.log_ns[-1]) == end)
    if start >= end:
        return Window(
            None,
            None,
            started_last,
            ended_first,
            reason=(
                f"{started_last} start after {ended_first} end — no moment has "
                "every topic recording (pass one-shot topics with --static-topic)"
            ),
        )
    return Window(start, end, started_last, ended_first)


# ---------------------------------------------------------------------------
# Image <-> camera_info pairing
# ---------------------------------------------------------------------------


@dataclass
class CameraPairing:
    image_topic: str
    info_topic: str
    images: int
    infos: int
    # Header stamps on one stream with no identical stamp on the other;
    # None when either stream's headers couldn't be read.
    images_without_info: int | None
    infos_without_image: int | None


def camera_pairings(bag: BagScan) -> tuple[list[CameraPairing], list[str]]:
    """Pair each image topic with the camera_info in its namespace.

    Returns the pairings and the image topics that have no camera_info sibling.
    """
    infos = {
        name.rsplit("/", 1)[0]: t
        for name, t in bag.topics.items()
        if t.type == CAMERA_INFO_TYPE
    }
    pairs: list[CameraPairing] = []
    unpaired: list[str] = []
    for name, img in sorted(bag.topics.items()):
        if img.type not in IMAGE_TYPES:
            continue
        ns = name.rsplit("/", 1)[0]
        info = infos.get(ns)
        if info is None:
            unpaired.append(name)
            continue
        if img.header_ns is not None and info.header_ns is not None:
            img_stamps, info_stamps = (
                set(img.header_ns.tolist()),
                set(info.header_ns.tolist()),
            )
            no_info = len(img_stamps - info_stamps)
            no_image = len(info_stamps - img_stamps)
        else:
            no_info = no_image = None
        pairs.append(
            CameraPairing(
                image_topic=name,
                info_topic=info.name,
                images=len(img.log_ns),
                infos=len(info.log_ns),
                images_without_info=no_info,
                infos_without_image=no_image,
            )
        )
    return pairs, unpaired


# ---------------------------------------------------------------------------
# Extrinsics
# ---------------------------------------------------------------------------


def unresolved_frames(bag: BagScan, ego_frame: str) -> dict[str, list[str]]:
    """Sensor-type frame_ids with no tf_static chain from ego_frame -> {frame: [topics]}."""
    by_frame: dict[str, list[str]] = {}
    for name, t in sorted(bag.topics.items()):
        if t.type in SENSOR_FRAME_TYPES:
            for fid in t.frame_ids:
                by_frame.setdefault(fid, []).append(name)
    return {
        fid: topics
        for fid, topics in sorted(by_frame.items())
        if _resolve_chain(bag.tf_static, ego_frame, fid) is None
    }


def profile_sensor_topics(cfg: IngestConfig) -> set[str]:
    """Topics whose frame ingest needs an extrinsic for: cameras and LiDARs."""
    out = set(cfg.topics.lidars.values())
    for c in cfg.topics.cameras.values():
        out.add(c.image)
        if c.info is not None:
            out.add(c.info)
    return out


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------


@dataclass
class BagReport:
    bag: str
    start_ns: int
    end_ns: int
    gap_factor: float
    topics: list[TopicReport]
    window: Window
    pairings: list[CameraPairing]
    images_without_info_topic: list[str]
    profile: topic_validation.TopicValidationResult
    ego_frame: str
    tf_static_topic: str | None
    # Frames with no tf_static chain from ego_frame, split by whether a topic
    # the ingest profile projects uses them (a failure) or only other topics
    # do (e.g. world-frame clouds in `map` — reported, not a failure).
    unresolved_frames: dict[str, list[str]]
    unresolved_other_frames: dict[str, list[str]]
    # Readable sensor topics whose header couldn't be read, so their frame was
    # not checked against tf_static.
    frames_unchecked: list[str]

    @property
    def failures(self) -> list[str]:
        """Problems that stop ingest or leave no usable window."""
        out = []
        if not self.profile.ok:
            out.append(f"ingest profile: {self.profile.describe()}")
        if self.unresolved_frames:
            out.append(
                f"no tf_static chain from {self.ego_frame!r} to "
                f"{sorted(self.unresolved_frames)}"
            )
        if self.window.start_ns is None:
            out.append(f"no common window: {self.window.reason}")
        return out


def build_report(
    bag: BagScan,
    cfg: IngestConfig,
    *,
    static_topics: set[str],
    gap_factor: float,
) -> BagReport:
    unknown = sorted(static_topics - set(bag.topics))
    if unknown:
        raise ValueError(f"--static-topic not in the bag: {unknown}")
    pairings, unpaired = camera_pairings(bag)
    needed = profile_sensor_topics(cfg)
    unresolved = unresolved_frames(bag, cfg.ego_frame)
    return BagReport(
        bag=bag.path,
        start_ns=bag.start_ns,
        end_ns=bag.end_ns,
        gap_factor=gap_factor,
        topics=[
            topic_report(t, bag, gap_factor, static=t.name in static_topics)
            for t in sorted(bag.topics.values(), key=lambda t: t.name)
        ],
        window=common_window(bag, static_topics),
        pairings=pairings,
        images_without_info_topic=unpaired,
        profile=topic_validation.validate(
            {n: t.type for n, t in bag.topics.items()}, cfg
        ),
        ego_frame=cfg.ego_frame,
        tf_static_topic=cfg.topics.tf_static,
        unresolved_frames={
            f: ts for f, ts in unresolved.items() if needed.intersection(ts)
        },
        unresolved_other_frames={
            f: ts for f, ts in unresolved.items() if not needed.intersection(ts)
        },
        frames_unchecked=sorted(
            n
            for n, t in bag.topics.items()
            if t.type in SENSOR_FRAME_TYPES and t.header_ns is None and len(t.log_ns)
        ),
    )


def _s(ns: float | int | None, origin: int = 0) -> str:
    return "-" if ns is None else f"{(ns - origin) / 1e9:.3f}"


def render(r: BagReport) -> str:
    """Human-readable report.  Times are seconds from the bag's first message."""
    o = r.start_ns
    lines = [
        f"bag: {r.bag}",
        f"log span: {(r.end_ns - r.start_ns) / 1e9:.3f} s  "
        f"(gap = more than {r.gap_factor:g} x the topic's median period)",
        "",
        "COVERAGE (log time, s from bag start)",
        f"  {'topic':<50} {'type':<34} {'msgs':>6} {'first':>8} {'last':>8} "
        f"{'Hz':>7} {'miss@start':>10} {'miss@end':>9} {'gaps':>5} {'gap_s':>7} {'max_gap':>7}",
    ]
    for t in r.topics:
        hz = (
            f"{1e9 / t.median_period_ns:.2f}"
            if t.median_period_ns
            else ("static" if t.static else "-")
        )
        total_gap = sum(g.duration_ns for g in t.gaps)
        max_gap = max((g.duration_ns for g in t.gaps), default=0)
        lines.append(
            f"  {t.topic:<50} {t.type:<34} {t.count:>6} {_s(t.first_ns, o):>8} "
            f"{_s(t.last_ns, o):>8} {hz:>7} "
            f"{_s(t.missing_start_ns) if t.missing_start_ns else '':>10} "
            f"{_s(t.missing_end_ns) if t.missing_end_ns else '':>9} "
            f"{len(t.gaps) or '':>5} {_s(total_gap) if t.gaps else '':>7} "
            f"{_s(max_gap) if t.gaps else '':>7}"
        )

    lines += ["", "GAPS (s from bag start: from -> to, duration)"]
    any_gap = False
    for t in r.topics:
        if not t.gaps:
            continue
        any_gap = True
        shown = sorted(t.gaps, key=lambda g: -g.duration_ns)[:5]
        more = f"  (+{len(t.gaps) - 5} shorter)" if len(t.gaps) > 5 else ""
        lines.append(
            f"  {t.topic}: "
            + ", ".join(
                f"{_s(g.start_ns, o)}->{_s(g.end_ns, o)} ({_s(g.duration_ns)})"
                for g in sorted(shown, key=lambda g: g.start_ns)
            )
            + more
        )
    if not any_gap:
        lines.append("  none")

    w = r.window
    lines += ["", "COMMON WINDOW (every non-static topic recording)"]
    if w.start_ns is None:
        lines.append(f"  NONE: {w.reason}")
    else:
        lines += [
            f"  {_s(w.start_ns, o)} -> {_s(w.end_ns, o)} s "
            f"({(w.end_ns - w.start_ns) / 1e9:.3f} of "
            f"{(r.end_ns - r.start_ns) / 1e9:.3f} s)",
            f"  start set by: {', '.join(w.started_last)}",
            f"  end set by:   {', '.join(w.ended_first)}",
        ]
    static = [t.topic for t in r.topics if t.static]
    lines.append(f"  static (excluded, always kept by trim-bag): {static or 'none'}")

    lines += [
        "",
        "HEADER vs LOG (header - log, ms)",
        f"  {'topic':<50} {'hdr first':>9} {'hdr last':>9} {'median':>9} {'max |.|':>9}",
    ]
    for t in r.topics:
        if t.skew_median_ns is None:
            lines.append(f"  {t.topic:<50} {t.header_note or 'no messages'}")
            continue
        lines.append(
            f"  {t.topic:<50} {_s(t.header_first_ns, o):>9} {_s(t.header_last_ns, o):>9} "
            f"{t.skew_median_ns / 1e6:>9.1f} {t.skew_max_abs_ns / 1e6:>9.1f}"
        )

    lines += [
        "",
        "IMAGE <-> CAMERA_INFO (by header stamp)",
        f"  {'image topic':<50} {'images':>6} {'infos':>6} {'img w/o info':>12} {'info w/o img':>12}",
    ]
    for p in r.pairings:
        lines.append(
            f"  {p.image_topic:<50} {p.images:>6} {p.infos:>6} "
            f"{'-' if p.images_without_info is None else p.images_without_info:>12} "
            f"{'-' if p.infos_without_image is None else p.infos_without_image:>12}"
        )
    for name in r.images_without_info_topic:
        lines.append(f"  {name:<50} no camera_info topic in its namespace")
    if not r.pairings and not r.images_without_info_topic:
        lines.append("  no image topics")

    lines += ["", f"EXTRINSICS (tf_static {r.tf_static_topic} from {r.ego_frame!r})"]
    if r.unresolved_frames:
        for fid, topics in r.unresolved_frames.items():
            lines.append(f"  MISSING {fid}  (used by {', '.join(topics)})")
    else:
        lines.append("  every camera / LiDAR frame of the ingest profile resolves")
    for fid, topics in r.unresolved_other_frames.items():
        lines.append(
            f"  no chain to {fid}  (not in the profile; used by {', '.join(topics)})"
        )
    if r.frames_unchecked:
        lines.append(f"  not checked (header unreadable): {r.frames_unchecked}")

    lines += ["", "INGEST PROFILE"]
    lines.append("  ok" if r.profile.ok else f"  {r.profile.describe()}")

    lines += ["", "RESULT"]
    lines += [f"  FAIL {f}" for f in r.failures] or ["  ok"]
    return "\n".join(lines)
