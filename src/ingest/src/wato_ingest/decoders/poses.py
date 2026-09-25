"""Extract ego pose over time from the configured pose topic.

The pose source is set by `topics.pose`: a nav_msgs/Odometry,
geometry_msgs/PoseStamped or geometry_msgs/PoseWithCovarianceStamped topic
giving the world pose of `ego_frame`.  Whatever the topic, ingest REQUIRES a
dense, smooth pose stream (README "Pose requirements"): every sweep pose and
every deskewed LiDAR point is interpolated between two samples, which is only
correct when those samples are close together.  The README's "How good are
the poses" section lists the sources measured so far; passing these checks is
necessary, not sufficient (they can't see a mis-stamped or attitude-biased
stream).

`extract` enforces the chunk-level part (dense coverage) and raises
PoseRequirementError on a sparse stream.  It also marks every stretch between
two consecutive samples as trusted or not (`interval_drop_reason`: too long,
or a jump); wato_common.pose_lookup reads those marks, so ingest's frame_index
and every downstream component that looks a pose up apply the same rules.

Only an Odometry names the body frame it describes (child_frame_id), so only
an Odometry is checked against `ego_frame`; for the two Pose types the config
is taken on trust.

A pose that exists only on /tf is not read: TF relays commonly re-stamp
transforms with wall-clock time (eidos_transform does), which would desync the
pose from the sweeps it is interpolated for.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from wato_common.artifact_store import ensure_local_dir, local_path, poses_path
from wato_common.geometry import PoseSample, flatten_se3, make_se3
from wato_common.io.parquet_io import write_table
from wato_common.io.rosbag_reader import messages
from wato_common.pose_lookup import interval_drop_reasons
from wato_common.schemas import POSES_SCHEMA, PoseRow
from wato_ingest.config import IngestConfig, PoseRequirements

log = logging.getLogger(__name__)

# A sample whose position is bit-identical to the previous kept sample while
# its own twist says the ego is moving can't be a new measurement — it repeats
# an older fix under a newer stamp.  An INS publishing faster than its position
# updates does exactly this (/novatel/oem7/odom: 100 Hz messages, 50 Hz
# position), and interpolating through the repeat stalls the pose for one
# period then doubles its speed.  A stream with no twist (zero, or a Pose
# message) is never filtered.
_HELD_POSITION_MIN_SPEED_MPS = 0.5

# The bag is windowed by record time, but a pose describes an earlier moment:
# it's recorded after the scan or fix it came from is processed (a LiDAR SLAM
# front end measured 239-337 ms after its stamp).  Reading this much further on
# each side keeps the samples around every sweep and camera frame near the
# chunk edges.  Pose topics are small, so the extra read is cheap.
_PUBLISH_LATENCY_ALLOWANCE_NS = 1_000_000_000


class PoseRequirementError(RuntimeError):
    """The pose topic can't give trustworthy per-sweep / per-point poses."""


@dataclass
class PoseExtractionResult:
    rows_written: int
    output_uri: str
    n_held_dropped: int = 0
    dense_fraction: Optional[float] = None


def _header_ts_ns(header) -> int:
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def extract(
    bag_path: str,
    bag_id: str,
    chunk_id: str,
    *,
    t_start_ns: int,
    t_end_ns: int,
    cfg: IngestConfig,
) -> PoseExtractionResult:
    """Read every pose message on `cfg.topics.pose` around the chunk's
    window and write `poses.parquet`.

    The window is padded on each side by `pose_requirements.max_bracket_ms`
    plus `_PUBLISH_LATENCY_ALLOWANCE_NS`, so a sweep or camera frame at the
    chunk edge still has a sample on both sides of it (poses are never
    extrapolated).  Raises PoseRequirementError — before writing anything — if
    the stream is sparse or its child frame isn't `ego_frame`.
    """
    req = cfg.pose_requirements
    pose_topic = cfg.topics.pose
    pad_ns = int(req.max_bracket_ms * 1e6) + _PUBLISH_LATENCY_ALLOWANCE_NS

    samples: list[tuple[dict, float]] = []
    child_frames: set[str] = set()
    with messages(
        bag_path,
        storage_id=cfg.storage_id,
        topics=[pose_topic],
        t_start_ns=t_start_ns - pad_ns,
        t_end_ns=t_end_ns + pad_ns,
    ) as iterator:
        for _topic, msg, _record_ts_ns in iterator:
            parsed = parse_pose_msg(msg)
            if parsed is None:
                continue
            pose, child, speed = parsed
            if child is not None:
                child_frames.add(child)
            samples.append(
                (_pose_row(bag_id, chunk_id, msg.header, pose, pose_topic), speed)
            )

    if samples and not child_frames:
        log.info(
            "pose topic %s names no child frame (Pose message); taking it as the "
            "pose of ego_frame=%r",
            pose_topic,
            cfg.ego_frame,
        )
    check_child_frame(child_frames, cfg.ego_frame, pose_topic)
    samples = _dedupe_and_sort(samples)
    rows, n_held = _drop_held_positions(samples)
    dense = check_density(
        [r["timestamp_ns"] for r in rows], req, topic=pose_topic, chunk_id=chunk_id
    )
    if n_held:
        log.info(
            "poses %s: dropped %d held-position samples (repeated fix while moving)",
            chunk_id,
            n_held,
        )
    reasons = interval_drop_reasons(
        [_sample(r) for r in rows],
        max_bracket_ns=int(req.max_bracket_ms * 1e6),
        max_speed_mps=req.max_speed_mps,
    )
    for row, reason in zip(rows, reasons):
        row["interval_drop_reason"] = reason

    out_uri = poses_path(bag_id, chunk_id)
    ensure_local_dir(local_path(out_uri).rsplit("/", 1)[0])
    write_table(rows, POSES_SCHEMA, out_uri)
    return PoseExtractionResult(
        rows_written=len(rows),
        output_uri=out_uri,
        n_held_dropped=n_held,
        dense_fraction=dense,
    )


def check_child_frame(child_frames: set[str], ego_frame: str, topic: str) -> None:
    """The pose describes its child frame; extrinsics are resolved relative to
    `ego_frame`.  If they differ, every point is shifted by the transform
    between them (e.g. base_footprint → base_link, 1.76 m on the WATO rig)."""
    named = {c for c in child_frames if c}
    if not named:
        if child_frames:
            log.warning(
                "pose topic %s has an empty child_frame_id; assuming ego_frame=%r",
                topic,
                ego_frame,
            )
        return
    if named != {ego_frame}:
        raise PoseRequirementError(
            f"pose topic {topic} has child_frame_id {sorted(named)} but ego_frame "
            f"is {ego_frame!r}. Extrinsics are resolved relative to ego_frame, so "
            "every point would be offset by the transform between the two frames. "
            "Set ego_frame to the pose's child frame."
        )


def dense_fraction(timestamps_ns: list[int], max_bracket_ms: float) -> float:
    """Fraction of [first, last] sample time lying between consecutive samples
    at most `max_bracket_ms` apart — i.e. the share of the span in which a
    sweep would get a trusted (pose_gap-free) pose."""
    dt = np.diff(np.sort(np.asarray(timestamps_ns, dtype=np.int64)))
    total = int(dt.sum())
    if total == 0:
        return 1.0
    return float(dt[dt <= max_bracket_ms * 1e6].sum() / total)


def check_density(
    timestamps_ns: list[int],
    req: PoseRequirements,
    *,
    topic: str,
    chunk_id: str,
) -> Optional[float]:
    """Dense fraction of the chunk's pose span; raises PoseRequirementError
    below `min_dense_fraction`.

    Fewer than two samples can't be judged — the chunk's sweeps simply get
    valid_pose=False (e.g. SLAM not yet initialised) and quality tags
    POSE_MISSING.  A short dropout inside a dense stream stays a per-sweep
    problem (pose_gap_*) as long as coverage stays above the threshold.
    """
    if len(timestamps_ns) < 2:
        return None
    frac = dense_fraction(timestamps_ns, req.max_bracket_ms)
    if frac < req.min_dense_fraction:
        ts = np.sort(np.asarray(timestamps_ns, dtype=np.int64))
        median_ms = float(np.median(np.diff(ts))) / 1e6
        raise PoseRequirementError(
            f"chunk {chunk_id}: pose topic {topic} is too sparse — only "
            f"{frac:.0%} of its {(ts[-1] - ts[0]) / 1e9:.1f} s span has pose "
            f"samples <= {req.max_bracket_ms:.0f} ms apart (required "
            f">= {req.min_dense_fraction:.0%}; {len(ts)} distinct poses, median "
            f"spacing {median_ms:.0f} ms). Sweep and per-point poses would be "
            "interpolated across the gaps assuming constant velocity. A "
            "keyframe-rate SLAM output or an unconverged INS looks like this. "
            "Use a dense pose source — see the ingest README, 'Pose "
            "requirements'."
        )
    return frac


def parse_pose_msg(msg) -> Optional[tuple[object, Optional[str], float]]:
    """(geometry_msgs/Pose, child frame or None, twist speed m/s) for a
    supported pose message; None for any other type."""
    mt = type(msg).__name__
    if mt == "Odometry":
        v = msg.twist.twist.linear
        speed = float(np.linalg.norm([v.x, v.y, v.z]))
        return msg.pose.pose, msg.child_frame_id, speed
    if mt == "PoseWithCovarianceStamped":
        return msg.pose.pose, None, 0.0
    if mt == "PoseStamped":
        return msg.pose, None, 0.0
    return None


def _pose_row(bag_id: str, chunk_id: str, header, pose, source_topic: str) -> dict:
    T = make_se3(
        np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64),
        (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ),
    )
    return PoseRow(
        bag_id=bag_id,
        chunk_id=chunk_id,
        timestamp_ns=_header_ts_ns(header),
        x=float(T[0, 3]),
        y=float(T[1, 3]),
        z=float(T[2, 3]),
        qx=float(pose.orientation.x),
        qy=float(pose.orientation.y),
        qz=float(pose.orientation.z),
        qw=float(pose.orientation.w),
        world_T_ego_flat=flatten_se3(T),
        source=source_topic,
        valid=True,
    ).model_dump()


def _sample(row: dict) -> PoseSample:
    return PoseSample(
        timestamp_ns=row["timestamp_ns"],
        translation=np.array([row["x"], row["y"], row["z"]], dtype=np.float64),
        quat_xyzw=np.array(
            [row["qx"], row["qy"], row["qz"], row["qw"]], dtype=np.float64
        ),
    )


def _dedupe_and_sort(
    samples: list[tuple[dict, float]],
) -> list[tuple[dict, float]]:
    """Drop duplicate timestamps (keeping the first) and sort ascending."""
    seen: set[int] = set()
    out: list[tuple[dict, float]] = []
    for s in sorted(samples, key=lambda s: s[0]["timestamp_ns"]):
        if s[0]["timestamp_ns"] in seen:
            continue
        seen.add(s[0]["timestamp_ns"])
        out.append(s)
    return out


def _drop_held_positions(
    samples: list[tuple[dict, float]],
) -> tuple[list[dict], int]:
    """Drop samples repeating the previous kept position while moving.

    Returns (kept rows, number dropped).  A stationary ego legitimately holds
    its position, so samples with twist speed <= _HELD_POSITION_MIN_SPEED_MPS
    are always kept.
    """
    kept: list[dict] = []
    dropped = 0
    for row, speed in samples:
        if (
            kept
            and speed > _HELD_POSITION_MIN_SPEED_MPS
            and (row["x"], row["y"], row["z"])
            == (kept[-1]["x"], kept[-1]["y"], kept[-1]["z"])
        ):
            dropped += 1
            continue
        kept.append(row)
    return kept, dropped
