"""Ego pose at any timestamp, looked up from ingest's poses.parquet.

This is the one place world_T_ego is interpolated.  Ingest writes the pose
topic's samples to poses.parquet and, for each sample, records whether the
stretch to the next sample can be interpolated across
(`interval_drop_reason`, computed by `interval_drop_reasons` from ingest's
`pose_requirements`).  Every component then looks the pose up at the
timestamp of its own data: ingest at each LiDAR sweep (frame_index),
lidar_preprocessing at each point, perception_2d and semantic_lifting at each
camera frame.  They all interpolate the same way and trust the same stretches.

Interpolation is linear in translation and SLERP in rotation between the two
samples around the target time, which assumes constant velocity between
them.  There is no extrapolation.  The ingest README's "Pose requirements"
section explains the rules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from wato_common.artifact_store import poses_path
from wato_common.geometry import PoseSample, interpolate_pose
from wato_common.io.parquet_io import read_rows

log = logging.getLogger(__name__)

# Implied speed is measured over at least one LiDAR sweep period.  Pose stamps
# jitter by milliseconds (NovAtel stamps consecutive fixes as little as 0.3 ms
# apart), so a normal 0.15 m step over a sub-millisecond interval would read as
# hundreds of m/s.  Over 50 ms, only a displacement larger than
# max_speed_mps × 50 ms (1.5 m at 30 m/s) can register as a jump.
_MIN_SPEED_BASE_NS = 50_000_000


@dataclass
class InterpolatedPose:
    target_timestamp_ns: int
    world_T_ego: np.ndarray  # 4x4
    interp_error_ns: float  # gap to the nearest sample
    pose_timestamp_ns: int  # nearest source sample
    valid: bool
    drop_reason: Optional[str] = None  # set iff valid is False


def interval_drop_reasons(
    samples: Sequence[PoseSample],
    *,
    max_bracket_ns: int,
    max_speed_mps: float,
) -> list[Optional[str]]:
    """Why each stretch between consecutive samples can't be interpolated.

    Element i describes the stretch from ``samples[i]`` to ``samples[i + 1]``:
      - samples more than ``max_bracket_ns`` apart → "pose_gap_<ms>ms"
        (constant velocity can't be assumed across it)
      - the ego would have moved faster than ``max_speed_mps`` → "pose_jump_<mps>mps"
        (a discontinuity such as a loop closure or INS reset, not motion)
      - otherwise None (trusted).
    The last element is always None: there is no stretch after the last sample.
    ``samples`` must be sorted by timestamp.
    """
    reasons: list[Optional[str]] = []
    for s0, s1 in zip(samples, samples[1:]):
        span_ns = s1.timestamp_ns - s0.timestamp_ns
        if span_ns > max_bracket_ns:
            reasons.append(f"pose_gap_{span_ns / 1e6:.0f}ms")
            continue
        step_m = float(np.linalg.norm(s1.translation - s0.translation))
        speed = step_m / (max(span_ns, _MIN_SPEED_BASE_NS) / 1e9)
        reasons.append(f"pose_jump_{speed:.0f}mps" if speed > max_speed_mps else None)
    if samples:
        reasons.append(None)
    return reasons


class PoseLookup:
    """Pose samples for one chunk, queried at arbitrary timestamps.

    ``interval_reasons[i]`` says whether the stretch from sample i to sample
    i + 1 is trusted (None) or why not; see ``interval_drop_reasons``.
    Omitted means every stretch is trusted.
    """

    def __init__(
        self,
        samples: Sequence[PoseSample],
        interval_reasons: Optional[Sequence[Optional[str]]] = None,
    ) -> None:
        order = sorted(range(len(samples)), key=lambda i: samples[i].timestamp_ns)
        self._samples = [samples[i] for i in order]
        if interval_reasons is None:
            self._reasons: list[Optional[str]] = [None] * len(order)
        else:
            if len(interval_reasons) != len(samples):
                raise ValueError(
                    f"{len(interval_reasons)} interval reasons for {len(samples)} samples"
                )
            self._reasons = [interval_reasons[i] for i in order]
        self._ts = np.fromiter(
            (s.timestamp_ns for s in self._samples),
            dtype=np.int64,
            count=len(self._samples),
        )

    @classmethod
    def load(cls, bag_id: str, chunk_id: str) -> "PoseLookup":
        """Read one chunk's poses.parquet (rows with valid=False are skipped)."""
        rows = [
            r for r in read_rows(poses_path(bag_id, chunk_id)) if r.get("valid", True)
        ]
        samples = [
            PoseSample(
                timestamp_ns=int(r["timestamp_ns"]),
                translation=np.array([r["x"], r["y"], r["z"]], dtype=np.float64),
                quat_xyzw=np.array(
                    [r["qx"], r["qy"], r["qz"], r["qw"]], dtype=np.float64
                ),
            )
            for r in rows
        ]
        if rows and "interval_drop_reason" not in rows[0]:
            log.warning(
                "chunk %s: poses.parquet has no interval_drop_reason column (written "
                "before ingest validated pose intervals) — trusting every interval; "
                "re-run ingest to apply pose_requirements",
                chunk_id,
            )
            return cls(samples)
        return cls(samples, [r.get("interval_drop_reason") for r in rows])

    @property
    def samples(self) -> list[PoseSample]:
        """Samples sorted by timestamp."""
        return self._samples

    def __len__(self) -> int:
        return len(self._samples)

    def at(self, target_ts_ns: int) -> InterpolatedPose:
        """World pose of the ego at ``target_ts_ns``.

        valid=False (with ``drop_reason``) when:
          - there are no samples                    → "no_pose_samples"
          - the target is outside the sampled span  → "outside_pose_span"
          - the stretch around the target is untrusted → that stretch's reason
            ("pose_gap_<ms>ms" / "pose_jump_<mps>mps")
        A target exactly on a sample is valid: nothing is interpolated.
        """
        target_ts_ns = int(target_ts_ns)
        if not self._samples:
            return InterpolatedPose(
                target_ts_ns, np.eye(4), float("inf"), 0, False, "no_pose_samples"
            )
        idx = int(np.searchsorted(self._ts, target_ts_ns))
        if idx < len(self._ts) and self._ts[idx] == target_ts_ns:
            s = self._samples[idx]
            T, _ = interpolate_pose([s], target_ts_ns)
            return InterpolatedPose(target_ts_ns, T, 0.0, s.timestamp_ns, True)
        if idx == 0 or idx == len(self._ts):
            s = self._samples[0] if idx == 0 else self._samples[-1]
            T, err = interpolate_pose([s], target_ts_ns)
            return InterpolatedPose(
                target_ts_ns, T, err, s.timestamp_ns, False, "outside_pose_span"
            )
        s0, s1 = self._samples[idx - 1], self._samples[idx]
        T, err = interpolate_pose([s0, s1], target_ts_ns)
        nearest = (
            s0
            if target_ts_ns - s0.timestamp_ns <= s1.timestamp_ns - target_ts_ns
            else s1
        )
        reason = self._reasons[idx - 1]
        return InterpolatedPose(
            target_timestamp_ns=target_ts_ns,
            world_T_ego=T,
            interp_error_ns=err,
            pose_timestamp_ns=nearest.timestamp_ns,
            valid=reason is None,
            drop_reason=reason,
        )
