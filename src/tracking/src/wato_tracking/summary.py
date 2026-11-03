"""What one track is: its class, whether it is parked or moving, and whether
it is kept.

Motion is judged on the whole track (3D Auto Labeling, MS3D), not per cluster
as Step F does: moving iff the Kalman-filtered centre moved at least
moving_displacement_m first → last, or at least fast_displacement_m at an
average of at least fast_speed_mps; "unknown" when the track is too short to
tell. A track is dropped only when it is both short and weak (Offline-Poly).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np

from wato_common.artifacts.tracks import MOVING, PARKED, UNKNOWN, TrackSummary
from wato_tracking.config import FilterConfig, MotionConfig
from wato_tracking.tracker import Observation


def track_class(obs: list[Observation]) -> Optional[str]:
    """Score-weighted vote over the frames' classes; None if none is classed."""
    votes: dict[str, float] = defaultdict(float)
    for o in obs:
        if o.det.proposal.cls is not None:
            votes[o.det.proposal.cls] += max(o.det.proposal.score, 1e-3)
    return max(sorted(votes), key=votes.__getitem__) if votes else None


def net_displacement(filtered_xy: np.ndarray, endpoint_frames: int) -> float:
    """BEV distance between the median of the first and of the last few
    filtered centres (net, so jitter doesn't accumulate)."""
    n = filtered_xy.shape[0]
    if n < 2:
        return 0.0
    k = min(endpoint_frames, max(1, n // 2))
    return float(
        np.linalg.norm(np.median(filtered_xy[-k:], 0) - np.median(filtered_xy[:k], 0))
    )


def motion_state(
    n_frames: int, net_m: float, duration_s: float, cfg: MotionConfig
) -> str:
    if n_frames < cfg.min_frames:
        return UNKNOWN
    if net_m >= cfg.moving_displacement_m:
        return MOVING
    speed = net_m / max(duration_s, 1e-3)
    if net_m >= cfg.fast_displacement_m and speed >= cfg.fast_speed_mps:
        return MOVING
    return PARKED


def summarize(
    track_id: str, chunk_id: str, obs: list[Observation], cfg: MotionConfig
) -> TrackSummary:
    filtered = np.array([[o.filtered.cx, o.filtered.cy] for o in obs])
    measured = np.array([[o.det.proposal.cx, o.det.proposal.cy] for o in obs])
    t = [o.frame.t_ns for o in obs]
    cls = track_class(obs)
    net = net_displacement(filtered, cfg.endpoint_frames)
    var = float(np.mean(np.var(measured, axis=0)))
    return TrackSummary(
        track_id=track_id,
        chunk_id=chunk_id,
        n_frames=len(obs),
        first_timestamp_ns=min(t),
        last_timestamp_ns=max(t),
        motion_state=motion_state(len(obs), net, (max(t) - min(t)) * 1e-9, cfg),
        cls=cls,
        n_strong=sum(o.det.strong for o in obs),
        frac_two_source=float(np.mean([o.det.two_source for o in obs])),
        frac_lidar_mos=float(np.mean([o.det.has_lidar_mos for o in obs])),
        net_displacement_m=net,
        centre_var_m2=var,
        median_points=float(
            np.median([o.det.proposal.lidar_point_count or 0 for o in obs])
        ),
    )


def keep(summary: TrackSummary, cfg: FilterConfig) -> bool:
    return summary.n_frames >= cfg.min_frames or summary.n_strong >= cfg.min_strong
