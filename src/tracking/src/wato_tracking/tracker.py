"""Tracks over one chunk's detections, with wato_common.tracking.

The same Kalman filter Step F scores motion with, Chen's cost or SimpleTrack's
GIoU, plus the cues tracking adds: strong detections are matched first, a
shared lifted instance id lowers a pair's cost, tracks may coast until the
chunk ends (config `association`), and a low-scored detector-only box never
starts a track (config `strong`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from wato_common.schemas import Box3D
from wato_common.tracking import AssociationGates, MultiObjectTracker
from wato_tracking.config import AssociationConfig
from wato_tracking.detections import DetFrame, Detection

# Frame period assumed for a chunk with a single stepped frame.
_DEFAULT_PERIOD_S = 0.05


@dataclass(frozen=True)
class Observation:
    frame: DetFrame
    det: Detection
    filtered: Box3D  # the Kalman filter's box after this detection
    velocity: tuple[float, float, float]  # the filter's velocity then, m/s


def run(frames: list[DetFrame], cfg: AssociationConfig) -> list[list[Observation]]:
    """Every track's observations in time order, tracks by creation."""
    if not frames:
        return []
    t = np.array([f.t_ns for f in frames], dtype=np.int64)
    period = (
        float(np.median(np.diff(t))) * 1e-9 if len(frames) > 1 else _DEFAULT_PERIOD_S
    )
    tracker = MultiObjectTracker(
        AssociationGates(
            t_d=cfg.max_centre_dist_m,
            t_o=cfg.max_one_minus_iou,
            t_v=cfg.max_one_minus_volume_ratio,
            cost=cfg.cost,
            t_giou=cfg.max_one_minus_giou,
        ),
        n_old=None
        if cfg.max_coast_s is None
        else max(1, math.ceil(cfg.max_coast_s / max(period, 1e-3))),
        key_bonus=cfg.instance_bonus,
        key_opens_shape_gates=cfg.instance_opens_shape_gates,
    )
    for f in frames:
        tracker.step(
            (f.t_ns - frames[0].t_ns) * 1e-9,
            [d.box for d in f.dets],
            strong=[d.strong for d in f.dets],
            keys=[d.key for d in f.dets],
            can_birth=[d.can_birth for d in f.dets],
        )
    return [
        [
            Observation(
                frames[o.frame_idx],
                frames[o.frame_idx].dets[o.det_idx],
                o.filtered,
                o.velocity,
            )
            for o in trk.history
        ]
        for trk in tracker.all_tracks()
    ]
