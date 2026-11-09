"""Re-linking a chunk's tracks after the forward pass (Offline-Poly's STWO).

A forward tracker splits one object into several tracks wherever no box fit
its prediction for a while: a gap, a face change the gates refused, a turn
the constant-velocity prediction missed. Offline, every track's whole life is
known, so two tracks that never exist at once can be tested as one object.

Offline-Poly (Li et al., arXiv 2602.13772 §IV-C, "STWO"), frame by frame:

1. Each track without an observation at frame t gets a predicted state:
   forward from its last observation after it ends, backward from its first
   before it starts, and the mean of both inside a gap of its own.
   "Predictions exceeding 1 second are discarded" (horizon_s).
2. The cost of two tracks at t is 1 − IoU of their states there (Eq. 3);
   pairs whose lives overlap in time, and pairs without a state at t, are
   invalid.
3. The valid pairs are matched one to one by the blossom algorithm (a
   general-graph matching: a track can be the earlier or the later of a
   pair), keeping matches with cost below θ_blo (max_cost, "around 0.9",
   §V-E). Each match is fused into one track, and the procedure is "applied
   iteratively".

The paper leaves open how frames combine and when iteration stops. Here each
sweep visits the frames in time order and fuses a pair at the first frame it
matches; a track fused in a sweep waits for the next sweep, and sweeps repeat
until one fuses nothing. Predictions use the tracker's constant-velocity
model: forward with the Kalman filter's velocity at the observation, backward
with that of the same filter run over the track in reverse (the forward
filter knows nothing of the velocity at a track's first box).

Two cues the paper doesn't use, as gates rather than costs: tracks whose
classes (score-weighted votes) are both known and differ are never fused
(class_consistent, as Offline-Poly's multi-tracker module and Poly-MOT keep
categories apart); and two tracks that share a lifted instance (SAM2 followed
one object through both) may be fused whatever their IoU, when their states
at t are within the association's distance gate (shared_instance), as the
per-frame association lets a shared instance open its shape gates. A pair
admitted that way costs 0. Only observed frames are output, as before: a
fused track's gaps are not filled in.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np

from wato_common.schemas import Box3D
from wato_common.tracking import (
    BoxKalmanFilter,
    giou_3d,
    iou_3d,
    iou_bev,
    wrap_heading_half_pi,
)
from wato_tracking.config import RelinkConfig
from wato_tracking.detections import DetFrame
from wato_tracking.summary import track_class
from wato_tracking.tracker import Observation

log = logging.getLogger(__name__)

_SIMILARITY = {"iou_bev": iou_bev, "iou_3d": iou_3d, "giou_3d": giou_3d}


@dataclass
class _Tracklet:
    """A track as STWO sees it: its observations' times (s), filtered boxes
    and real-time velocities forward and backward of each."""

    order: int  # the earliest original track's index, to keep naming stable
    obs: list[Observation]
    t: np.ndarray
    boxes: list[Box3D]
    v_fwd: list[np.ndarray]
    v_bwd: list[np.ndarray]
    cls: Optional[str] = None
    keys: frozenset = field(default_factory=frozenset)

    @property
    def first(self) -> float:
        return float(self.t[0])

    @property
    def last(self) -> float:
        return float(self.t[-1])

    def state(self, t: float, horizon_s: float) -> Optional[Box3D]:
        """Observed or predicted box at t; None past the horizon."""
        k = int(np.searchsorted(self.t, t))
        if k < len(self.t) and self.t[k] == t:
            return self.boxes[k]
        fwd = bwd = None
        if k > 0 and t - self.t[k - 1] <= horizon_s:
            fwd = _moved(self.boxes[k - 1], self.v_fwd[k - 1], t - self.t[k - 1])
        if k < len(self.t) and self.t[k] - t <= horizon_s:
            bwd = _moved(self.boxes[k], self.v_bwd[k], t - self.t[k])
        if fwd is not None and bwd is not None:
            return _mean(fwd, bwd)
        return fwd if fwd is not None else bwd


def _moved(box: Box3D, v: np.ndarray, dt: float) -> Box3D:
    return Box3D(
        cx=box.cx + float(v[0]) * dt,
        cy=box.cy + float(v[1]) * dt,
        cz=box.cz + float(v[2]) * dt,
        w=box.w,
        l=box.l,
        h=box.h,
        heading=box.heading,
    )


def _mean(a: Box3D, b: Box3D) -> Box3D:
    """Mean box; the undirected heading is averaged on the doubled angle."""
    heading = 0.5 * math.atan2(
        math.sin(2 * a.heading) + math.sin(2 * b.heading),
        math.cos(2 * a.heading) + math.cos(2 * b.heading),
    )
    return Box3D(
        cx=(a.cx + b.cx) / 2,
        cy=(a.cy + b.cy) / 2,
        cz=(a.cz + b.cz) / 2,
        w=(a.w + b.w) / 2,
        l=(a.l + b.l) / 2,
        h=(a.h + b.h) / 2,
        heading=wrap_heading_half_pi(heading),
    )


def _backward_velocities(t: np.ndarray, measured: list[Box3D]) -> list[np.ndarray]:
    """Per observation, the real-time velocity of the tracker's Kalman filter
    run over the track from its last box to its first."""
    kf = BoxKalmanFilter(measured[-1])
    out = [np.zeros(3)] * len(measured)
    for k in range(len(measured) - 2, -1, -1):
        kf.predict(t[k + 1] - t[k])
        kf.update(measured[k])
        out[k] = -kf.velocity
    return out


def _tracklet(order: int, obs: list[Observation], t0_ns: int) -> _Tracklet:
    t = np.array([(o.frame.t_ns - t0_ns) * 1e-9 for o in obs])
    return _Tracklet(
        order=order,
        obs=list(obs),
        t=t,
        boxes=[o.filtered for o in obs],
        v_fwd=[np.asarray(o.velocity, dtype=float) for o in obs],
        v_bwd=_backward_velocities(t, [o.det.box for o in obs]),
        cls=track_class(obs),
        keys=frozenset(o.det.key for o in obs if o.det.key is not None),
    )


def _fuse(a: _Tracklet, b: _Tracklet) -> _Tracklet:
    first, second = (a, b) if a.last < b.first else (b, a)
    obs = first.obs + second.obs
    return _Tracklet(
        order=min(a.order, b.order),
        obs=obs,
        t=np.concatenate([first.t, second.t]),
        boxes=first.boxes + second.boxes,
        v_fwd=first.v_fwd + second.v_fwd,
        v_bwd=first.v_bwd + second.v_bwd,
        cls=track_class(obs),
        keys=a.keys | b.keys,
    )


def _sweep(
    tracklets: list[_Tracklet],
    times: np.ndarray,
    cfg: RelinkConfig,
    max_centre_dist_m: float,
) -> tuple[list[_Tracklet], int]:
    """One pass over the frames; returns the tracklets and the fusions made."""
    h = cfg.horizon_s
    similarity = _SIMILARITY[cfg.metric]
    # Candidate pairs: the earlier ends before the later starts, and both can
    # have a state at one frame (a gap of at most two horizons).
    by_first = sorted(range(len(tracklets)), key=lambda i: tracklets[i].first)
    firsts = np.array([tracklets[i].first for i in by_first])
    at_frame: dict[int, list[tuple[int, int]]] = {}
    for a, A in enumerate(tracklets):
        lo = int(np.searchsorted(firsts, A.last, side="right"))
        hi = int(np.searchsorted(firsts, A.last + 2 * h, side="right"))
        for b in (by_first[x] for x in range(lo, hi)):
            B = tracklets[b]
            if cfg.class_consistent and A.cls and B.cls and A.cls != B.cls:
                continue
            f_lo = int(np.searchsorted(times, max(B.first - h, A.first), "left"))
            f_hi = int(np.searchsorted(times, min(A.last + h, B.last), "right"))
            for f in range(f_lo, f_hi):
                at_frame.setdefault(f, []).append((a, b))

    fused: set[int] = set()
    out: list[_Tracklet] = []
    for f in sorted(at_frame):
        t = float(times[f])
        g = nx.Graph()
        for a, b in at_frame[f]:
            if a in fused or b in fused:
                continue
            A, B = tracklets[a], tracklets[b]
            sa, sb = A.state(t, h), B.state(t, h)
            if sa is None or sb is None:
                continue
            cost = 1.0 - similarity(sa, sb)
            if cfg.shared_instance and A.keys & B.keys:
                d = math.dist((sa.cx, sa.cy, sa.cz), (sb.cx, sb.cy, sb.cz))
                if d <= max_centre_dist_m:
                    cost = 0.0
            if cost < cfg.max_cost:
                g.add_edge(a, b, weight=cfg.max_cost - cost)
        for a, b in nx.max_weight_matching(g):
            fused.update((a, b))
            out.append(_fuse(tracklets[a], tracklets[b]))
    out += [trk for i, trk in enumerate(tracklets) if i not in fused]
    return out, len(fused) // 2


def relink(
    tracks: list[list[Observation]],
    frames: list[DetFrame],
    cfg: RelinkConfig,
    max_centre_dist_m: float,
) -> list[list[Observation]]:
    """The chunk's tracks with every STWO match fused, ordered by their
    earliest original track (config `relink`)."""
    if len(tracks) < 2 or not frames:
        return tracks
    t0 = frames[0].t_ns
    times = np.array([(f.t_ns - t0) * 1e-9 for f in frames])
    tracklets = [_tracklet(i, obs, t0) for i, obs in enumerate(tracks)]
    total = sweeps = 0
    while True:
        tracklets, n = _sweep(tracklets, times, cfg, max_centre_dist_m)
        sweeps += 1
        total += n
        if n == 0:
            break
    log.info(
        "relink: %d tracks → %d (%d fusions over %d sweeps)",
        len(tracks),
        len(tracklets),
        total,
        sweeps,
    )
    return [trk.obs for trk in sorted(tracklets, key=lambda x: x.order)]
