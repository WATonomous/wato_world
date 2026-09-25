"""Box-to-track association and a minimal multi-object tracker.

Follows Chen et al., "Automatic Labeling to Generate Training Data for Online
LiDAR-based Moving Object Segmentation" (arXiv 2201.04501), §III-D:

  C_ij = α_d·c_d + α_o·c_o + α_v·c_v                       (Eq. 4)
  c_d  = ‖c_i − c_j‖₂                                      (Eq. 5)
  c_o  = 1 − IoU(b_i, b_j)                                 (Eq. 6)
  c_v  = 1 − min(v_i, v_j) / max(v_i, v_j)                 (Eq. 7)

solved with the Hungarian method; a matched pair is rejected — and the
detection starts a new track — if any term exceeds its gate (Flag_add,
Eq. 8). Tracks that miss a frame are kept for `n_old` frames and re-identify
against later detections using their continued motion prediction.

Defaults are the paper's (§III-F): α = 1, T_d = 2 m, T_o = 0.95, T_v = 0.7,
n_old = 5.

This tracker is deliberately minimal — geometry only, no appearance, no
class. lidar_preprocessing uses it to score whether a cluster moved; the
`tracking` component is expected to build on the same primitives (adding
ReID and class) rather than grow a second motion model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from wato_common.schemas import Box3D
from wato_common.tracking.boxes import box_volume, iou_3d
from wato_common.tracking.kalman import BoxKalmanFilter, KalmanNoise


@dataclass(frozen=True)
class AssociationGates:
    alpha_d: float = 1.0
    alpha_o: float = 1.0
    alpha_v: float = 1.0
    t_d: float = 2.0  # m — max centre distance
    t_o: float = 0.95  # max (1 − IoU)
    t_v: float = 0.7  # max (1 − volume ratio)


def cost_terms(det: Box3D, pred: Box3D) -> tuple[float, float, float]:
    """(c_d, c_o, c_v) between a detection and a track's predicted box."""
    c_d = float(np.linalg.norm([det.cx - pred.cx, det.cy - pred.cy, det.cz - pred.cz]))
    c_o = 1.0 - iou_3d(det, pred)
    vd, vp = box_volume(det), box_volume(pred)
    c_v = 1.0 - min(vd, vp) / max(vd, vp)
    return c_d, c_o, c_v


def associate(
    dets: list[Box3D],
    preds: list[Box3D],
    gates: AssociationGates = AssociationGates(),
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian assignment of detections to predicted track boxes.

    Returns (matches as (det_idx, pred_idx), unmatched det idxs, unmatched
    pred idxs). A Hungarian pair that violates any gate is split into one
    unmatched detection and one unmatched prediction (Chen Flag_add).
    """
    nd, npred = len(dets), len(preds)
    if nd == 0 or npred == 0:
        return [], list(range(nd)), list(range(npred))

    cost = np.empty((nd, npred))
    ok = np.empty((nd, npred), dtype=bool)
    for i, d in enumerate(dets):
        for j, p in enumerate(preds):
            c_d, c_o, c_v = cost_terms(d, p)
            cost[i, j] = gates.alpha_d * c_d + gates.alpha_o * c_o + gates.alpha_v * c_v
            ok[i, j] = c_d <= gates.t_d and c_o <= gates.t_o and c_v <= gates.t_v
    # Gate BEFORE solving so an infeasible pair can't displace a feasible one.
    big = 1e6
    rows, cols = linear_sum_assignment(np.where(ok, cost, big))
    matches = [(int(r), int(c)) for r, c in zip(rows, cols) if ok[r, c]]
    md = {m[0] for m in matches}
    mp = {m[1] for m in matches}
    return (
        matches,
        [i for i in range(nd) if i not in md],
        [j for j in range(npred) if j not in mp],
    )


@dataclass
class TrackObservation:
    """One associated detection in a track's history."""

    frame_idx: int
    t: float  # seconds
    det_idx: int  # index into that frame's detection list
    measured: Box3D
    filtered: Box3D


@dataclass
class Track:
    track_id: int
    kf: BoxKalmanFilter
    last_t: float
    misses: int = 0
    history: list[TrackObservation] = field(default_factory=list)


class MultiObjectTracker:
    """Frame-by-frame tracker: predict → associate → update / birth / expire.

    Tracks missing up to `n_old` consecutive frames stay associable (their
    Kalman prediction keeps coasting); older ones are retired. Every track
    ever created — retired or live — is available via `all_tracks()`.
    """

    def __init__(
        self,
        gates: AssociationGates = AssociationGates(),
        n_old: int = 5,
        noise: KalmanNoise | None = None,
    ) -> None:
        self.gates = gates
        self.n_old = int(n_old)
        self.noise = noise
        self._live: list[Track] = []
        self._retired: list[Track] = []
        self._next_id = 0
        self._frame_idx = -1

    def step(self, t: float, dets: list[Box3D]) -> list[int]:
        """Process one frame at time t (seconds). Returns track_id per det."""
        self._frame_idx += 1
        preds = [trk.kf.predict(t - trk.last_t) for trk in self._live]
        for trk in self._live:
            trk.last_t = t
        matches, unmatched_dets, unmatched_trks = associate(dets, preds, self.gates)

        ids = [-1] * len(dets)
        for di, ti in matches:
            trk = self._live[ti]
            filtered = trk.kf.update(dets[di])
            trk.misses = 0
            trk.history.append(
                TrackObservation(self._frame_idx, t, di, dets[di], filtered)
            )
            ids[di] = trk.track_id

        for ti in unmatched_trks:
            self._live[ti].misses += 1

        still_live: list[Track] = []
        for trk in self._live:
            (still_live if trk.misses <= self.n_old else self._retired).append(trk)
        self._live = still_live

        for di in unmatched_dets:
            kf = BoxKalmanFilter(dets[di], self.noise)
            trk = Track(track_id=self._next_id, kf=kf, last_t=t)
            trk.history.append(
                TrackObservation(self._frame_idx, t, di, dets[di], kf.box)
            )
            self._live.append(trk)
            ids[di] = trk.track_id
            self._next_id += 1
        return ids

    def all_tracks(self) -> list[Track]:
        return sorted(self._retired + self._live, key=lambda trk: trk.track_id)
