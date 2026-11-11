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

`AssociationGates.cost = "giou_3d"` swaps in SimpleTrack's cost instead
(arXiv 2111.09621 §4.3.1): 1 − GIoU (3D), one gate T_giou (1.5, GIoU > −0.5
for every class on nuScenes and WOD). IoU is 0 for every pair of disjoint
boxes; GIoU keeps falling as they move apart, so it still ranks a one-sided
sliver against the track's whole box.

This tracker is deliberately minimal — geometry only, no appearance, no
class. lidar_preprocessing uses it as is to score whether a cluster moved.
The `tracking` component adds optional cues on top, all off by default:
the GIoU cost; tracks that never retire (n_old None, Immortal Tracker);
detections flagged strong are matched before weak ones (SimpleTrack's
two-stage matching, so a weak box can't take a track a strong one fits), and a
detection whose key (a lifted instance id) equals the track's last key gets
`key_bonus` off its cost. With `key_opens_shape_gates`, such a pair is held
only to the distance gate T_d: the shape gates (IoU and volume, or GIoU) exist
to stop a track jumping to a neighbour, and a shared upstream identity
already rules that out, while a one-sided box's IoU and volume swing as its
visible face changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from wato_common.schemas import Box3D
from wato_common.tracking.boxes import (
    box_volume,
    giou_3d,
    giou_3d_cannot_exceed,
    iou_3d,
)
from wato_common.tracking.kalman import BoxKalmanFilter, KalmanNoise

CHEN = "chen"
GIOU_3D = "giou_3d"


@dataclass(frozen=True)
class AssociationGates:
    alpha_d: float = 1.0
    alpha_o: float = 1.0
    alpha_v: float = 1.0
    t_d: float = 2.0  # m — max centre distance
    t_o: float = 0.95  # max (1 − IoU)
    t_v: float = 0.7  # max (1 − volume ratio)
    cost: str = CHEN  # or GIOU_3D: cost 1 − GIoU, gated by t_giou alone
    t_giou: float = 1.5  # max (1 − GIoU); SimpleTrack's GIoU > −0.5

    def __post_init__(self) -> None:
        if self.cost not in (CHEN, GIOU_3D):
            raise ValueError(f"cost {self.cost!r}: expected {CHEN!r} or {GIOU_3D!r}")


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
    cost_adjust: Optional[np.ndarray] = None,
    distance_gate_only: Optional[np.ndarray] = None,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian assignment of detections to predicted track boxes.

    Returns (matches as (det_idx, pred_idx), unmatched det idxs, unmatched
    pred idxs). A Hungarian pair that violates any gate is split into one
    unmatched detection and one unmatched prediction (Chen Flag_add).
    cost_adjust, (n_dets, n_preds), is added to the cost after gating: it
    reorders feasible pairs and never makes an infeasible one feasible.
    distance_gate_only, (n_dets, n_preds) bool, exempts those pairs from the
    shape gates (IoU and volume, or GIoU); the distance gate still holds.
    """
    nd, npred = len(dets), len(preds)
    if nd == 0 or npred == 0:
        return [], list(range(nd)), list(range(npred))

    # Pairs that provably fail every gate open to them are skipped; their
    # cost is never read, so the result is the same as scoring every pair.
    centre = np.linalg.norm(
        np.array([[d.cx, d.cy, d.cz] for d in dets])[:, None, :]
        - np.array([[p.cx, p.cy, p.cz] for p in preds])[None, :, :],
        axis=2,
    )
    exempt = (
        np.zeros((nd, npred), dtype=bool)
        if distance_gate_only is None
        else np.asarray(distance_gate_only, dtype=bool)
    )
    near = centre <= gates.t_d
    if gates.cost == GIOU_3D:
        hopeless = giou_3d_cannot_exceed(dets, preds, 1.0 - gates.t_giou)
        candidate = ~hopeless | (exempt & near)
    else:
        candidate = centre <= gates.t_d + 1e-9  # c_d is re-checked exactly
    cost = np.zeros((nd, npred))
    ok = np.zeros((nd, npred), dtype=bool)
    for i, j in zip(*np.nonzero(candidate)):
        d, p = dets[i], preds[j]
        if gates.cost == GIOU_3D:
            c = 1.0 - giou_3d(d, p)
            cost[i, j] = c
            ok[i, j] = c <= gates.t_giou or (exempt[i, j] and near[i, j])
            continue
        c_d, c_o, c_v = cost_terms(d, p)
        cost[i, j] = gates.alpha_d * c_d + gates.alpha_o * c_o + gates.alpha_v * c_v
        shape_ok = (c_o <= gates.t_o and c_v <= gates.t_v) or exempt[i, j]
        ok[i, j] = c_d <= gates.t_d and shape_ok
    if cost_adjust is not None:
        cost = cost + cost_adjust
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
    velocity: Optional[tuple[float, float, float]] = None  # filtered, m/s


@dataclass
class Track:
    track_id: int
    kf: BoxKalmanFilter
    last_t: float
    misses: int = 0
    history: list[TrackObservation] = field(default_factory=list)
    last_key: Optional[Hashable] = None  # key of its latest detection, if any


class MultiObjectTracker:
    """Frame-by-frame tracker: predict → associate → update / birth / expire.

    Tracks missing up to `n_old` consecutive frames stay associable (their
    Kalman prediction keeps coasting); older ones are retired. n_old None
    never retires a track (Immortal Tracker, arXiv 2111.13672 §3: most ID
    switches are tracks ended early). Every track ever created — retired or
    live — is available via `all_tracks()`.
    """

    def __init__(
        self,
        gates: AssociationGates = AssociationGates(),
        n_old: Optional[int] = 5,
        noise: KalmanNoise | None = None,
        key_bonus: float = 0.0,
        key_opens_shape_gates: bool = False,
    ) -> None:
        self.gates = gates
        self.n_old = None if n_old is None else int(n_old)
        self.noise = noise
        self.key_bonus = float(key_bonus)
        self.key_opens_shape_gates = bool(key_opens_shape_gates)
        self._live: list[Track] = []
        self._retired: list[Track] = []
        self._next_id = 0
        self._frame_idx = -1

    def step(
        self,
        t: float,
        dets: list[Box3D],
        *,
        strong: Optional[Sequence[bool]] = None,
        keys: Optional[Sequence[Optional[Hashable]]] = None,
        can_birth: Optional[Sequence[bool]] = None,
    ) -> list[int]:
        """Process one frame at time t (seconds). Returns track_id per det
        (-1 for a det that neither matched nor started a track).

        strong: per det; when given, strong dets are matched first and weak
        ones only to the tracks left. keys: per det (None = no key); a det
        whose key equals a track's last key costs key_bonus less.
        can_birth: per det; when given, an unmatched det marked False starts
        no track and is dropped (ByteTrack: a low-score box may extend a
        track, never start one).
        """
        self._frame_idx += 1
        preds = [trk.kf.predict(t - trk.last_t) for trk in self._live]
        for trk in self._live:
            trk.last_t = t

        def same_key(det_idx: list[int], trk_idx: list[int]) -> Optional[np.ndarray]:
            if keys is None:
                return None
            m = np.zeros((len(det_idx), len(trk_idx)), dtype=bool)
            for r, di in enumerate(det_idx):
                if keys[di] is None:
                    continue
                for c, ti in enumerate(trk_idx):
                    m[r, c] = self._live[ti].last_key == keys[di]
            return m

        stages = (
            [
                [i for i in range(len(dets)) if strong[i]],
                [i for i in range(len(dets)) if not strong[i]],
            ]
            if strong is not None
            else [list(range(len(dets)))]
        )
        free_trks = list(range(len(self._live)))
        matches: list[tuple[int, int]] = []
        unmatched_dets: list[int] = []
        for det_idx in stages:
            shared = same_key(det_idx, free_trks)
            m, ud, ut = associate(
                [dets[i] for i in det_idx],
                [preds[j] for j in free_trks],
                self.gates,
                -self.key_bonus * shared
                if shared is not None and self.key_bonus
                else None,
                shared if self.key_opens_shape_gates else None,
            )
            matches += [(det_idx[d], free_trks[p]) for d, p in m]
            unmatched_dets += [det_idx[d] for d in ud]
            free_trks = [free_trks[p] for p in ut]
        unmatched_trks = free_trks

        ids = [-1] * len(dets)
        for di, ti in matches:
            trk = self._live[ti]
            filtered = trk.kf.update(dets[di])
            trk.misses = 0
            trk.history.append(
                TrackObservation(self._frame_idx, t, di, dets[di], filtered, _v(trk.kf))
            )
            if keys is not None:
                trk.last_key = keys[di] if keys[di] is not None else trk.last_key
            ids[di] = trk.track_id

        for ti in unmatched_trks:
            self._live[ti].misses += 1

        still_live: list[Track] = []
        for trk in self._live:
            retire = self.n_old is not None and trk.misses > self.n_old
            (self._retired if retire else still_live).append(trk)
        self._live = still_live

        for di in sorted(unmatched_dets):
            if can_birth is not None and not can_birth[di]:
                continue
            kf = BoxKalmanFilter(dets[di], self.noise)
            trk = Track(
                track_id=self._next_id,
                kf=kf,
                last_t=t,
                last_key=keys[di] if keys is not None else None,
            )
            trk.history.append(
                TrackObservation(self._frame_idx, t, di, dets[di], kf.box, _v(kf))
            )
            self._live.append(trk)
            ids[di] = trk.track_id
            self._next_id += 1
        return ids

    def all_tracks(self) -> list[Track]:
        return sorted(self._retired + self._live, key=lambda trk: trk.track_id)


def _v(kf: BoxKalmanFilter) -> tuple[float, float, float]:
    vx, vy, vz = kf.velocity
    return float(vx), float(vy), float(vz)
