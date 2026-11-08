"""Ground-truth motion of one labelled object.

motion_state: its parked / moving state over its whole life. 3D Auto Labeling
(Qi et al., CVPR 2021) labels an object static when it neither travels nor
moves fast over its life; the same cut-offs serve here: parked iff the
first-to-last centre distance is under MAX_NET_DISPLACEMENT_M and no step
between consecutive labels is faster than MAX_SPEED_MPS. A car that waits at a
light and then leaves is moving.

instant_speeds: its speed at each label, for scoring per-instant moving-point
verdicts (Step F), where that waiting car is not moving.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from wato_common.artifacts.gt_boxes import MOVING, PARKED

MAX_NET_DISPLACEMENT_M = 1.0
MAX_SPEED_MPS = 1.0


def motion_state(t_ns: np.ndarray, xy: np.ndarray) -> Optional[str]:
    """PARKED or MOVING from an object's labelled BEV centres; None with
    fewer than two labels (nothing to judge motion from).

    t_ns: (N,) label times; xy: (N, 2) centres, any order (sorted here).
    """
    if t_ns.shape[0] < 2:
        return None
    order = np.argsort(t_ns, kind="stable")
    t = t_ns[order].astype(np.float64) * 1e-9
    p = xy[order].astype(np.float64)
    net = float(np.linalg.norm(p[-1] - p[0]))
    dt = np.diff(t)
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    ok = dt > 0
    speed = float((step[ok] / dt[ok]).max()) if ok.any() else 0.0
    if net < MAX_NET_DISPLACEMENT_M and speed < MAX_SPEED_MPS:
        return PARKED
    return MOVING


def instant_speeds(t_ns: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """(N,) BEV speed [m/s] at each label, in the input order: the centred
    difference of its neighbouring labels (one-sided at the ends); 0 for a
    single label."""
    n = t_ns.shape[0]
    out = np.zeros(n)
    if n < 2:
        return out
    order = np.argsort(t_ns, kind="stable")
    t = t_ns[order].astype(np.float64) * 1e-9
    p = xy[order].astype(np.float64)
    lo = np.r_[0, np.arange(n - 1)]
    hi = np.r_[np.arange(1, n), n - 1]
    dt = t[hi] - t[lo]
    v = np.linalg.norm(p[hi] - p[lo], axis=1) / np.where(dt > 0, dt, np.inf)
    out[order] = v
    return out
