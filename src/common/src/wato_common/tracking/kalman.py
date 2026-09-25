"""Constant-velocity Kalman filter over a yaw-only 3D box.

State (10): [x, y, z, heading, l, w, h, vx, vy, vz]
Measurement (7): [x, y, z, heading, l, w, h]

The motion model Chen et al. (arXiv 2201.04501 §III-D) and AB3DMOT use:
position integrates velocity, heading and extents are slow random walks.
Linear in every term, so a plain KF is exact — the only non-linearity is the
heading wrap, handled on the innovation. Cluster boxes have no front/back, so
heading is undirected: innovations wrap into (-pi/2, pi/2].

Shared by lidar_preprocessing's motion-proposal scoring and (later) the
`tracking` component, so both reason about motion with the same model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wato_common.schemas import Box3D
from wato_common.tracking.boxes import wrap_heading_half_pi

_N_STATE = 10
_N_MEAS = 7


@dataclass(frozen=True)
class KalmanNoise:
    """Noise model. Units are per-second for process terms, 1σ throughout.

    Defaults target LiDAR cluster boxes (noisier than detector boxes):
    extents and centres of a partially visible cluster move by tens of cm
    between sweeps even for a static object.
    """

    accel_sigma: float = 3.0  # m/s² — white-noise acceleration on x/y/z
    heading_rate_sigma: float = 0.3  # rad/s random walk
    extent_rate_sigma: float = 0.2  # m/s random walk on l/w/h
    meas_pos_sigma: float = 0.3  # m
    meas_heading_sigma: float = 0.3  # rad
    meas_extent_sigma: float = 0.3  # m
    init_vel_sigma: float = 10.0  # m/s — velocity unknown at birth


class BoxKalmanFilter:
    """One track's filter. Construct from its first measured box."""

    def __init__(self, box: Box3D, noise: KalmanNoise | None = None) -> None:
        self.noise = noise or KalmanNoise()
        n = self.noise
        self.x = np.zeros(_N_STATE)
        self.x[:7] = _box_to_meas(box)
        self.P = np.diag(
            [n.meas_pos_sigma**2] * 3
            + [n.meas_heading_sigma**2]
            + [n.meas_extent_sigma**2] * 3
            + [n.init_vel_sigma**2] * 3
        )
        self._H = np.zeros((_N_MEAS, _N_STATE))
        self._H[:, :7] = np.eye(_N_MEAS)
        self._R = np.diag(
            [n.meas_pos_sigma**2] * 3
            + [n.meas_heading_sigma**2]
            + [n.meas_extent_sigma**2] * 3
        )

    def predict(self, dt: float) -> Box3D:
        """Advance the state by dt seconds; returns the predicted box."""
        dt = max(float(dt), 0.0)
        n = self.noise
        F = np.eye(_N_STATE)
        F[0, 7] = F[1, 8] = F[2, 9] = dt
        Q = np.zeros((_N_STATE, _N_STATE))
        q = n.accel_sigma**2
        for p, v in ((0, 7), (1, 8), (2, 9)):
            Q[p, p] = q * dt**4 / 4.0
            Q[p, v] = Q[v, p] = q * dt**3 / 2.0
            Q[v, v] = q * dt**2
        Q[3, 3] = (n.heading_rate_sigma * dt) ** 2
        for i in (4, 5, 6):
            Q[i, i] = (n.extent_rate_sigma * dt) ** 2
        self.x = F @ self.x
        self.x[3] = wrap_heading_half_pi(self.x[3])
        self.P = F @ self.P @ F.T + Q
        return self.box

    def update(self, box: Box3D) -> Box3D:
        """Fuse a measured box; returns the filtered box."""
        z = _box_to_meas(box)
        y = z - self._H @ self.x
        y[3] = wrap_heading_half_pi(y[3])
        S = self._H @ self.P @ self._H.T + self._R
        K = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[3] = wrap_heading_half_pi(self.x[3])
        self.P = (np.eye(_N_STATE) - K @ self._H) @ self.P
        return self.box

    @property
    def box(self) -> Box3D:
        x = self.x
        return Box3D(
            cx=float(x[0]),
            cy=float(x[1]),
            cz=float(x[2]),
            w=float(max(x[5], 0.0)),
            l=float(max(x[4], 0.0)),
            h=float(max(x[6], 0.0)),
            heading=float(x[3]),
        )

    @property
    def velocity(self) -> np.ndarray:
        return self.x[7:10].copy()


def _box_to_meas(box: Box3D) -> np.ndarray:
    return np.array(
        [box.cx, box.cy, box.cz, box.heading, box.l, box.w, box.h], dtype=np.float64
    )
