"""Datasheet-derived LiDAR sensor model.

Single source of the inverse-sensor-model constants the static/dynamic
classifier needs. The classifier takes no hand-tuned log-odds knobs; it reads
a profile here and derives everything (l_occ, l_free, clamp, decision
thresholds, range-credibility falloff, carve margin, grazing gate) from the
profile's physical specs. p_hit/p_miss/p_clamp are the standard occupancy-grid
inverse-sensor-model probabilities (Thrun, Probabilistic Robotics ch. 9).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np


def logit(p: float) -> float:
    """ln(p / (1 - p))."""
    if not (0.0 < p < 1.0):
        raise ValueError(f"logit() needs p in (0, 1), got {p}")
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class SensorModel:
    """One LiDAR family's physical specs + the constants derived from them.

    range_sigma_m:       1σ range accuracy [m].
    beam_divergence_rad: full-angle beam divergence [rad].
    max_range_m:         max usable range [m]; a compute guard only.
    p_hit:   P(occupied | a return landed here). High but < 1 (dust/rain/mixed).
    p_miss:  P(occupied | a ray passed through). < 0.5 (a pass-through is free
             evidence, weakened by beam divergence / thin-object misses).
    p_clamp: saturation confidence; caps |log-odds|.
    k_sigma: σ-count of endpoint uncertainty protected from self-carving.
    p_map_prior: one-time reinforcement for a global-map-matched voxel.
    rotation_dir: scanner spin direction (Velodyne spins CW from above).
    """

    name: str
    range_sigma_m: float
    beam_divergence_rad: float
    max_range_m: float
    p_hit: float
    p_miss: float
    p_clamp: float
    rotation_dir: Literal["cw", "ccw"]
    k_sigma: float = 3.0
    p_map_prior: float = 0.75

    # --- Log-odds increments / bounds ---

    @property
    def l_occ(self) -> float:
        """Log-odds added at an endpoint."""
        return logit(self.p_hit)

    @property
    def l_free(self) -> float:
        """Log-odds magnitude subtracted per through-ray (kernel subtracts it)."""
        return logit(1.0 - self.p_miss)

    @property
    def log_odds_clamp(self) -> float:
        return logit(self.p_clamp)

    @property
    def l_map_prior(self) -> float:
        return logit(self.p_map_prior)

    # --- Decision thresholds: static at p_hit, dynamic symmetric at 1-p_hit ---

    @property
    def p_static_threshold(self) -> float:
        return self.p_hit

    @property
    def p_dynamic_threshold(self) -> float:
        # NOT p_miss: p_miss (≈0.4) sits just below 0.5 and floods dynamic.
        return 1.0 - self.p_hit

    # --- Range credibility: beam footprint d·divergence crosses a voxel at d* ---

    def credibility_crossover_m(self, voxel_size_m: float) -> float:
        """Range beyond which pass-through evidence is down-weighted ∝ 1/d."""
        return voxel_size_m / self.beam_divergence_rad

    def range_weight(self, d: np.ndarray | float, voxel_size_m: float) -> np.ndarray:
        """min(1, d* / d)."""
        d_star = self.credibility_crossover_m(voxel_size_m)
        return np.minimum(
            1.0, d_star / np.maximum(np.asarray(d, dtype=np.float64), 1e-9)
        )

    # --- Carving geometry ---

    def grazing_cos_threshold(self, voxel_size_m: float) -> float:
        """|ray·n| below this → ray grazes an occupied voxel; skip the carve.

        A ray clears a surface centred in a voxel only if its normal-direction
        travel across the voxel exceeds the surface half-thickness:
        voxel·|ray·n| > 0.5·voxel + k_sigma·σ_range. Derived from voxel
        geometry, not a picked angle (0.5 ⇒ within 30° of grazing).
        """
        return min(0.95, 0.5 + self.k_sigma * self.range_sigma_m / voxel_size_m)

    def carve_margin_m(self, pose_sigma_m: float) -> float:
        """Stop carving k_sigma·√(σ_range² + σ_pose²) short of the endpoint."""
        return self.k_sigma * math.sqrt(self.range_sigma_m**2 + pose_sigma_m**2)


# Velodyne datasheet specs; p_hit/p_miss/p_clamp are inverse-sensor-model
# reliability params, not dataset-tuned.
_PROFILES: dict[str, SensorModel] = {
    # WATO rig: VLP-32C centre + 2× VLP-16. ±3 cm, ~3 mrad; usable ~120 m
    # (VLP-32C spec 200 m but far returns sparse, VLP-16 reaches ~100 m).
    "velodyne_vlp": SensorModel(
        name="velodyne_vlp",
        range_sigma_m=0.03,
        beam_divergence_rad=0.003,
        max_range_m=120.0,
        p_hit=0.88,
        p_miss=0.40,
        p_clamp=0.99,
        rotation_dir="cw",
    ),
    # HDL-32E-class LIDAR_TOP. ±2 cm, ~2.8 mrad, usable ~80 m.
    "nuscenes": SensorModel(
        name="nuscenes",
        range_sigma_m=0.02,
        beam_divergence_rad=0.0028,
        max_range_m=80.0,
        p_hit=0.88,
        p_miss=0.40,
        p_clamp=0.99,
        rotation_dir="cw",
    ),
}


def get_sensor_model(profile: str) -> SensorModel:
    try:
        return _PROFILES[profile]
    except KeyError:
        raise ValueError(
            f"unknown sensor_model profile {profile!r}; valid: {sorted(_PROFILES)}"
        ) from None


def estimate_pose_sigma_m(translations_m: np.ndarray, *, floor_m: float = 0.0) -> float:
    """Estimate inter-sample SLAM pose noise [m] from a translation trajectory.

    Ego motion is smooth, so the discrete second difference p[i-1]-2p[i]+p[i+1]
    cancels constant velocity and leaves curvature + noise (each component
    ~N(0, 6σ²) for per-axis noise σ). Robust estimate from the pooled
    components: σ ≈ 1.4826·median(|·|)/√6. Returns max(σ, floor_m).
    """
    t = np.asarray(translations_m, dtype=np.float64)
    if t.ndim != 2 or t.shape[1] != 3 or t.shape[0] < 3:
        return float(floor_m)
    comp = (t[:-2] - 2.0 * t[1:-1] + t[2:]).reshape(-1)
    sigma = 1.4826 * float(np.median(np.abs(comp))) / math.sqrt(6.0)
    return float(max(sigma, floor_m))
