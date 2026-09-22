"""Datasheet-derived LiDAR sensor model — one profile per physical scanner.

This module is the single source of every per-sensor number the component
uses. YAML picks a profile (per lidar_id on a mixed rig); it never states a
physical constant. Two groups of fields live here:

**Datasheet specs** — `beams`, `fov_up_deg`, `fov_down_deg`, `range_sigma_m`,
`beam_divergence_rad`, `max_range_m`, `sweep_duration_ms`, `intensity_scale`,
`rotation_dir`. These are published numbers; a wrong value is a bug with a
citable answer. Everything downstream reads them: deskew's azimuth-to-time
synthesis, classify's log-odds constants and carve geometry, and MF-MOS's
spherical projection.

**Inverse-sensor-model probabilities** — `p_hit`, `p_miss`, `p_clamp`,
`k_sigma`, `p_map_prior`. These are NOT on any datasheet. They are the
occupancy-grid reliability parameters of Thrun, *Probabilistic Robotics* ch. 9,
and they are the only tuning surface left in the component. They live here, in
code, with their reasoning written down (see the comment block below) rather
than in YAML, because a config that can move them is a
config that will be tuned per bag until the labels stop being comparable.

Derived quantities (`l_occ`, `l_free`, `log_odds_clamp`, the decision
thresholds, the range-credibility falloff, the carve margin, the grazing gate)
are properties/methods here. Nothing recomputes them elsewhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

# Why the inverse-sensor-model probabilities are what they are.
#
# p_hit = 0.88 — P(voxel occupied | a return landed in it). Below 1 because a
#   return is not proof of a persistent surface: rain, dust, exhaust, spray and
#   mixed (edge-straddling) pixels all produce returns from empty space, and a
#   single sweep cannot tell those from a wall. 0.88 gives l_occ ≈ 2.0, so
#   three clean hits saturate a voxel against the 0.99 clamp (l ≈ 4.6). Any
#   value in 0.85–0.92 behaves the same; the classifier is insensitive here
#   because the decision is a threshold on accumulated evidence, not on one
#   observation.
#
# p_miss = 0.40 — P(voxel occupied | a ray passed through it). Below 0.5 (a
#   pass-through IS evidence of free space) but not far below, because a
#   pass-through is the weaker of the two measurements: the beam has finite
#   width, so a ray recorded as passing "through" a voxel may have missed a
#   thin object — a pole, a railing, a pedestrian's arm — inside it. The
#   asymmetry l_occ ≈ 2.0 vs l_free ≈ 0.4 encodes "it takes about five carves
#   to undo one hit", which is what makes a parked car read static and a
#   driving car read dynamic over a chunk.
#
# p_clamp = 0.99 — saturation confidence, |log-odds| ≤ 4.6. Standard
#   occupancy-grid practice: without a clamp a voxel observed for a thousand
#   sweeps ossifies and can never be revised when the world changes (a car
#   parks, then leaves). The clamp bounds how much history one observation has
#   to overcome.
#
# k_sigma = 3.0 — how many sigmas of endpoint uncertainty to protect from
#   self-carving, used by both `carve_margin_m` and `grazing_cos_threshold`.
#   3σ ⇒ ~99.7% of the endpoint's error distribution falls inside the margin,
#   so a surface carves itself away in under 0.3% of rays. This is a
#   confidence level, not a tuned distance — the metres come from the
#   sensor's σ_range and the bag's estimated σ_pose.
#
# p_map_prior = 0.75 — one-time reinforcement applied to a voxel matched
#   against the bag-level global static map in two-pass mode (l ≈ 1.1, i.e.
#   worth about half a hit). Deliberately weaker than l_occ: the prior says
#   "other chunks saw structure here", which is supporting evidence, not a
#   measurement of this chunk.
#
# What would move these: a labelled static/dynamic benchmark on WATO bags.
# Until that exists, they stay fixed — shared across all profiles, so a
# profile switch changes geometry only, never the decision rule.
_P_HIT = 0.88
_P_MISS = 0.40
_P_CLAMP = 0.99


def logit(p: float) -> float:
    """ln(p / (1 - p))."""
    if not (0.0 < p < 1.0):
        raise ValueError(f"logit() needs p in (0, 1), got {p}")
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class SensorModel:
    """One physical scanner's datasheet specs + the constants derived from them.

    Datasheet fields:
        beams:               vertical channel count (= MF-MOS range-image rows).
        fov_up_deg:          vertical FoV upper bound [deg].
        fov_down_deg:        vertical FoV lower bound [deg], negative.
        range_sigma_m:       1σ range accuracy [m].
        beam_divergence_rad: full-angle beam divergence [rad].
        max_range_m:         max usable range [m]; also the carve compute guard.
        sweep_duration_ms:   one full rotation [ms]; azimuth→time synthesis.
        intensity_scale:     divisor mapping raw intensity into [0, 1].
        rotation_dir:        spin direction seen from above.

    Inverse-sensor-model fields (picked, not measured — see module docstring):
        p_hit, p_miss, p_clamp, k_sigma, p_map_prior.
    """

    name: str
    beams: int
    fov_up_deg: float
    fov_down_deg: float
    range_sigma_m: float
    beam_divergence_rad: float
    max_range_m: float
    sweep_duration_ms: float
    intensity_scale: float
    rotation_dir: Literal["cw", "ccw"]
    p_hit: float = _P_HIT
    p_miss: float = _P_MISS
    p_clamp: float = _P_CLAMP
    k_sigma: float = 3.0
    p_map_prior: float = 0.75

    # --- Acquisition ---

    @property
    def sweep_rate_hz(self) -> float:
        return 1000.0 / self.sweep_duration_ms

    @property
    def sweep_duration_ns(self) -> float:
        return self.sweep_duration_ms * 1_000_000.0

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


# One entry per physical scanner. Mixed rigs map lidar_id → profile in YAML.
_PROFILES: dict[str, SensorModel] = {
    # WATO rig centre scanner. Datasheet: 32 channels, +15°/-25°, ±3 cm,
    # ~3 mrad, 5-20 Hz. Spec range is 200 m but returns past ~120 m are too
    # sparse to carve with, so the usable guard is 120 m.
    "vlp32c": SensorModel(
        name="vlp32c",
        beams=32,
        fov_up_deg=15.0,
        fov_down_deg=-25.0,
        range_sigma_m=0.03,
        beam_divergence_rad=0.003,
        max_range_m=120.0,
        sweep_duration_ms=50.0,  # WATO runs the rig at ~20 Hz
        intensity_scale=255.0,
        rotation_dir="cw",
    ),
    # WATO rig corner scanners. Datasheet: 16 channels, ±15°, ±3 cm, ~3 mrad,
    # 100 m range. Too few channels for the MF-MOS projection (see mf_mos).
    "vlp16": SensorModel(
        name="vlp16",
        beams=16,
        fov_up_deg=15.0,
        fov_down_deg=-15.0,
        range_sigma_m=0.03,
        beam_divergence_rad=0.003,
        max_range_m=100.0,
        sweep_duration_ms=50.0,
        intensity_scale=255.0,
        rotation_dir="cw",
    ),
    # nuScenes LIDAR_TOP. HDL-32E: 32 channels, +10.67°/-30.67°, ±2 cm,
    # ~2.8 mrad, 20 Hz, usable ~80 m.
    "hdl32e": SensorModel(
        name="hdl32e",
        beams=32,
        fov_up_deg=10.67,
        fov_down_deg=-30.67,
        range_sigma_m=0.02,
        beam_divergence_rad=0.0028,
        max_range_m=80.0,
        sweep_duration_ms=50.0,
        intensity_scale=255.0,
        rotation_dir="cw",
    ),
}


def available_profiles() -> list[str]:
    return sorted(_PROFILES)


def get_sensor_model(profile: str) -> SensorModel:
    try:
        return _PROFILES[profile]
    except KeyError:
        raise ValueError(
            f"unknown sensor_model profile {profile!r}; valid: {available_profiles()}"
        ) from None


def estimate_pose_sigma_m(
    translations_m: np.ndarray, *, floor_m: Optional[float] = 0.0
) -> float:
    """Estimate inter-sample SLAM pose noise [m] from a translation trajectory.

    Ego motion is smooth, so the discrete second difference p[i-1]-2p[i]+p[i+1]
    cancels constant velocity and leaves curvature + noise (each component
    ~N(0, 6σ²) for per-axis noise σ). Robust estimate from the pooled
    components: σ ≈ 1.4826·median(|·|)/√6. Returns max(σ, floor_m).
    """
    floor = float(floor_m or 0.0)
    t = np.asarray(translations_m, dtype=np.float64)
    if t.ndim != 2 or t.shape[1] != 3 or t.shape[0] < 3:
        return floor
    comp = (t[:-2] - 2.0 * t[1:-1] + t[2:]).reshape(-1)
    sigma = 1.4826 * float(np.median(np.abs(comp))) / math.sqrt(6.0)
    return float(max(sigma, floor))
