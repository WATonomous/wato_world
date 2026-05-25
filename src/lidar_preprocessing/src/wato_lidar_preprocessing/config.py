"""Pydantic-loaded config for lidar_preprocessing."""

from __future__ import annotations

from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class FrameSyncParams(BaseModel):
    """Multi-lidar sweep-to-frame grouping for SAM4D-style downstream fusion.

    A "frame" is one tick of `canonical_lidar`.  Every non-canonical sweep
    whose reference_timestamp_ns falls within ±tolerance_ms of a canonical
    sweep inherits that sweep's frame_id.

    canonical_lidar=None disables grouping: each sweep becomes its own frame
    indexed sequentially per lidar_id.  That's the right behavior for
    single-lidar bags (NuScenes mini, KITTI) and for the current state of
    the WATO recordings where the 3-Velodyne rig isn't yet wired in.
    """

    model_config = ConfigDict(extra="forbid")

    canonical_lidar: Optional[str] = None
    tolerance_ms: float = 25.0

    @field_validator("tolerance_ms")
    @classmethod
    def _positive_tolerance(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"tolerance_ms must be > 0, got {v}")
        return v

    def tolerance_ns(self) -> int:
        return int(self.tolerance_ms * 1_000_000)


class PatchworkParams(BaseModel):
    """Patchwork++ ground-segmentation parameters.

    Defaults mirror wato_monorepo/src/perception/patchwork/patchwork/config/params.yaml.
    """

    model_config = ConfigDict(extra="forbid")

    sensor_height: float = 1.8
    num_iter: int = 3
    num_lpr: int = 20
    num_min_pts: int = 10
    th_seeds: float = 0.3
    th_dist: float = 0.15
    th_seeds_v: float = 0.25
    th_dist_v: float = 0.85
    max_range: float = 90.0
    min_range: float = 1.0
    uprightness_thr: float = 0.101
    enable_RNR: bool = False
    verbose: bool = False
    ground_cell_size_m: float = 0.25

    def to_patchwork_dict(self) -> dict[str, Any]:
        """Return kwargs accepted by pypatchworkpp.patchworkpp()."""
        return {
            "sensor_height": self.sensor_height,
            "num_iter": self.num_iter,
            "num_lpr": self.num_lpr,
            "num_min_pts": self.num_min_pts,
            "th_seeds": self.th_seeds,
            "th_dist": self.th_dist,
            "th_seeds_v": self.th_seeds_v,
            "th_dist_v": self.th_dist_v,
            "max_range": self.max_range,
            "min_range": self.min_range,
            "uprightness_thr": self.uprightness_thr,
            "enable_RNR": self.enable_RNR,
            "verbose": self.verbose,
        }


class MFMosParams(BaseModel):
    """MF-MOS moving-object segmentation parameters.

    Runs as step A.5 (between deskew and classify) when enabled is True.
    Requires a CUDA-capable GPU; set device="cpu" only for tiny smoke tests.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    checkpoint_path: str = "/data/models/mf_mos/mf_mos_semantic_kitti.pt"
    arch_config: str = "/data/models/mf_mos/arch_cfg.yaml"
    data_config: str = "/data/models/mf_mos/data_cfg.yaml"
    residual_steps: list[int] = Field(default_factory=lambda: [1, 2, 4, 8])
    range_image_h: int = 32
    range_image_w: int = 1024
    fov_up_deg: float = 10.0
    fov_down_deg: float = -30.0
    device: str = "cuda"
    score_threshold: float = 0.5
    save_scores: bool = False
    # independent: both masks written side-by-side, downstream chooses.
    # union: classify accumulators (static_map/dynamic_map) use voxel | mf_mos.
    # mfmos_only: classify accumulators use mf_mos mask only.
    # Per-sweep _dynamic_mask.npy always keeps the voxel-only mask.
    fusion_mode: str = "independent"
    max_pose_gap_ms: float = 200.0
    # Range gating — match the training preprocessing
    # (data_preparing.yaml: min_range=2.0, max_range=50.0).
    min_range_m: float = 2.0
    max_range_m: float = 50.0
    # Intensity scale: divide raw intensity by this before projection so the
    # channel lands in [0, 1] like KITTI remission. NuScenes intensity is
    # uint8 [0, 255] → 255.0. Set to 1.0 for sensors that already produce [0, 1].
    intensity_scale: float = 255.0
    # Per-LiDAR allowlist. The configured fov_up_deg / fov_down_deg / image
    # H/W are global — running them on a LiDAR whose mount/geometry differs
    # significantly (e.g. corner-mounted lidar_ne / lidar_nw vs. center
    # lidar_cc) projects each sweep into a range image that no longer
    # resembles KITTI training data, so MF-MOS predictions on those LiDARs
    # are unreliable. None = no restriction (run on every LiDAR seen in the
    # chunk). To restrict, list lidar_ids: e.g. ["lidar_cc"] runs only the
    # center LiDAR.
    lidar_id_allowlist: Optional[list[str]] = None

    @field_validator("residual_steps")
    @classmethod
    def _residuals_positive(cls, v: list[int]) -> list[int]:
        if any(k <= 0 for k in v):
            raise ValueError(f"residual_steps must all be > 0, got {v}")
        if len(v) != len(set(v)):
            raise ValueError(f"residual_steps must be unique, got {v}")
        return sorted(v)

    @field_validator("fusion_mode")
    @classmethod
    def _fusion_mode_valid(cls, v: str) -> str:
        valid = {"independent", "union", "mfmos_only"}
        if v not in valid:
            raise ValueError(f"fusion_mode must be one of {valid}, got {v!r}")
        return v

    @field_validator("score_threshold")
    @classmethod
    def _threshold_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"score_threshold must be in [0, 1], got {v}")
        return v


class ComponentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Step A — deskew filter.
    filter_nonfinite_points: bool = True

    # Step A — rolling-shutter motion compensation.
    # When a raw sweep NPZ has no per-point timestamp field (t_offset_us),
    # deskew falls back to using the header timestamp for all points,
    # which leaves intra-sweep ego motion uncompensated.  For a 20 Hz
    # NuScenes LiDAR at 7 m/s, that's ~25 cm of position smear per sweep
    # — wider than typical voxel sizes — which spreads statics across
    # multiple voxels and causes the "buildings leak into dynamic_map.npz"
    # symptom.  With this enabled, deskew synthesizes per-point times from
    # each point's azimuth (atan2(y, x)) assuming uniform rotation.
    synthesize_per_point_times: bool = True
    # Sweep duration in milliseconds.  Velodyne HDL-32E default is 10 Hz
    # (100 ms); NuScenes uses the same hardware at 20 Hz (50 ms).
    lidar_sweep_duration_ms: float = 100.0
    # Rotation direction.  "auto" (default) detects from the data by
    # comparing the first and ~200th point's azimuths.  Force "ccw" or
    # "cw" only if auto-detect picks wrong (e.g., very short sweeps).
    # Note: NuScenes LIDAR_TOP scans CW (azimuth decreases with time)
    # in the lidar frame, contrary to the common "Velodyne is CCW"
    # assumption — the auto-detect handles both.
    lidar_rotation_dir: Literal["ccw", "cw", "auto"] = "auto"

    # Strictness flags: if a critical input is missing, the pipeline raises
    # by default instead of silently degrading.  Each flag is an explicit
    # opt-out for users who understand the consequences.
    #
    # require_patchwork: when True (default), deskew raises if pypatchworkpp
    # is not importable.  Without Patchwork, per-sweep ground extraction is
    # skipped, ground points pollute static_map.npz AND dynamic_map.npz.
    require_patchwork: bool = True
    #
    # allow_uncompensated_motion: when False (default), deskew raises if a
    # sweep has no per-point timestamps AND synthesize_per_point_times is
    # False.  Without intra-sweep motion compensation, statics smear across
    # voxels and leak into dynamic_map.npz (the "buildings as dynamic" bug).
    allow_uncompensated_motion: bool = False

    # Step B — voxel classification.
    voxel_size_m: float = 0.15
    classification_method: Literal["log_odds", "persistence"] = "log_odds"

    # Persistence-counting parameters (used when classification_method="persistence").
    static_sweep_fraction: float = 0.3
    static_sweep_min: int = 5

    # Log-odds ray-casting parameters (used when classification_method="log_odds").
    l_occ: float = 0.85
    l_free: float = 0.40
    # Symmetric clamp on log_odds (±value).  Must be MUCH larger than l_occ
    # so well-observed statics can build a healthy margin against carving:
    # at clamp=5.0 with l_occ=1.20, the +5 ceiling is reached after only
    # ~4 hits — every additional hit is wasted, but carves keep subtracting
    # at -l_free each, so a heavily-observed wall + a moderate number of
    # through-rays drops below 0 and gets classified dynamic.  50.0 gives
    # well-observed voxels ~40+ hits of "headroom" before clamping, which
    # is enough to survive realistic pose-drift / occlusion carving.
    log_odds_clamp: float = 50.0
    p_static_threshold: float = 0.7
    p_dynamic_threshold: float = 0.3
    min_observations: int = 3
    # Voxels with fewer endpoint hits than this are free-space-only; not dynamic.
    min_occupied_hits: int = 1
    # Default raised 80 → 200 m to pair with range-credibility weighting.
    # With use_range_weighted_log_odds=False, long rays still contribute full
    # weight, so users who want the historical "drop noisy distant evidence"
    # behaviour should set this back down to 50–80 m in yaml.
    max_ray_length_m: float = 200.0
    free_space_margin_voxels: float = 1.0
    # Range-credibility weighting (UniLiPs r*_j extension, Option B):
    #   when use_range_weighted_log_odds=True, every log-odds update is
    #   multiplied by min(1, r_max_credibility_m / d), where d is the
    #   distance from the sensor to that update site:
    #     - endpoint  (l_occ):  d = sweep ray length
    #     - traversed (l_free): d = parametric distance to that voxel's
    #                              near edge along the ray
    #   This down-weights endpoints AND distant carving while leaving
    #   near-field carving at full weight on long rays (the principled
    #   extension of UniLiPs Eq. 2 to ray-traversal log-odds).
    # Opt-in (default False) so existing pipelines reproduce exactly.
    use_range_weighted_log_odds: bool = False
    r_max_credibility_m: float = 200.0
    # "skip_endpoint": traverse ground rays for free-space evidence, skip l_occ at endpoint.
    # "skip_ray": skip ground rays entirely (legacy).
    ground_endpoint_strategy: Literal["skip_endpoint", "skip_ray"] = "skip_endpoint"

    cache_world_xyz_in_memory: bool = True

    # Step D — global static map reduce.
    global_map_voxel_size_m: float = 0.30

    # Per-point timestamp unit stored by ingest's t_offset_us field.
    # Velodyne "t" field is in seconds; some lidars use microseconds.
    # Options: "seconds" | "microseconds" | "nanoseconds"
    point_time_unit: str = "seconds"

    # Step C — Patchwork++ parameters.
    patchwork: PatchworkParams = PatchworkParams()

    # Multi-lidar frame grouping (SAM4D alignment).
    frame_sync: FrameSyncParams = FrameSyncParams()

    # Step A.5 — MF-MOS learned moving-object segmentation.
    mf_mos: MFMosParams = MFMosParams()
    # Voxel-level MF-MOS vote aggregation thresholds.
    # A voxel is considered MF-MOS-dynamic if it received at least
    # min_mf_mos_votes votes AND the vote fraction >= mf_mos_vote_fraction_threshold.
    mf_mos_vote_fraction_threshold: float = 0.5
    min_mf_mos_votes: int = 1

    # SAM4D/MinkUNet alignment: export binary voxel occupancy alongside
    # static_map.npz.  Includes ALL occupied voxels (static + dynamic), not
    # just static ones — that's what the MinkUNet encoder consumes.
    save_voxel_occupancy: bool = True

    # Debugging: also export voxel_diag.npz with per-voxel log_odds, n_obs,
    # n_hits, and classification label for EVERY classified voxel (including
    # carved ones with log_odds < 0 that voxel_occupancy.npz filters out).
    # Powers the viz's p_occ color mode for the dynamic cloud.  Adds a few
    # MB per chunk; safe to leave on during development.
    save_voxel_diagnostics: bool = False

    # Per-frame voxel occupancy for SAM4D's MinkUNet encoder.  When true,
    # writes one voxel_occupancy_frame_NNNN.npz per frame_id in the chunk
    # (all sweeps sharing that frame_id merged into one occupancy grid).
    # This is what perception_2d actually feeds to MinkUNet — the per-chunk
    # voxel_occupancy.npz aggregates all frames and is only useful for QA.
    # Disabled by default; enable once perception_2d development starts.
    save_per_frame_voxel_occupancy: bool = False

    upstream_versions: dict[str, str] = Field(default_factory=dict)

    @field_validator("voxel_size_m", "global_map_voxel_size_m", "max_ray_length_m")
    @classmethod
    def _positive_voxel(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"value must be > 0, got {v}")
        return v

    @field_validator("l_occ", "l_free")
    @classmethod
    def _positive_log_odds_param(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"log-odds weight must be > 0, got {v}")
        return v

    @field_validator("r_max_credibility_m")
    @classmethod
    def _positive_r_max(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"r_max_credibility_m must be > 0, got {v}")
        return v

    @field_validator("p_static_threshold", "p_dynamic_threshold")
    @classmethod
    def _probability_range(cls, v: float) -> float:
        if not (0 < v < 1):
            raise ValueError(f"probability threshold must be in (0, 1), got {v}")
        return v

    @field_validator("min_observations", "min_occupied_hits")
    @classmethod
    def _positive_min_obs(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"value must be >= 1, got {v}")
        return v

    @field_validator("min_mf_mos_votes")
    @classmethod
    def _positive_min_mf_mos_votes(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"min_mf_mos_votes must be >= 1, got {v}")
        return v

    @field_validator("static_sweep_fraction")
    @classmethod
    def _fraction_range(cls, v: float) -> float:
        if not (0 < v <= 1):
            raise ValueError(f"static_sweep_fraction must be in (0, 1], got {v}")
        return v

    @field_validator("static_sweep_min")
    @classmethod
    def _positive_min(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"static_sweep_min must be >= 1, got {v}")
        return v

    def point_time_scale_to_ns(self) -> float:
        """Multiplier to convert t_offset_us values to nanoseconds."""
        scales = {"seconds": 1e9, "microseconds": 1e3, "nanoseconds": 1.0}
        if self.point_time_unit not in scales:
            raise ValueError(
                f"point_time_unit must be one of {list(scales)}, got {self.point_time_unit!r}"
            )
        return scales[self.point_time_unit]


def load_config(path: str) -> ComponentConfig:
    with open(path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    section = data.get("lidar_preprocessing", {})
    return ComponentConfig.model_validate(section)
