"""Pydantic-loaded config for lidar_preprocessing."""

from __future__ import annotations

from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class FrameSyncParams(BaseModel):
    """Multi-lidar sweep-to-frame grouping for SAM4D-style downstream fusion.

    A "frame" is one tick of canonical_lidar. Non-canonical sweeps within
    ±tolerance_ms inherit the canonical frame_id.

    canonical_lidar=None disables grouping — each sweep becomes its own
    frame indexed sequentially per lidar_id (correct for single-lidar bags).
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

    Defaults mirror wato_monorepo perception/patchwork/patchwork/config/params.yaml.
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

    Step A.5 between deskew and classify when enabled. Requires a CUDA GPU
    for realistic data; device="cpu" is for tiny smoke tests only.
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
    # fusion_mode controls how classify uses the MF-MOS mask:
    #   independent — masks written side-by-side, downstream decides.
    #   union       — static/dynamic accumulators use (voxel | mf_mos).
    #   mfmos_only  — accumulators use the mf_mos mask only.
    fusion_mode: str = "independent"
    max_pose_gap_ms: float = 200.0
    # Match training preprocessing (data_preparing.yaml).
    min_range_m: float = 2.0
    max_range_m: float = 50.0
    # Divisor that scales raw intensity into [0, 1] like KITTI remission.
    # NuScenes intensity is uint8 [0, 255] → 255.0. Use 1.0 for sensors
    # that already produce [0, 1].
    intensity_scale: float = 255.0
    # Restrict MF-MOS to specific lidar_ids. fov_up/down + image H/W are
    # global, so LiDARs with significantly different mount geometry would
    # project into non-KITTI-like range images and the model mispredicts.
    # None = run on every LiDAR; e.g. ["lidar_cc"] = centre only.
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

    # Step A — rolling-shutter motion compensation. When raw NPZ has no
    # per-point timestamps (t_offset_us), this synthesizes them from each
    # point's azimuth assuming uniform rotation. Disabling = all points
    # share the header pose → ~25cm smear at 7 m/s ego on a 50ms sweep,
    # which spreads statics across voxels and leaks them into dynamic_map.
    synthesize_per_point_times: bool = True
    # Velodyne HDL-32E @ 10 Hz = 100 ms; NuScenes uses same hardware @ 20 Hz = 50 ms.
    lidar_sweep_duration_ms: float = 100.0
    # "auto" detects from data by comparing phi[0] to phi[~200]. NuScenes
    # LIDAR_TOP scans CW (counter to the common "Velodyne is CCW" assumption).
    lidar_rotation_dir: Literal["ccw", "cw", "auto"] = "auto"

    # Strictness flags — defaults make the pipeline fail loudly on missing
    # inputs rather than silently degrade.
    #
    # require_patchwork=True → deskew raises if pypatchworkpp is missing.
    # Without it, ground points pollute static_map.npz AND dynamic_map.npz.
    require_patchwork: bool = True
    #
    # allow_uncompensated_motion=False → deskew raises when a sweep has no
    # per-point timestamps and synthesize_per_point_times is also False.
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
    # Must be MUCH larger than l_occ so well-observed statics can build a
    # margin against carving — at clamp=5.0 with l_occ=1.20, the +5 ceiling
    # is reached after ~4 hits and additional hits are wasted while carves
    # keep subtracting. 50.0 gives ~40+ hits of headroom.
    log_odds_clamp: float = 50.0
    p_static_threshold: float = 0.7
    p_dynamic_threshold: float = 0.3
    min_observations: int = 3
    # Voxels with fewer endpoint hits go to free_only (never dynamic).
    min_occupied_hits: int = 1
    # Default 200 m pairs with range-credibility weighting. Without that
    # weighting, long rays still contribute full weight — set to 50-80 m
    # to recover the historical "drop distant evidence" behaviour.
    max_ray_length_m: float = 200.0
    free_space_margin_voxels: float = 1.0
    # UniLiPs r*_j extension. When True, every log-odds update is scaled by
    # min(1, r_max_credibility_m / d), where d = ray length for the endpoint
    # update and d = parametric entry distance for free-space carving.
    # Down-weights endpoints AND distant carving while preserving near-field
    # carving on long rays.
    use_range_weighted_log_odds: bool = False
    r_max_credibility_m: float = 200.0
    # "skip_endpoint" → traverse ground rays for free-space evidence, no l_occ at endpoint.
    # "skip_ray"      → skip ground rays entirely (legacy).
    ground_endpoint_strategy: Literal["skip_endpoint", "skip_ray"] = "skip_endpoint"

    cache_world_xyz_in_memory: bool = True

    # Step D — global static map reduce.
    global_map_voxel_size_m: float = 0.30

    # Two-pass global map prior (UniLiPs IWU). --two-pass runs single-pass,
    # then reduce, then re-classifies every chunk using global_static_map.npz
    # as a per-sweep KDTree prior.
    #
    # Match radius for the KDTree query. Should be >= global_map_voxel_size_m
    # because reduce snaps points to voxel centres (worst-case offset is
    # voxel_size × √3/2 ≈ 0.26 m at 0.30 m voxel). Raising the global map
    # voxel size without also raising this silently breaks boundary matches.
    global_map_match_radius_m: float = 0.30
    # Per-sweep log-odds boost for map-matched endpoints. Accumulates across
    # every sweep that hits the voxel, so total ≈ l_occ_global_map * r_star
    # * n_sweeps_hitting. Keep at ~1-5% of l_occ so the prior nudges without
    # overriding real evidence. At 0.03, a voxel seen in every sweep of a
    # 30-second 20 Hz chunk accumulates ~0.9 boost.
    l_occ_global_map: float = 0.03

    # Unit of ingest's t_offset_us field.
    # Options: "seconds" | "microseconds" | "nanoseconds"
    point_time_unit: str = "seconds"

    # Step C — Patchwork++ parameters.
    patchwork: PatchworkParams = PatchworkParams()

    # Multi-lidar frame grouping (SAM4D alignment).
    frame_sync: FrameSyncParams = FrameSyncParams()

    # Step A.5 — MF-MOS learned moving-object segmentation.
    mf_mos: MFMosParams = MFMosParams()
    # Voxel-level MF-MOS aggregation: a voxel is MF-MOS-dynamic if it got
    # >= min_mf_mos_votes AND vote fraction >= mf_mos_vote_fraction_threshold.
    mf_mos_vote_fraction_threshold: float = 0.5
    min_mf_mos_votes: int = 1
    # True: a point is fused as MF-MOS-dynamic only if (a) the per-sweep mask
    # flags it AND (b) its voxel passed the chunk-wide vote.
    # False: (a) alone is enough.
    # The chunk-wide vote denoises across sweeps; the per-sweep mask prevents
    # the chunk-wide voxel set from bleeding dynamic labels back in time.
    # Log-odds path only — persistence has no chunk-wide votes.
    mf_mos_require_chunk_wide_vote: bool = True

    # voxel_occupancy.npz alongside static_map.npz. Includes ALL occupied
    # voxels (static + dynamic) — that's what MinkUNet consumes.
    save_voxel_occupancy: bool = True

    # voxel_diag.npz with per-voxel log_odds/n_obs/n_hits/classification for
    # EVERY classified voxel, including carved (log_odds < 0) ones that
    # voxel_occupancy.npz filters out. Powers viz's p_occ color mode.
    save_voxel_diagnostics: bool = False

    # One voxel_occupancy_frame_NNNN.npz per frame_id (all sweeps sharing
    # that frame_id merged). What perception_2d feeds to MinkUNet — the
    # per-chunk file is only useful for QA.
    save_per_frame_voxel_occupancy: bool = False

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
