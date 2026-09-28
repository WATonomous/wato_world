"""Pydantic-loaded config for lidar_preprocessing.

Classifier constants are derived from the datasheet SensorModel selected by
sensor_model.profile (see sensor_model.py), not configured as raw numbers.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from wato_lidar_preprocessing.sensor_model import SensorModel, get_sensor_model


_SAFE_LIDAR_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


class SensorModelParams(BaseModel):
    """Selects the datasheet sensor profile. The physical numbers live in
    sensor_model.py's profile table, not in user YAML."""

    model_config = ConfigDict(extra="forbid")

    # "velodyne_vlp" (WATO 3-Velodyne rig) | "nuscenes" (HDL-32E-class).
    profile: str = "velodyne_vlp"

    @field_validator("profile")
    @classmethod
    def _known_profile(cls, v: str) -> str:
        get_sensor_model(v)  # raises ValueError listing valid profiles
        return v

    def build(self) -> SensorModel:
        return get_sensor_model(self.profile)


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
    # Occlusion gate for unprojecting the per-pixel moving mask back to points.
    occlusion_range_tol_m: float = 1.0
    # Seed each lidar's residual sliding window from the temporally-preceding
    # chunk's sweeps so the first max(residual_steps) sweeps of a chunk get
    # full residual channels instead of cold-start zeros.
    prime_window_from_prior_chunk: bool = True

    # --- Per-sweep spatial denoise (replaces the chunk-wide vote tier) ---
    # MF-MOS speckle is removed spatially per sweep: cluster moving points on a
    # 26-connected 3D grid and drop clusters below the size floor. Temporal
    # confirmation of a mover is the downstream tracker's job.
    # Cluster grid resolution [m] (~2× voxel keeps an object connected).
    moving_cluster_voxel_m: float = 0.5
    # Min points per moving cluster (a pedestrian at MF-MOS range is well above
    # this; single-sweep mispredictions are 1–few points).
    moving_min_cluster_pts: int = 8

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

    @field_validator("occlusion_range_tol_m", "moving_cluster_voxel_m")
    @classmethod
    def _positive_float(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"value must be > 0, got {v}")
        return v

    @field_validator("moving_min_cluster_pts")
    @classmethod
    def _positive_min_cluster(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"moving_min_cluster_pts must be >= 1, got {v}")
        return v


class MFMosProfile(BaseModel):
    """Sensor-specific MF-MOS projection and fusion contract."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    fusion_mode: Literal["independent", "union", "mfmos_only"] = "independent"
    range_image_h: int = 32
    range_image_w: int = 1024
    fov_up_deg: float = 10.0
    fov_down_deg: float = -30.0
    min_range_m: float = 2.0
    max_range_m: float = 50.0
    intensity_scale: float = 255.0
    occlusion_range_tol_m: float = 1.0

    @field_validator("range_image_h", "range_image_w")
    @classmethod
    def _positive_image_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"range-image dimensions must be >= 1, got {v}")
        return v

    @field_validator(
        "min_range_m", "max_range_m", "intensity_scale", "occlusion_range_tol_m"
    )
    @classmethod
    def _positive_geometry_value(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"MF-MOS geometry values must be > 0, got {v}")
        return v

    @model_validator(mode="after")
    def _valid_range_interval(self) -> "MFMosProfile":
        if self.max_range_m <= self.min_range_m:
            raise ValueError("max_range_m must be greater than min_range_m")
        return self


class LidarProfile(BaseModel):
    """Physical parameters that must be resolved independently per LiDAR."""

    model_config = ConfigDict(extra="forbid")

    sensor_model: SensorModelParams = Field(default_factory=SensorModelParams)
    lidar_sweep_duration_ms: float = 100.0
    point_time_unit: Literal["seconds", "microseconds", "nanoseconds"] = "seconds"
    patchwork_sensor_height: float = 1.8
    mf_mos: MFMosProfile = Field(default_factory=MFMosProfile)

    @field_validator("lidar_sweep_duration_ms", "patchwork_sensor_height")
    @classmethod
    def _positive_physical_value(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"LiDAR physical values must be > 0, got {v}")
        return v

    def point_time_scale_to_ns(self) -> float:
        return {
            "seconds": 1e9,
            "microseconds": 1e3,
            "nanoseconds": 1.0,
        }[self.point_time_unit]

    def build_sensor_model(self) -> SensorModel:
        return self.sensor_model.build()


class ComponentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # --- Sensor model: source of all derived classifier constants ----------
    sensor_model: SensorModelParams = SensorModelParams()

    # Step A — deskew filter.
    filter_nonfinite_points: bool = True

    # Step A — rolling-shutter motion compensation. When raw NPZ has no
    # per-point timestamps (t_offset_us), this synthesizes them from each
    # point's azimuth assuming uniform rotation. Disabling = all points
    # share the header pose → intra-sweep smear that spreads statics across
    # voxels and leaks them into dynamic_map.
    synthesize_per_point_times: bool = True
    # Velodyne VLP @ 10 Hz = 100 ms; NuScenes @ 20 Hz = 50 ms.
    lidar_sweep_duration_ms: float = 100.0
    # Scan rotation direction comes from the sensor_model profile.

    # Strictness flags — fail loudly on missing inputs rather than degrade.
    require_patchwork: bool = True
    allow_uncompensated_motion: bool = False

    # Step B — voxel classification.
    voxel_size_m: float = 0.15

    # Evidence gates (statistical, sensor-independent).
    # Voxels observed fewer than min_observations times stay UNKNOWN.
    min_observations: int = 3
    # Voxels with fewer endpoint hits go to free_only (never dynamic).
    min_occupied_hits: int = 1

    # Optional hard cap on ray length [m]. None → sensor profile's max_range_m
    # (a compute guard; far-field carving noise is handled by range weighting).
    max_ray_length_m: Optional[float] = None

    # "skip_endpoint" → traverse ground rays for free-space evidence, no l_occ at endpoint.
    # "skip_ray"      → skip ground rays entirely (legacy).
    ground_endpoint_strategy: Literal["skip_endpoint", "skip_ray"] = "skip_endpoint"

    cache_world_xyz_in_memory: bool = True

    # Step D — global static map reduce.
    global_map_voxel_size_m: float = 0.30

    # Two-pass global map prior (UniLiPs IWU): KDTree match radius, >=
    # global_map_voxel_size_m (reduce snaps to voxel centres). The prior's
    # strength + range weighting are derived from the sensor model.
    global_map_match_radius_m: float = 0.30

    # Unit of ingest's t_offset_us field.
    # Options: "seconds" | "microseconds" | "nanoseconds"
    point_time_unit: str = "seconds"

    # Step C — Patchwork++ parameters.
    patchwork: PatchworkParams = PatchworkParams()

    # Multi-lidar frame grouping (SAM4D alignment).
    frame_sync: FrameSyncParams = FrameSyncParams()

    # Step A.5 — MF-MOS learned moving-object segmentation.
    mf_mos: MFMosParams = MFMosParams()

    # Explicit profiles make sensor membership strict unless a default is
    # supplied. With neither field set, legacy global fields form the implicit
    # default profile so existing configuration files remain loadable.
    default_lidar_profile: Optional[LidarProfile] = None
    lidar_profiles: dict[str, LidarProfile] = Field(default_factory=dict)

    # voxel_occupancy.npz alongside static_map.npz. Includes ALL occupied
    # voxels (static + dynamic) — that's what MinkUNet consumes.
    save_voxel_occupancy: bool = True

    # voxel_diag.npz with per-voxel log_odds/n_obs/n_hits/classification for
    # EVERY classified voxel, including carved (log_odds < 0) ones that
    # voxel_occupancy.npz filters out. Powers viz's p_occ color mode.
    save_voxel_diagnostics: bool = False

    # One voxel_occupancy_frame_NNNN.npz per frame_id.
    save_per_frame_voxel_occupancy: bool = False

    @field_validator("voxel_size_m", "global_map_voxel_size_m")
    @classmethod
    def _positive_voxel(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"value must be > 0, got {v}")
        return v

    @field_validator("max_ray_length_m")
    @classmethod
    def _positive_ray_length(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError(f"max_ray_length_m must be > 0 or null, got {v}")
        return v

    @field_validator("min_observations", "min_occupied_hits")
    @classmethod
    def _positive_min_obs(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"value must be >= 1, got {v}")
        return v

    @field_validator("lidar_profiles")
    @classmethod
    def _safe_lidar_profile_keys(
        cls, profiles: dict[str, LidarProfile]
    ) -> dict[str, LidarProfile]:
        invalid = [
            lidar_id
            for lidar_id in profiles
            if not _SAFE_LIDAR_ID.fullmatch(lidar_id)
        ]
        if invalid:
            raise ValueError(
                "lidar_id profile keys must match [A-Za-z0-9_.-]+; "
                f"got {invalid!r}"
            )
        return profiles

    @model_validator(mode="after")
    def _compatible_aw_profile_contracts(self) -> "ComponentConfig":
        profiles = list(self.lidar_profiles.values())
        if self.default_lidar_profile is not None:
            profiles.append(self.default_lidar_profile)
        contracts = {
            (
                profile.build_sensor_model().p_hit,
                profile.build_sensor_model().p_miss,
                profile.build_sensor_model().p_clamp,
                profile.build_sensor_model().p_map_prior,
            )
            for profile in profiles
        }
        if len(contracts) > 1:
            raise ValueError(
                "all LiDAR profiles sharing an AW grid must use the same "
                "inverse-occupancy probabilities and decision thresholds"
            )
        return self

    def _legacy_lidar_profile(self) -> LidarProfile:
        return LidarProfile(
            sensor_model=self.sensor_model,
            lidar_sweep_duration_ms=self.lidar_sweep_duration_ms,
            point_time_unit=self.point_time_unit,
            patchwork_sensor_height=self.patchwork.sensor_height,
            mf_mos=MFMosProfile(
                enabled=self.mf_mos.enabled,
                fusion_mode=self.mf_mos.fusion_mode,
                range_image_h=self.mf_mos.range_image_h,
                range_image_w=self.mf_mos.range_image_w,
                fov_up_deg=self.mf_mos.fov_up_deg,
                fov_down_deg=self.mf_mos.fov_down_deg,
                min_range_m=self.mf_mos.min_range_m,
                max_range_m=self.mf_mos.max_range_m,
                intensity_scale=self.mf_mos.intensity_scale,
                occlusion_range_tol_m=self.mf_mos.occlusion_range_tol_m,
            ),
        )

    def profile_for(self, lidar_id: str) -> LidarProfile:
        """Resolve one physical LiDAR without treating frame_id as identity."""
        if not _SAFE_LIDAR_ID.fullmatch(lidar_id):
            raise ValueError(
                "lidar_id must match [A-Za-z0-9_.-]+; " f"got {lidar_id!r}"
            )
        if lidar_id in self.lidar_profiles:
            return self.lidar_profiles[lidar_id]
        if self.default_lidar_profile is not None:
            return self.default_lidar_profile
        if self.lidar_profiles:
            raise ValueError(
                f"unknown lidar_id {lidar_id!r}; configured sensors: "
                f"{sorted(self.lidar_profiles)}"
            )
        return self._legacy_lidar_profile()

    def patchwork_for(self, lidar_id: str) -> PatchworkParams:
        return self.patchwork.model_copy(
            update={"sensor_height": self.profile_for(lidar_id).patchwork_sensor_height}
        )

    def point_time_scale_to_ns(self) -> float:
        """Multiplier to convert t_offset_us values to nanoseconds."""
        scales = {"seconds": 1e9, "microseconds": 1e3, "nanoseconds": 1.0}
        if self.point_time_unit not in scales:
            raise ValueError(
                f"point_time_unit must be one of {list(scales)}, got {self.point_time_unit!r}"
            )
        return scales[self.point_time_unit]

    def build_sensor_model(self) -> SensorModel:
        """The datasheet SensorModel selected by sensor_model.profile."""
        return self.sensor_model.build()

    def effective_max_ray_length_m(self) -> float:
        """Compute guard: explicit override, else the sensor's max range."""
        if self.max_ray_length_m is not None:
            return float(self.max_ray_length_m)
        return self.build_sensor_model().max_range_m


def load_config(path: str) -> ComponentConfig:
    with open(path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    section = data.get("lidar_preprocessing", {})
    return ComponentConfig.model_validate(section)
