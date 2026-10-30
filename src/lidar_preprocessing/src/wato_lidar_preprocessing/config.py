"""Pydantic-loaded config for lidar_preprocessing.

Every physical constant — log-odds increments, decision thresholds, carve
geometry, scan rate, FoV, beam count, intensity scale — is derived from the
datasheet SensorModel selected by ``sensor_model`` (see sensor_model.py).
YAML picks scanners and states the few genuinely free choices (voxel size,
evidence count, which stages run). It does not state physics.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from wato_lidar_preprocessing.sensor_model import SensorModel, get_sensor_model


class SensorModelParams(BaseModel):
    """Selects a datasheet scanner profile, optionally one per lidar_id.

    ``profile`` is the rig default; ``per_lidar`` overrides it for a mixed rig
    (WATO: VLP-32C centre, two VLP-16 corners), so per-sweep work uses each
    scanner's own geometry. The chunk-level decision constants are identical
    across profiles by construction (sensor_model.py).
    """

    model_config = ConfigDict(extra="forbid")

    profile: str = "vlp32c"
    per_lidar: dict[str, str] = Field(default_factory=dict)

    @field_validator("profile")
    @classmethod
    def _known_profile(cls, v: str) -> str:
        get_sensor_model(v)  # raises ValueError listing valid profiles
        return v

    @field_validator("per_lidar")
    @classmethod
    def _known_per_lidar(cls, v: dict[str, str]) -> dict[str, str]:
        for lidar_id, prof in v.items():
            try:
                get_sensor_model(prof)
            except ValueError as exc:
                raise ValueError(f"per_lidar[{lidar_id!r}]: {exc}") from None
        return v

    def build(self, lidar_id: Optional[str] = None) -> SensorModel:
        """The profile for ``lidar_id``, falling back to the rig default."""
        if lidar_id is not None and lidar_id in self.per_lidar:
            return get_sensor_model(self.per_lidar[lidar_id])
        return get_sensor_model(self.profile)


class FrameSyncParams(BaseModel):
    """Multi-lidar sweep-to-frame grouping.

    A "frame" is one tick of canonical_lidar; non-canonical sweeps within
    ±tolerance_ms inherit its frame_id. canonical_lidar=None numbers each
    lidar's sweeps sequentially (single-lidar bags).
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


class DynamicFilterParams(BaseModel):
    """Step B — cleanup of the voxel verdict's dynamic points (classify/dynamic_filter.py).

    Acts on dynamic_mask / dynamic_map only: the voxel-key sets keep the raw
    verdict, so Step F's recall bits see everything this removes. Tuning:
    docs/lidar_preprocessing_measurements.md.
    """

    model_config = ConfigDict(extra="forbid")

    # A frame's dynamic points must form a connected cluster of at least
    # min_cluster_pts (cluster_voxel_m grid). Isolated returns are mixed
    # pixels and foliage speckle. 0 disables.
    min_cluster_pts: int = 5
    cluster_voxel_m: float = 0.5
    # Drop points whose voxel (persistence_voxel_m grid) holds dynamic points
    # across this much sweep time, however many scanners. Structure dwells;
    # movers pass through. Below ~0.4 s it starts cutting real movers.
    # Converted to a sweep count with the chunk's sweep rate. 0.0 disables.
    persistence_max_s: float = 0.5
    persistence_voxel_m: float = 0.5

    @field_validator("min_cluster_pts")
    @classmethod
    def _nonneg_pts(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"min_cluster_pts must be >= 0, got {v}")
        return v

    @field_validator("persistence_max_s")
    @classmethod
    def _nonneg_duration(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"persistence_max_s must be >= 0, got {v}")
        return v

    @field_validator("cluster_voxel_m", "persistence_voxel_m")
    @classmethod
    def _positive_voxel(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"voxel size must be > 0, got {v}")
        return v


class IWUParams(BaseModel):
    """Step E — UniLiPs IWU over the bag static map.

    The paper's constants live in iwu/_core.py; the match radius is
    global_map_voxel_size_m.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # Updates per second per lidar (stride = round(datasheet rate / this)).
    # The EMA is dominated by its last ~10 updates, so 20 Hz is redundant.
    update_rate_hz: float = 4.0

    @field_validator("update_rate_hz")
    @classmethod
    def _positive_rate(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"update_rate_hz must be > 0, got {v}")
        return v


class MotionProposalParams(BaseModel):
    """Step F — recall-oriented moving-object proposals (Chen et al. 2022)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # Points lower than this over the ground grid are never flagged (road
    # voxels are never classified static, so road false positives survive
    # every voxel test); BOX_FILL restores wheels/feet. 0.0 disables.
    min_height_above_ground_m: float = 0.25
    # Chen N_min — smallest cluster kept (HDBSCAN min_cluster_size).
    min_cluster_pts: int = 5
    # Chen T_size — clusters whose longest BEV side exceeds this are structure.
    max_side_m: float = 20.0
    # Chen box fill: points inside a moving cluster's box get BOX_FILL.
    box_fill: bool = True
    # frac_persistent: a cluster member is persistent when its voxel holds
    # members across >= this many sweeps. A sweep COUNT, tuned with
    # BOX_FILL's frac_persistent gate; scale with sensor rate. 0 disables.
    persistence_max_sweeps: int = 20
    persistence_voxel_m: float = 0.5

    @field_validator("min_height_above_ground_m")
    @classmethod
    def _nonneg_height(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"min_height_above_ground_m must be >= 0, got {v}")
        return v

    @field_validator("min_cluster_pts")
    @classmethod
    def _min_cluster(cls, v: int) -> int:
        if v < 2:  # HDBSCAN's floor
            raise ValueError(f"min_cluster_pts must be >= 2, got {v}")
        return v

    @field_validator("max_side_m")
    @classmethod
    def _positive_side(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"max_side_m must be > 0, got {v}")
        return v


class ComponentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sensor_model: SensorModelParams = SensorModelParams()

    # Step A — deskew filter.
    filter_nonfinite_points: bool = True

    # Step A — when a sweep has no usable per-point times, synthesize them
    # from azimuth (rotation period and direction from the sensor profile).
    # Without it every point gets the header pose and statics smear.
    synthesize_per_point_times: bool = True

    # Strictness flags — fail loudly on missing inputs rather than degrade.
    require_patchwork: bool = True
    allow_uncompensated_motion: bool = False

    # Step B — voxel classification.
    voxel_size_m: float = 0.15

    # Ray traversals a voxel needs before its occupancy is trusted; below it a
    # voxel is UNDER_EVIDENCED (neither static nor dynamic). A voxel with no
    # endpoint hits is FREE_ONLY and never dynamic, whatever this is.
    min_observations: int = 3

    # Points within this horizontal range (m) of the sensor are never dynamic
    # (ego body, near clutter, maximal carving). 0.0 disables.
    dynamic_min_range_m: float = 2.5

    # Step B — cleanup of the dynamic points the voxel verdict leaves.
    dynamic_filter: DynamicFilterParams = DynamicFilterParams()

    cache_world_xyz_in_memory: bool = True

    # Step D — global static map voxel. Also the match radius of everything
    # compared against that map (two-pass prior, IWU, Step F): reduce snaps
    # points to voxel centres.
    global_map_voxel_size_m: float = 0.30

    # Unit of ingest's t_offset_us field.
    # Options: "seconds" | "microseconds" | "nanoseconds"
    point_time_unit: str = "seconds"

    # Where the header stamp sits in the rotation, for synthesized per-point
    # times: "sweep_start" = first-fired point, "sweep_end" = last-fired. A
    # driver property (Velodyne `timestamp_first_packet`), set per profile.
    # Unused when the sweep has real per-point times.
    header_stamp_at: Literal["sweep_start", "sweep_end"] = "sweep_start"

    # Step C — Patchwork++ parameters.
    patchwork: PatchworkParams = PatchworkParams()

    frame_sync: FrameSyncParams = FrameSyncParams()

    # Step E — bag-level UniLiPs IWU.
    iwu: IWUParams = IWUParams()

    # Step F — per-chunk recall-oriented motion proposals.
    motion_proposals: MotionProposalParams = MotionProposalParams()

    # voxel_occupancy.npz: every occupied voxel, static and dynamic.
    save_voxel_occupancy: bool = True
    # voxel_diag.npz: per-voxel log_odds/n_obs/n_hits/classification, carved
    # voxels included. Powers viz's p_occ color mode.
    save_voxel_diagnostics: bool = False
    # One voxel_occupancy_frame_NNNN.npz per frame_id. No consumer yet.
    save_per_frame_voxel_occupancy: bool = False

    @field_validator("voxel_size_m", "global_map_voxel_size_m")
    @classmethod
    def _positive_voxel(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"value must be > 0, got {v}")
        return v

    @field_validator("min_observations")
    @classmethod
    def _positive_min_obs(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"min_observations must be >= 1, got {v}")
        return v

    @field_validator("dynamic_min_range_m")
    @classmethod
    def _nonneg_dyn_range(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"dynamic_min_range_m must be >= 0, got {v}")
        return v

    def point_time_scale_to_ns(self) -> float:
        """Multiplier to convert t_offset_us values to nanoseconds."""
        scales = {"seconds": 1e9, "microseconds": 1e3, "nanoseconds": 1.0}
        if self.point_time_unit not in scales:
            raise ValueError(
                f"point_time_unit must be one of {list(scales)}, got {self.point_time_unit!r}"
            )
        return scales[self.point_time_unit]

    def build_sensor_model(self, lidar_id: Optional[str] = None) -> SensorModel:
        """The datasheet SensorModel for ``lidar_id`` (default: the rig's)."""
        return self.sensor_model.build(lidar_id)


def load_config(path: str) -> ComponentConfig:
    with open(path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    section = data.get("lidar_preprocessing", {})
    return ComponentConfig.model_validate(section)
