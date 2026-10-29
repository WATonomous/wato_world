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


class MFMosParams(BaseModel):
    """MF-MOS moving-object segmentation (`--seg mos|union`; never `aw`).

    Needs a CUDA GPU for real data; device="cpu" is for smoke tests. The
    projection geometry comes from each lidar's sensor profile and the
    checkpoint's constants from mf_mos/_core.py, so only running the model is
    configured here.
    """

    model_config = ConfigDict(extra="forbid")

    checkpoint_path: str = "/data/models/mf_mos/mf_mos_semantic_kitti.pt"
    arch_config: str = "/data/models/mf_mos/arch_cfg.yaml"
    data_config: str = "/data/models/mf_mos/data_cfg.yaml"
    device: str = "cuda"
    score_threshold: float = 0.5
    save_scores: bool = False
    # Max pose-interpolation gap (ms) used when warping a historical sweep
    # into the current frame for a residual. Gates POSE quality only.
    max_pose_gap_ms: float = 200.0
    # Max sweep-to-sweep baseline (ms) accepted for a residual channel. Not
    # max_pose_gap_ms: a residual at step k spans k * sweep_dt. Steps are
    # rate-scaled to KITTI's 10 Hz (mf_mos/_core.residual_steps_for), so the
    # longest baseline is ~n_input_scans * 100 ms (800 ms for the released
    # checkpoint) on any rig; keep this above it or long channels are zeroed.
    max_residual_gap_ms: float = 1000.0
    # Occlusion gate for unprojecting the per-pixel moving mask back to points.
    occlusion_range_tol_m: float = 1.0
    # Seed each lidar's residual sliding window from the temporally-preceding
    # chunk's sweeps so the first sweeps of a chunk get full residual channels
    # instead of cold-start zeros.
    prime_window_from_prior_chunk: bool = True

    # Per-sweep denoise: cluster moving points on a 26-connected grid of this
    # cell size [m] and drop clusters under moving_min_cluster_pts (speckle is
    # 1-few points; a pedestrian at MF-MOS range is well above it). Temporal
    # confirmation is the tracker's job.
    moving_cluster_voxel_m: float = 0.5
    moving_min_cluster_pts: int = 8

    @field_validator("score_threshold")
    @classmethod
    def _threshold_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"score_threshold must be in [0, 1], got {v}")
        return v

    @field_validator(
        "occlusion_range_tol_m", "moving_cluster_voxel_m", "max_residual_gap_ms"
    )
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


class MotionFilterParams(BaseModel):
    """Post-veto temporal filter on union's accumulated dynamic cloud.

      persistence — drop a point whose voxel is occupied across
                    >= persistence_max_sweeps distinct sweeps (a mover
                    crosses a 0.5 m voxel in 1-2 sweeps; mislabelled
                    structure dwells).
      coherence   — drop a point whose per-sweep cluster doesn't link into a
                    track of >= coherence_min_life sweeps. Off by default: it
                    cuts real movers on sparse LiDAR.

    Both thresholds are sweep COUNTS: scale them with sensor rate. Tuning
    measurements: docs/lidar_preprocessing_measurements.md.
    """

    model_config = ConfigDict(extra="forbid")

    # False = write the raw post-veto cloud, to measure the filter.
    enabled: bool = True

    # 0 disables. Lower cuts more slow/large movers; higher leaks structure.
    persistence_max_sweeps: int = 20
    # Coarser than voxel_size_m so a mover's returns land in one voxel.
    persistence_voxel_m: float = 0.5

    # 0 disables (default). Membership only, not a velocity test.
    coherence_min_life: int = 0
    # Connected-components cell (m) for per-sweep clustering.
    coherence_cell_m: float = 0.4
    # Max centroid step (m) between consecutive sweeps when linking clusters.
    coherence_link_gate_m: float = 3.0
    # Clusters larger than this (m) are structure and never seed a track.
    coherence_max_object_m: float = 7.0

    @field_validator(
        "persistence_max_sweeps",
        "coherence_min_life",
    )
    @classmethod
    def _nonneg_int(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"value must be >= 0, got {v}")
        return v

    @field_validator(
        "persistence_voxel_m",
        "coherence_cell_m",
        "coherence_link_gate_m",
        "coherence_max_object_m",
    )
    @classmethod
    def _positive_float(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"value must be > 0, got {v}")
        return v


class UnionParams(BaseModel):
    """`--seg union`: AW's static map verbatim; MF-MOS's moving points, ground
    removed and vetoed by that static map, as the dynamic cloud."""

    model_config = ConfigDict(extra="forbid")

    # Drop MF-MOS-moving points whose voxel AW calls static. False = raw
    # MF-MOS dynamic (still ground-removed), to measure the veto.
    aw_static_veto: bool = True
    # Also add AW's own dynamic verdict (vetoed the same way). Off: AW
    # dynamics hug static surfaces.
    keep_aw_dynamic: bool = False
    # MF-MOS moving probability at or above which a point survives the veto
    # (an object parked most of the chunk, then driven off, has static voxel
    # evidence). Needs mf_mos.save_scores. None = no exemption.
    veto_score_exempt: float | None = None
    # Drop candidates below this height (m) over Step C's ground grid. AW
    # never calls road voxels static, so road false positives pass every
    # voxel veto. Costs wheels/feet. 0.0 disables; skipped with a warning
    # when ground.npz is missing.
    ground_height_veto_m: float = 0.25
    # Also veto voxels within this Chebyshev distance of a static voxel
    # (surface returns straddle voxel boundaries), unless AW called the
    # candidate's own voxel dynamic. 0 = exact voxel only.
    veto_dilation_voxels: int = 1
    # Removes leakage the voxel vetoes miss (sparsely mapped walls, foliage).
    motion_filter: MotionFilterParams = MotionFilterParams()

    @field_validator("veto_score_exempt")
    @classmethod
    def _exempt_range(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(f"veto_score_exempt must be in [0, 1], got {v}")
        return v

    @field_validator("ground_height_veto_m")
    @classmethod
    def _ground_height_nonneg(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError(f"ground_height_veto_m must be >= 0, got {v}")
        return v

    @field_validator("veto_dilation_voxels")
    @classmethod
    def _dilation_range(cls, v: int) -> int:
        # (2D+1)^3 - 1 neighbour lookups per candidate; D=3 is already 342.
        if not (0 <= v <= 3):
            raise ValueError(f"veto_dilation_voxels must be in [0, 3], got {v}")
        return v


class IWUParams(BaseModel):
    """Step E — UniLiPs IWU over the bag static map (any --seg).

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
    """Step F — recall-oriented moving-object proposals (Chen et al. 2022; any --seg)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # Points lower than this over the ground grid are never flagged (as
    # union.ground_height_veto_m); BOX_FILL restores wheels/feet. 0.0 disables.
    min_height_above_ground_m: float = 0.25
    # Chen N_min — smallest cluster kept (HDBSCAN min_cluster_size).
    min_cluster_pts: int = 5
    # Chen T_size — clusters whose longest BEV side exceeds this are structure.
    max_side_m: float = 20.0
    # Chen box fill: points inside a moving cluster's box get BOX_FILL.
    box_fill: bool = True

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

    # Step B static/dynamic split (`--seg` overrides):
    #   "aw"    — Amanatides-Woo log-odds ray-casting (classify/)
    #   "mos"   — MF-MOS learned segmentation (mf_mos/)
    #   "union" — both, fused (union/; the only one importing the other two)
    segmentation: Literal["aw", "mos", "union"] = "aw"

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
    # (ego body, near clutter, maximal carving). Every --seg. 0.0 disables.
    dynamic_min_range_m: float = 2.5

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

    # MF-MOS parameters (used when segmentation is "mos" or "union").
    mf_mos: MFMosParams = MFMosParams()

    # Fusion parameters (used only when segmentation == "union").
    union: UnionParams = UnionParams()

    # Step E — bag-level UniLiPs IWU (any segmentation method).
    iwu: IWUParams = IWUParams()

    # Step F — per-chunk recall-oriented motion proposals (any method).
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
