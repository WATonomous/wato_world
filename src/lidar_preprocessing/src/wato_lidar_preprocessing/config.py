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

    The physical numbers live in sensor_model.py's profile table, never in
    user YAML. ``profile`` is the rig default; ``per_lidar`` overrides it for
    a mixed rig (the WATO car runs a VLP-32C centre and two VLP-16 corners),
    so each scanner's own FoV, beam count, range and rate are used wherever
    the work is per-sweep.

    The default profile also supplies the chunk-level decision constants
    (l_occ, l_free, thresholds), which are shared across profiles by
    construction — see sensor_model.py — so a mixed rig does not classify
    one lidar's returns by a different rule than another's.
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

    The spherical-projection geometry (range-image height, vertical FoV,
    intensity scale, residual spacing) is NOT configured here — it is read
    from each lidar's sensor profile, and the checkpoint-side constants
    (image width, training range window, KITTI reference rate) live in
    mf_mos/_core.py. What remains below is genuinely about running the model:
    where its weights are, how confident a pixel must be, and how the output
    is cleaned and fused.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    checkpoint_path: str = "/data/models/mf_mos/mf_mos_semantic_kitti.pt"
    arch_config: str = "/data/models/mf_mos/arch_cfg.yaml"
    data_config: str = "/data/models/mf_mos/data_cfg.yaml"
    device: str = "cuda"
    score_threshold: float = 0.5
    save_scores: bool = False
    # How classify uses the MF-MOS mask:
    #   independent — masks written side-by-side; the voxel classifier alone
    #                 decides dynamic_map.npz. Downstream may read both.
    #   union       — a point is dynamic if the voxel classifier OR this
    #                 sweep's MF-MOS voxel set says so. A missing or empty
    #                 mask leaves the classifier's verdict untouched.
    # There is deliberately no "mfmos_only": handing the entire verdict to a
    # model that silently emits nothing on a skipped sweep empties
    # dynamic_map.npz with no failure. Use union and read the skip warning.
    fusion_mode: Literal["independent", "union"] = "independent"
    max_pose_gap_ms: float = 200.0
    # Occlusion gate for unprojecting the per-pixel moving mask back to points.
    occlusion_range_tol_m: float = 1.0
    # Seed each lidar's residual sliding window from the temporally-preceding
    # chunk's sweeps so the first sweeps of a chunk get full residual channels
    # instead of cold-start zeros.
    prime_window_from_prior_chunk: bool = True

    # --- Per-sweep spatial denoise (replaces the chunk-wide vote tier) ---
    # MF-MOS speckle is removed spatially per sweep: cluster moving points on a
    # 26-connected 3D grid and drop clusters below the size floor. Temporal
    # confirmation of a mover is the downstream tracker's job.
    # Cluster grid resolution [m] (~2x voxel keeps an object connected).
    moving_cluster_voxel_m: float = 0.5
    # Min points per moving cluster (a pedestrian at MF-MOS range is well above
    # this; single-sweep mispredictions are 1-few points).
    moving_min_cluster_pts: int = 8

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
    # Rotation period and spin direction come from each lidar's sensor
    # profile (sweep_duration_ms / rotation_dir) — they are datasheet facts.

    # Strictness flags — fail loudly on missing inputs rather than degrade.
    require_patchwork: bool = True
    allow_uncompensated_motion: bool = False

    # Step B — voxel classification.
    voxel_size_m: float = 0.15

    # The one evidence gate: how many ray traversals a voxel needs before its
    # occupancy probability is trusted at all. Statistical, not physical — it
    # trades recall on sparsely-seen structure against noise from single
    # observations. Voxels below it are UNDER_EVIDENCED: neither static nor
    # dynamic. A voxel with zero endpoint hits is FREE_ONLY and can never be
    # dynamic regardless (that rule needs no threshold: "was anything ever
    # measured here?" is a yes/no question).
    min_observations: int = 3

    cache_world_xyz_in_memory: bool = True

    # Step D — global static map reduce. The two-pass prior's KDTree match
    # radius is this same value: reduce snaps map points to voxel centres, so
    # "within one map voxel" is exactly what a match means. Its strength and
    # range weighting are derived from the sensor model.
    global_map_voxel_size_m: float = 0.30

    # Unit of ingest's t_offset_us field.
    # Options: "seconds" | "microseconds" | "nanoseconds"
    point_time_unit: str = "seconds"

    # Step C — Patchwork++ parameters.
    patchwork: PatchworkParams = PatchworkParams()

    # Multi-lidar frame grouping (SAM4D alignment).
    frame_sync: FrameSyncParams = FrameSyncParams()

    # Step A.5 — MF-MOS learned moving-object segmentation.
    mf_mos: MFMosParams = MFMosParams()

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

    @field_validator("min_observations")
    @classmethod
    def _positive_min_obs(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"min_observations must be >= 1, got {v}")
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
