"""Pydantic-loaded config for lidar_preprocessing."""

from __future__ import annotations

from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class MapMOSFusionParams(BaseModel):
    """Knobs that govern how MapMOS logits fold into the log-odds grid.

    Shipping default `alpha = 0.0` means enabling `mapmos.enabled = true` is
    bit-safe: the additive update collapses to `+= l_occ + 0.0 * prior = +=
    l_occ` and the static/dynamic counts must equal the no-MapMOS run. This
    is the canonical regression invariant the Step 1 gate checks.
    """

    model_config = ConfigDict(extra="forbid")

    # Per-point additive prior on l_occ at every endpoint update.
    # Locked to "per_point_prior" today; the field exists so future
    # alternatives (per-voxel mean, override-only) can land without an
    # incompatible config break.
    mode: Literal["per_point_prior"] = "per_point_prior"

    # Bias scaling. SHIPPING DEFAULT IS 0.0 — see class docstring.
    alpha: float = 0.0

    # Symmetric clamp applied to logits at the read boundary in
    # classify/log_odds.py BEFORE they reach the kernel. Overshoots log a
    # warning (model-blowup signal). Plan non-negotiable #5.
    logit_clamp: float = 10.0

    # Rescue path for under-evidenced voxels (n_obs < min_observations).
    # When True AND a voxel's MAX raw logit (per the kernel's parallel
    # max_logit accumulator) crosses under_evidenced_logit_threshold, the
    # voxel drops out of not_dynamic_arr so its points can take the dynamic
    # label. Threshold is on RAW logits, independent of `alpha`.
    apply_to_under_evidenced: bool = True
    under_evidenced_logit_threshold: float = 2.0

    @field_validator("alpha", "logit_clamp", "under_evidenced_logit_threshold")
    @classmethod
    def _nonneg(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"value must be >= 0, got {v}")
        return v


class MapMOSParams(BaseModel):
    """Inference + fusion config for MapMOS (third evidence source).

    `enabled=False` by default. When False, classify never reads or writes
    sidecar logit files — geometry-only path is bit-identical to today.

    Internal model voxel size is intentionally NOT a knob here: it's locked
    to the checkpoint's recorded value (see mapmos.weights.load_and_validate)
    because the pretrained MinkUNet was trained at a specific resolution and
    changing it silently breaks the model. Plan non-negotiable #10.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False

    weights_path: str = "/data/models/mapmos/mapmos.ckpt"
    # Range filter applied at inference time, relative to ego position.
    # Source: mapmos/pipeline.py:117-121 + config/config.py:44-45
    # (MOSConfig.max_range_mos = 50.0, min_range_mos = 0.0).
    max_range_m: float = 50.0
    min_range_m: float = 0.0
    device: str = "cuda"  # falls back to cpu with a warning if cuda unavailable

    # NOTE: V1 verification (see mapmos/inference.py docstring) showed
    # MapMOS does NOT use a fixed N-past-sweeps window or time-in-seconds.
    # The time channel is the integer scan index, normalized internally
    # by MinkUNet.forward() to a feature in [1, 2]. Past context comes
    # from a running VoxelHashMap accumulator, not a windowed buffer.
    # The previously-planned `n_past_scans` and `sweep_period_s` fields
    # were removed here — they encoded the wrong convention.

    fusion: MapMOSFusionParams = MapMOSFusionParams()

    @field_validator("max_range_m")
    @classmethod
    def _positive_max_range(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"max_range_m must be > 0, got {v}")
        return v

    @field_validator("min_range_m")
    @classmethod
    def _nonneg_min_range(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"min_range_m must be >= 0, got {v}")
        return v


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


class ComponentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Step A — deskew filter.
    filter_nonfinite_points: bool = True

    # Step B — voxel classification.
    voxel_size_m: float = 0.15
    classification_method: Literal["log_odds", "persistence"] = "log_odds"

    # Persistence-counting parameters (used when classification_method="persistence").
    static_sweep_fraction: float = 0.3
    static_sweep_min: int = 5

    # Log-odds ray-casting parameters (used when classification_method="log_odds").
    l_occ: float = 0.85
    l_free: float = 0.40
    log_odds_clamp: float = 5.0
    p_static_threshold: float = 0.7
    p_dynamic_threshold: float = 0.3
    min_observations: int = 3
    # Voxels with fewer endpoint hits than this are free-space-only; not dynamic.
    min_occupied_hits: int = 1
    max_ray_length_m: float = 80.0
    free_space_margin_voxels: float = 1.0
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

    # MapMOS (third evidence source — learned MinkUNet over past N scans).
    # Disabled by default; geometry-only classify path is unchanged when off.
    mapmos: MapMOSParams = MapMOSParams()

    # SAM4D/MinkUNet alignment: export binary voxel occupancy alongside
    # static_map.npz.  Includes ALL occupied voxels (static + dynamic), not
    # just static ones — that's what the MinkUNet encoder consumes.
    save_voxel_occupancy: bool = True

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

    @model_validator(mode="after")
    def _thresholds_consistent(self) -> "ComponentConfig":
        # Three-band classifier: p_dynamic < p_static MUST hold, otherwise the
        # ambiguous middle band is empty or inverted and the classifier
        # silently mis-buckets voxels. See classify_from_log_odds docstring.
        if self.p_dynamic_threshold >= self.p_static_threshold:
            raise ValueError(
                f"p_dynamic_threshold ({self.p_dynamic_threshold}) must be < "
                f"p_static_threshold ({self.p_static_threshold})"
            )
        return self

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
