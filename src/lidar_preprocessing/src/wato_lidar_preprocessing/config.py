"""Pydantic-loaded config for lidar_preprocessing."""

from __future__ import annotations

from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    ground_cell_size_m: float = 0.5

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
    static_sweep_fraction: float = 0.3
    static_sweep_min: int = 5
    cache_world_xyz_in_memory: bool = True

    # Step D — global static map reduce.
    global_map_voxel_size_m: float = 0.30

    # Per-point timestamp unit stored by ingest's t_offset_us field.
    # Velodyne "t" field is in seconds; some lidars use microseconds.
    # Options: "seconds" | "microseconds" | "nanoseconds"
    point_time_unit: str = "seconds"

    # Step C — Patchwork++ parameters.
    patchwork: PatchworkParams = PatchworkParams()

    # Optional SAM4D/MinkUNet alignment: export binary voxel occupancy.
    # When True, classify writes voxel_occupancy.npz alongside static_map.npz.
    # Includes ALL occupied voxels (static + dynamic), not just static ones.
    save_voxel_occupancy: bool = False

    upstream_versions: dict[str, str] = Field(default_factory=dict)

    @field_validator("voxel_size_m", "global_map_voxel_size_m")
    @classmethod
    def _positive_voxel(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"voxel size must be > 0, got {v}")
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
