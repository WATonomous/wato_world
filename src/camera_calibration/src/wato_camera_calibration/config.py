"""Strict configuration; empirical validation qualification is explicit."""

from __future__ import annotations

from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SamplingConfig(Strict):
    lidar_id: str | None = None
    max_views_per_camera: int = Field(36, ge=6)
    interval_s: float = Field(2.0, gt=0)
    accumulation_s: float = Field(0.4, ge=0)
    block_s: float = Field(12.0, gt=0)
    guard_s: float = Field(1.0, ge=0)
    max_points_per_view: int = Field(20000, ge=100)
    min_points_per_view: int = Field(500, ge=100)
    voxel_m: float = Field(0.15, gt=0)
    min_range_m: float = Field(2.0, ge=0)
    max_range_m: float = Field(60.0, gt=0)
    min_intensity_span: float = Field(5.0, gt=0)

    @model_validator(mode="after")
    def coherent(self):
        if self.guard_s < self.accumulation_s:
            raise ValueError("guard_s must cover accumulation_s")
        if self.block_s <= 2 * self.guard_s:
            raise ValueError("block_s must exceed twice guard_s")
        if self.max_range_m <= self.min_range_m:
            raise ValueError("max_range_m must exceed min_range_m")
        if self.min_points_per_view > self.max_points_per_view:
            raise ValueError("min_points_per_view exceeds point budget")
        return self


class SolverConfig(Strict):
    motion_model: Literal["rigid", "rotation_only"] = "rigid"
    executable: str | None = None
    max_iterations: int = Field(80, ge=1)
    outer_iterations: int = Field(3, ge=1, le=10)
    timeout_s: float = Field(120.0, gt=0)
    bins: int = Field(16, ge=4, le=64)
    starts: int = Field(3, ge=2, le=10)
    subsets: int = Field(2, ge=2, le=5)
    perturb_rotation_deg: float = Field(0.5, ge=0)
    max_rotation_deg: float = Field(5.0, gt=0)
    max_translation_m: float = Field(0.3, gt=0)

    @model_validator(mode="after")
    def starts_inside_bounds(self):
        if self.perturb_rotation_deg >= self.max_rotation_deg:
            raise ValueError(
                "perturb_rotation_deg must be smaller than max_rotation_deg"
            )
        return self


class ValidationConfig(Strict):
    # False until controls and independently annotated WATO observations
    # establish the acceptance procedure. Convergence/NID cannot enable it.
    qualified: bool = False
    profile_id: str | None = None
    correspondences: str | None = None
    min_views_per_split: int = Field(3, ge=1)
    min_blocks_per_split: int = Field(2, ge=1)
    min_correspondences: int = Field(12, ge=4)
    max_reprojection_median_px: float = Field(2.0, gt=0)
    max_reprojection_p90_px: float = Field(5.0, gt=0)
    min_improvement_px: float = Field(0.5, ge=0)
    max_rotation_spread_deg: float = Field(0.3, gt=0)
    max_translation_spread_m: float = Field(0.05, gt=0)
    min_coverage_fraction: float = Field(0.25, gt=0, le=1)
    min_depth_ratio: float = Field(1.5, gt=1)
    max_edge_regression_px: float = Field(1.0, ge=0)

    @model_validator(mode="after")
    def qualification(self):
        if self.qualified and (not self.profile_id or not self.correspondences):
            raise ValueError(
                "qualified validation requires profile_id and independent correspondences"
            )
        return self


class ComponentConfig(Strict):
    enabled: bool = False
    cameras: list[str] | None = None
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    mode: Literal["run", "prepare"] = "run"

    @model_validator(mode="after")
    def configured_lidar(self):
        if self.enabled and not self.sampling.lidar_id:
            raise ValueError(
                "enabled calibration requires an explicit sampling.lidar_id"
            )
        return self


def load_config(path: str) -> ComponentConfig:
    with open(path, encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    return ComponentConfig.model_validate(data.get("camera_calibration", data))
