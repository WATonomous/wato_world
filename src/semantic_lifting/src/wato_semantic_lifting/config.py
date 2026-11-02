"""Pydantic config for semantic_lifting, sourced from semantic_lifting.yaml."""

from __future__ import annotations

from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class VisibilityConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    tolerance_m: float = 0.5
    neighborhood: int = 3
    depth_map_max_range_m: float = 10.0
    lidar_cell_px: int = 8
    lidar_neighborhood_cells: int = 1
    lidar_tolerance_per_m: float = 0.05


class InstancesConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    min_shared_points: int = 20
    min_shared_fraction: float = 0.3
    max_separation_m: float = 5.0


class ComponentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    visibility: VisibilityConfig = Field(default_factory=VisibilityConfig)
    instances: InstancesConfig = Field(default_factory=InstancesConfig)


def load_config(path: str) -> ComponentConfig:
    with open(path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    section = data.get("semantic_lifting", data)
    return ComponentConfig.model_validate(section)
