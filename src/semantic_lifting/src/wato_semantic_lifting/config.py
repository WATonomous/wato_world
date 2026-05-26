"""Pydantic config for semantic_lifting, sourced from semantic_lifting.yaml."""

from __future__ import annotations

from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class TemporalConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    max_offset_s: float = 0.05


class VisibilityConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    tolerance_m: float = 0.5
    neighborhood: int = 3


class VotingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    min_supporting_cameras: int = 1
    innermost_wins: bool = True


class DepthConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    skip_on_fit_failure: bool = True


class ComponentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    visibility: VisibilityConfig = Field(default_factory=VisibilityConfig)
    voting: VotingConfig = Field(default_factory=VotingConfig)
    depth: DepthConfig = Field(default_factory=DepthConfig)
    upstream_versions: dict[str, str] = Field(default_factory=dict)


def load_config(path: str) -> ComponentConfig:
    with open(path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    section = data.get("semantic_lifting", data)
    return ComponentConfig.model_validate(section)
