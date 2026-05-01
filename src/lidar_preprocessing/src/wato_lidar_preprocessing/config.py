"""Pydantic-loaded config for lidar_preprocessing, sourced from config/pipeline.yaml."""

from __future__ import annotations

from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict


class ComponentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    upstream_versions: dict[str, str] = {}


def load_config(path: str) -> ComponentConfig:
    with open(path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    section = data.get("lidar_preprocessing", {})
    return ComponentConfig.model_validate(section)
