"""Pydantic config for perception_2d, sourced from pipeline.yaml."""

from __future__ import annotations

import os
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ReidConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "dinov2_vitl14"
    every_k_frames: int = 5


class ComponentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    # 2D object detector backend.
    detector: str = "grounding_dino"
    detector_score_threshold: float = 0.25

    # SAM3 checkpoint name or local path.
    sam3_checkpoint: str = "sam3_hiera_large"

    # Whether to project LiDAR dynamic-mask points into image space and pass
    # them as additional SAM3 point prompts (cross-modal prompting, SAM4D-style).
    use_lidar_prompts: bool = True
    lidar_prompt_max_points: int = 50

    reid_features: ReidConfig = Field(default_factory=ReidConfig)

    # 3D radius (metres, world frame) for clustering tracklets from different
    # cameras into the same global_object_id.
    cross_camera_match_radius_m: float = 1.5

    # Path to the prompts YAML (read at runtime).
    prompts_path: str = "/config/prompts.yaml"

    upstream_versions: dict[str, str] = Field(default_factory=dict)

    def text_prompts(self) -> list[str]:
        """Return flat synonym list for the detector text query.

        Reads prompts.yaml if available; falls back to a hardcoded set.
        """
        path = self.prompts_path
        if not os.path.exists(path):
            return [
                "car",
                "truck",
                "bus",
                "motorcycle",
                "bicycle",
                "pedestrian",
                "traffic cone",
                "barrier",
            ]
        with open(path, "r", encoding="utf-8") as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}
        synonyms: list[str] = []
        for entry in data.get("primary_taxonomy", []):
            synonyms.extend(entry.get("synonyms", [entry["name"]]))
        return synonyms

    def class_from_synonym(self, synonym: str) -> str:
        """Map a detected synonym back to its canonical class name."""
        path = self.prompts_path
        if not os.path.exists(path):
            return synonym
        with open(path, "r", encoding="utf-8") as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}
        for entry in data.get("primary_taxonomy", []):
            if synonym in entry.get("synonyms", []):
                return entry["name"]
        return synonym


def load_config(path: str) -> ComponentConfig:
    with open(path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    section = data.get("perception_2d", {})
    return ComponentConfig.model_validate(section)
