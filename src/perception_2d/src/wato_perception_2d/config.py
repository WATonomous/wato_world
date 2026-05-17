"""Pydantic config for perception_2d, sourced from pipeline.yaml."""

from __future__ import annotations

import os
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ReidConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "dinov2_vitl14"
    every_k_frames: int = 5


class DetectorEntry(BaseModel):
    """One entry in the detectors list (multi-detector ensemble).

    `name` is the canonical adapter identifier (must match the dispatch in
    `pipeline.py::_build_detector`).  `checkpoint` is the filename inside
    the model's MODELS_ROOT subdir (e.g. "yolov8l-worldv2.pt"), or null to
    use each adapter's default.
    """

    model_config = ConfigDict(extra="forbid")

    name: str                                  # "grounding_dino" | "yolo_world"
    enabled: bool = True
    score_threshold: float = 0.25
    checkpoint: Optional[str] = None           # adapter default if None


class DepthEstimatorConfig(BaseModel):
    """Depth Anything V2 settings — see docs/research/depth_anything_guidance.md."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    model: str = "depth_anything_v2_large"
    save_dtype: str = "float16"                # what to persist in depth_2d/*.npy
    align_to_lidar: bool = True
    # Minimum overlapping LiDAR points required to trust the (scale, shift) fit.
    min_overlap_pts: int = 50
    # RANSAC inlier distance threshold in metres.
    inlier_thresh_m: float = 0.5


class ComponentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    # ------------------------------------------------------------------
    # Detection.
    # ------------------------------------------------------------------
    # Legacy single-detector knobs.  Still consumed when `detectors` is empty
    # so existing pipeline.yamls keep working.
    detector: str = "grounding_dino"
    detector_score_threshold: float = 0.25

    # Multi-detector ensemble (per detector_ensemble_guidance.md).  When
    # this list is non-empty it supersedes the legacy `detector` field.
    detectors: list[DetectorEntry] = Field(default_factory=list)
    detector_ensemble_iou: float = 0.6

    # ------------------------------------------------------------------
    # Segmentation.
    # ------------------------------------------------------------------
    sam2_checkpoint: str = "sam2_hiera_large"
    use_lidar_prompts: bool = True
    lidar_prompt_max_points: int = 50

    # ------------------------------------------------------------------
    # Depth (new — Depth Anything V2 + LiDAR-based metric alignment).
    # ------------------------------------------------------------------
    depth_estimator: DepthEstimatorConfig = Field(default_factory=DepthEstimatorConfig)

    # ------------------------------------------------------------------
    # ReID + cross-camera merge (existing).
    # ------------------------------------------------------------------
    reid_features: ReidConfig = Field(default_factory=ReidConfig)
    cross_camera_match_radius_m: float = 1.5

    # ------------------------------------------------------------------
    # Misc.
    # ------------------------------------------------------------------
    prompts_path: str = "/config/prompts.yaml"
    upstream_versions: dict[str, str] = Field(default_factory=dict)

    def text_prompts(self) -> list[str]:
        """Return flat synonym list for the detector text query.

        Reads prompts.yaml if available; falls back to a hardcoded set.
        """
        path = self.prompts_path
        if not os.path.exists(path):
            return ["car", "truck", "bus", "motorcycle", "bicycle",
                    "pedestrian", "traffic cone", "barrier"]
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
