"""Pydantic config for perception_2d v2, sourced from perception_2d.yaml."""

from __future__ import annotations

import os
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DiscoveryConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    # "florence2" → open-vocabulary phrases from Florence-2.
    # "fixed"     → bypass Florence-2 and prompt SAM 3.1 directly with
    #               `fixed_classes` (closed-set, e.g. COCO classes).
    backend: str = "florence2"
    model: str = "microsoft/Florence-2-large-ft"
    task: str = "<DENSE_REGION_CAPTION>"
    min_confidence: float = 0.3
    # Closed-set class list used when backend == "fixed".  Empty → falls back to
    # the prompts.yaml taxonomy (text_prompts()).
    fixed_classes: list[str] = Field(default_factory=list)


class SegmentationConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "sam3.1"
    checkpoint: str = "facebook/sam3.1"
    min_score: float = 0.5


class PhraseDeduplicationConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    synonym_clip_threshold: float = 0.85
    nms_3d_radius_m: float = 1.0
    nms_2d_iou: float = 0.5


class TrackerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    backend: str = "sam3"  # "sam3" (primary) or "deva" (IoU fallback, Tracker2D)


class DepthConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    model: str = "depth-anything-v2-large"
    min_lidar_anchors: int = 30
    ransac_n_iter: int = 200
    ransac_inlier_threshold_m: float = 0.5
    fallback_window: int = 5
    sky_mask_top_fraction: float = 0.3


class EmbeddingConfig(BaseModel):
    """DINOv2 appearance-embedding extraction.

    This stage only *extracts and persists* embeddings (for the downstream
    `tracking` component to use for re-identification); it does not itself
    re-identify anything.
    """

    model_config = ConfigDict(extra="allow")

    model: str = "dinov2_vitl14"
    every_k_frames: int = 5


class ComponentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    phrase_dedup: PhraseDeduplicationConfig = Field(
        default_factory=PhraseDeduplicationConfig
    )
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    depth: DepthConfig = Field(default_factory=DepthConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    prompts_path: str = "/config/prompts.yaml"
    upstream_versions: dict[str, str] = Field(default_factory=dict)

    def text_prompts(self) -> list[str]:
        """Return flat synonym list used to filter Florence-2 proposals.

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

    def synonym_to_class(self) -> dict[str, str]:
        """Build a {synonym → canonical class name} map from prompts.yaml.

        Returns an empty map when prompts.yaml is absent (callers then keep the
        raw phrase).  Built once per chunk and passed to row construction so
        masklet `cls` carries the canonical taxonomy label rather than the raw
        Florence-2 phrase.
        """
        path = self.prompts_path
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}
        mapping: dict[str, str] = {}
        for entry in data.get("primary_taxonomy", []):
            name = entry["name"]
            for syn in entry.get("synonyms", [name]):
                mapping[syn] = name
            mapping.setdefault(name, name)
        return mapping


def load_config(path: str) -> ComponentConfig:
    with open(path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    # Support both top-level and nested under "perception_2d" key.
    section = data.get("perception_2d", data)
    return ComponentConfig.model_validate(section)
