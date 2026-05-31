"""Pydantic config for perception_2d, sourced from perception_2d.yaml."""

from __future__ import annotations

import os
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class SegmentationConfig(BaseModel):
    """SAM 3.1 multiplex concept-video tracker settings."""

    model_config = ConfigDict(extra="allow")

    version: str = "sam3.1"           # download_ckpt_from_hf(version)
    use_fa3: bool = False             # FlashAttention 3 (GPU-only); off by default
    output_prob_thresh: float = 0.5   # SAM 3.1 mask probability threshold


class DepthConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    model: str = "depth-anything-v2-large"
    min_lidar_anchors: int = 30  # min anchor pairs to attempt an affine fit
    ransac_n_iter: int = 200
    ransac_inlier_threshold_m: float = 0.5
    # Always static: the pipeline only loads static LiDAR (dynamic points desync
    # ~75cm at 25ms; see depth_align). Flag retained as explicit intent.
    use_static_anchors_only: bool = True
    fallback_window: int = 5
    sky_mask_top_fraction: float = 0.3
    output_dtype: str = "float16"  # stored depth-map dtype (apply_affine)


class ReidConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "dinov2_vitl14"
    every_k_frames: int = 5


class CrossCamConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    match_radius_m: float = 1.5


# Cache of {synonym(lowercased): canonical class}, keyed by prompts_path, so
# prompts.yaml is parsed once rather than per masklet.
_synonym_map_cache: dict[str, dict[str, str]] = {}


def _build_synonym_map(path: str) -> dict[str, str]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    mapping: dict[str, str] = {}
    for entry in data.get("primary_taxonomy", []):
        name = entry["name"]
        for syn in entry.get("synonyms", [name]):
            mapping[str(syn).lower()] = name
    return mapping


# Fallback taxonomy when prompts.yaml isn't mounted: (text_prompt, canonical).
_FALLBACK_CONCEPTS: list[tuple[str, str]] = [
    ("car", "car"),
    ("truck", "truck"),
    ("bus", "bus"),
    ("motorcycle", "motorcycle"),
    ("bicycle", "bicycle"),
    ("pedestrian", "pedestrian"),
    ("traffic cone", "traffic_cone"),
    ("barrier", "barrier"),
]


class ComponentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    depth: DepthConfig = Field(default_factory=DepthConfig)
    reid: ReidConfig = Field(default_factory=ReidConfig)
    cross_cam: CrossCamConfig = Field(default_factory=CrossCamConfig)

    prompts_path: str = "/config/prompts.yaml"
    upstream_versions: dict[str, str] = Field(default_factory=dict)

    def concept_prompts(self) -> list[tuple[str, str]]:
        """Return (text_prompt, canonical_class) per taxonomy class for SAM 3.1.

        text_prompt is a natural phrase (the first synonym) that seeds SAM 3.1
        concept detection; canonical_class is the taxonomy name stored as the
        masklet cls. Falls back to a hardcoded set when prompts.yaml is absent.
        """
        path = self.prompts_path
        if not os.path.exists(path):
            return list(_FALLBACK_CONCEPTS)
        with open(path, "r", encoding="utf-8") as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}
        out: list[tuple[str, str]] = []
        for entry in data.get("primary_taxonomy", []):
            name = str(entry["name"])
            syns = entry.get("synonyms", [])
            text = str(syns[0]) if syns else name
            out.append((text, name))
        return out or list(_FALLBACK_CONCEPTS)

    def synonym_to_class_map(self) -> dict[str, str]:
        """Cached {synonym(lowercased): canonical class} from prompts.yaml.

        Built once per prompts_path so canonicalisation is a dict lookup rather
        than a YAML re-parse per masklet.
        """
        cached = _synonym_map_cache.get(self.prompts_path)
        if cached is None:
            cached = _build_synonym_map(self.prompts_path)
            _synonym_map_cache[self.prompts_path] = cached
        return cached

    def class_from_synonym(self, synonym: str) -> str:
        """Map a detected synonym back to its canonical class name."""
        return self.synonym_to_class_map().get(synonym.lower(), synonym)


def load_config(path: str) -> ComponentConfig:
    with open(path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    # Support both top-level and nested under "perception_2d" key.
    section = data.get("perception_2d", data)
    return ComponentConfig.model_validate(section)
