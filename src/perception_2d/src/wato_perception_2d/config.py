"""Pydantic config for perception_2d, sourced from perception_2d.yaml.

The tracker is a 2D detector (GroundingDINO) + SAM2 video predictor
(detector.py + sam2_tracker.py): the detector emits class-labeled boxes, SAM2
turns each into a mask and tracks it across the camera stream into masklets.

The *class vocabulary* fed to the detector comes from one of two sources,
selected by ``discovery.backend``:

- ``fixed``     (default) — the closed-set taxonomy (``discovery.fixed_classes``
  or prompts.yaml ``primary_taxonomy``). Cleanest output, no synonym dupes.
- ``florence2`` — open-vocabulary noun phrases discovered per frame by Florence-2,
  pooled into a concept set per camera stream.

DINOv2 appearance embeddings are extracted per masklet (``embeddings``) for the
downstream ``tracking`` component; perception_2d does not re-identify here.
Cross-camera identity merging is likewise deferred downstream.
"""

from __future__ import annotations

import os
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DiscoveryConfig(BaseModel):
    """Source of the class vocabulary fed to the detector."""

    model_config = ConfigDict(extra="allow")

    # "fixed"     → prompt the detector with the closed-set taxonomy. DEFAULT.
    # "florence2" → open-vocabulary noun phrases from Florence-2, deduped into a
    #               concept set per camera stream, then fed to the detector.
    backend: str = "fixed"
    model: str = "microsoft/Florence-2-large-ft"
    task: str = "<DENSE_REGION_CAPTION>"
    min_confidence: float = 0.3
    # Run Florence-2 on every k-th frame when backend == "florence2".
    sample_every_k: int = 10
    # Closed-set class list used when backend == "fixed".  Empty → falls back to
    # the prompts.yaml taxonomy (concept_prompts()).
    fixed_classes: list[str] = Field(default_factory=list)


class DetectionConfig(BaseModel):
    """GroundingDINO detector — the per-keyframe box source for SAM2."""

    model_config = ConfigDict(extra="allow")

    # HuggingFace zero-shot detection checkpoint. The Transformers backend needs
    # no CUDA custom-op compile (unlike the standalone groundingdino package).
    model: str = "IDEA-Research/grounding-dino-base"
    box_threshold: float = 0.35  # min box confidence
    text_threshold: float = 0.25  # min text-token match score
    nms_iou: float = 0.5  # per-class NMS IoU to drop duplicate boxes
    # Run the detector every k frames to introduce objects entering mid-clip.
    # Large value → effectively detect-once at frame 0.
    redetect_every_k: int = 10


class SegmentationConfig(BaseModel):
    """SAM2 video predictor settings."""

    model_config = ConfigDict(extra="allow")

    # Local SAM2.1 checkpoint path. data/models is bind-mounted read-only at
    # /data/models in the container; the checkpoint is a loose .pt (placed there
    # by scripts/fetch_models.py), loaded via build_sam2_video_predictor — not
    # from_pretrained, since the container runs HF_HUB_OFFLINE=1.
    checkpoint: str = "/data/models/sam2.1_hiera_large.pt"
    # Hydra config name bundled in the `sam2` package (matches the checkpoint).
    config: str = "configs/sam2.1/sam2.1_hiera_l.yaml"


class TrackingConfig(BaseModel):
    """How detector boxes become tracked masklets via SAM2 propagation."""

    model_config = ConfigDict(extra="allow")

    # A re-detected box whose IoU with an already-tracked object's mask bbox is
    # >= this is treated as the existing object (not a new track).
    iou_match_threshold: float = 0.5
    # Keep the SAM2 frame stack in host RAM (stream per frame to GPU). Essential
    # for long panoramic clips that would otherwise OOM at init_state.
    offload_video_to_cpu: bool = True
    # OOM fallback: if the whole clip OOMs, retry in windows of this many frames
    # (fresh session each — object ids reset between windows, re-linked downstream
    # via DINOv2). A window that itself OOMs is recursively halved to the floor.
    sub_clip_frames: int = 150
    min_sub_clip_frames: int = 16


class DepthConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    model: str = "depth-anything-v2-large"
    # Frames pushed through the DA-V2 backbone in one GPU batch during the depth
    # pass (depth is VRAM-disjoint from the tracker; batching is a throughput win
    # when the card has headroom). 1 = original per-frame streaming.
    batch_size: int = 1
    min_lidar_anchors: int = 30  # min anchor pairs to attempt an affine fit
    ransac_n_iter: int = 200
    ransac_inlier_threshold_m: float = 0.5
    # Always static: the pipeline only loads static LiDAR (dynamic points desync
    # ~75cm at 25ms; see depth_align). Flag retained as explicit intent.
    use_static_anchors_only: bool = True
    fallback_window: int = 5
    sky_mask_top_fraction: float = 0.3
    output_dtype: str = "float16"  # stored depth-map dtype (apply_affine)


class EmbeddingConfig(BaseModel):
    """DINOv2 appearance-embedding extraction.

    This stage only *extracts and persists* embeddings (for the downstream
    `tracking` component to use for re-identification); it does not itself
    re-identify anything.
    """

    model_config = ConfigDict(extra="allow")

    model: str = "dinov2_vitl14"
    every_k_frames: int = 5


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
        mapping.setdefault(str(name).lower(), name)
    return mapping


# Fallback taxonomy when prompts.yaml isn't mounted and fixed_classes is empty:
# (text_prompt, canonical).
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

    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    depth: DepthConfig = Field(default_factory=DepthConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    prompts_path: str = "/config/prompts.yaml"
    upstream_versions: dict[str, str] = Field(default_factory=dict)

    def concept_prompts(self) -> list[tuple[str, str]]:
        """Return (text_prompt, canonical_class) concepts for the fixed backend.

        text_prompt seeds the detector's text query; canonical_class is the
        taxonomy name stored as the masklet cls.  Source priority:

        1. ``discovery.fixed_classes`` (e.g. COCO classes), canonicalised
           through the prompts.yaml synonym map when one matches.
        2. the prompts.yaml ``primary_taxonomy`` (first synonym as the prompt
           text, taxonomy name as canonical).
        3. a hardcoded fallback list.
        """
        syn2cls = self.synonym_to_class_map()
        if self.discovery.fixed_classes:
            return [
                (c, syn2cls.get(c.lower(), c))
                for c in self.discovery.fixed_classes
                if c and c.strip()
            ]
        path = self.prompts_path
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                data: dict[str, Any] = yaml.safe_load(fh) or {}
            out: list[tuple[str, str]] = []
            for entry in data.get("primary_taxonomy", []):
                name = str(entry["name"])
                syns = entry.get("synonyms", [])
                text = str(syns[0]) if syns else name
                out.append((text, name))
            if out:
                return out
        return list(_FALLBACK_CONCEPTS)

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
