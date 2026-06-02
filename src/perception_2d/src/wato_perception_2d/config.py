"""Pydantic config for perception_2d, sourced from perception_2d.yaml.

The tracker is SAM 3.1's multiplex concept-video predictor (sam3_concept_tracker).
The *concept vocabulary* it tracks comes from one of two sources, selected by
``discovery.backend``:

- ``fixed``     (default) — a closed-set class list (``discovery.fixed_classes``,
  e.g. COCO classes); falls back to the prompts.yaml taxonomy when empty.
- ``florence2`` — open-vocabulary noun phrases discovered per frame by Florence-2.

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
    """Source of the concept vocabulary fed to the SAM 3.1 concept tracker."""

    model_config = ConfigDict(extra="allow")

    # "fixed"     → bypass Florence-2 and prompt SAM 3.1 directly with
    #               `fixed_classes` (closed-set, e.g. COCO classes). DEFAULT.
    # "florence2" → open-vocabulary noun phrases from Florence-2, deduped into
    #               a concept set per camera stream.
    backend: str = "fixed"
    model: str = "microsoft/Florence-2-large-ft"
    task: str = "<DENSE_REGION_CAPTION>"
    min_confidence: float = 0.3
    # Run Florence-2 on every k-th frame when backend == "florence2" (the
    # discovered phrases are pooled across the stream into one concept set).
    sample_every_k: int = 10
    # Closed-set class list used when backend == "fixed".  Empty → falls back to
    # the prompts.yaml taxonomy (concept_prompts()).
    fixed_classes: list[str] = Field(default_factory=list)


class SegmentationConfig(BaseModel):
    """SAM 3.1 multiplex concept-video tracker settings."""

    model_config = ConfigDict(extra="allow")

    version: str = "sam3.1"  # download_ckpt_from_hf(version)
    use_fa3: bool = False  # FlashAttention 3 (GPU-only); off by default
    output_prob_thresh: float = 0.5  # SAM 3.1 mask probability threshold

    # --- VRAM controls for long panoramic clips ---
    # The multiplex predictor otherwise (a) pins the whole video tensor on the
    # GPU at init_state and (b) accumulates every propagated frame's
    # mask/memory features in VRAM. On a ~650-frame, 1008px clip with 16
    # multiplex slots that OOMs a 24 GB card mid-propagation.
    #
    # offload_video_to_cpu: keep the (N, 3, 1008, 1008) fp16 frame stack in host
    #   RAM and stream one frame to the GPU per step (_get_image_feature already
    #   does `.cuda()` per frame). Frees ~4 GB up front for a 650-frame clip.
    offload_video_to_cpu: bool = True
    # offload_output_to_cpu: move each frame's pred_masks / maskmem features to
    #   host RAM as propagation advances (SAM 3.1's
    #   offload_output_to_cpu_for_eval), bounding the per-frame VRAM growth that
    #   OOMs mid-propagation; the tracker pages them back to the GPU on demand.
    #   Disable only if you suspect it regresses tracking on your build.
    offload_output_to_cpu: bool = True
    # trim_past_non_cond_mem: drop the memory features of frames that have slid
    #   past SAM 3.1's num_maskmem attention window (obj pointers are kept), so
    #   VRAM does not grow without bound over a long clip. This is the ONLY lever
    #   that bounds the per-clip OOM (multiplex_count is fixed at 16 by the
    #   checkpoint, so it can't be lowered): with trim OFF, long high-res clips
    #   OOM ~frame 80 at 1024x1280 on a 24 GB card; with trim ON, VRAM plateaus.
    #
    #   Enabling it triggers an upstream sam3 KeyError('multistep_point_inputs')
    #   mid-clip — _patch_trim_keyerror() in _sam3_runtime.py neutralises that
    #   (the key is write-only in inference). get_sam3_predictor only turns trim
    #   on if that patch is in place, so leaving this True is safe.
    trim_past_non_cond_mem: bool = True
    # forward_backbone_per_frame: compute the image backbone one frame at a time
    #   instead of batching all frames. SAM 3.1's first recommended remedy "to
    #   avoid backbone OOM errors on very long videos". Free; quality-neutral.
    forward_backbone_per_frame: bool = True

    # --- multiplex_count is NOT a tunable OOM lever ---
    # multiplex_count is a BUILD-TIME ARCHITECTURAL DIMENSION baked into the
    # published checkpoint (sam3.1_multiplex.pt was trained at 16). It sizes
    # several tracker params (no_obj_embed_spatial [16,256], mask_tokens
    # [48,256]=16*3, iou_token, mask_downsampler, ...). Setting it to anything
    # but 16 makes load_state_dict raise size-mismatch errors (which fire even
    # under strict=False), get_sam3_predictor swallows that, and the predictor
    # comes back None — i.e. SAM 3.1 silently unavailable. Verified empirically:
    # multiplex_count=8 -> "size mismatch ... [16,256] vs [8,256]". MUST stay 16.
    #
    # max_num_objects only sizes the torch.compile object-count cache
    # (num_obj_for_compile); it does NOT touch param shapes, so it is a safe
    # runtime cap. Lower it to bound how many instances are tracked per
    # (concept, camera) pass. Build-time: the predictor is rebuilt when these
    # change.
    #
    multiplex_count: int = 16
    max_num_objects: int = 12

    # --- The knobs that actually keep SAM 3.1 inside a 24 GB card ---
    # build_sam3_multiplex_video_predictor bakes these (1008 / 16 / 16, sized for
    # an 80 GB card) and exposes no override, so get_sam3_predictor sets them as
    # attributes on the built model afterwards. None = keep the build default.
    #
    # batched_grounding_batch_size: frames the detector grounds through its ViT
    #   in one batch. At the built-in 16 + image_size 1008 that single allocation
    #   is ~20 GB and OOMs the FIRST batch on a 24 GB card (propagation dies at
    #   frame 16). This is the primary fix — keep it small (1-2).
    batched_grounding_batch_size: int | None = 1
    # postprocess_batch_size: frames accumulated before postprocessing runs on
    #   them together (buffer = postprocess_batch_size * max_num_objects). Class
    #   default is 1 ("set to 1 to disable batching"); the builder forces 16.
    postprocess_batch_size: int | None = 1
    # image_size: resolution each frame is resized to before the backbone (real-
    #   valued RoPE tolerates a smaller size; memory falls ~quadratically). None
    #   keeps the built-in 1008. Lower (e.g. 768) for extra headroom at the cost
    #   of small-object recall.
    image_size: int | None = None

    # OOM fallback window. With multiplex_count locked at 16, a long enough clip
    # still OOMs mid-propagation even with every offload/trim lever on. The
    # tracker tries the whole clip first and, only on a CUDA OOM, retries in
    # windows of this many frames (fresh session each — object ids reset between
    # windows, which the downstream `tracking` component re-links via DINOv2).
    # A window that itself OOMs is recursively halved down to
    # min_sub_clip_frames before being skipped, so a dense stretch fragments
    # finely instead of dropping a whole sub_clip_frames-wide swath. Graceful
    # degradation, not an upfront decision: clips that fit are untouched.
    sub_clip_frames: int = 150
    min_sub_clip_frames: int = 16


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
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    depth: DepthConfig = Field(default_factory=DepthConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    prompts_path: str = "/config/prompts.yaml"
    upstream_versions: dict[str, str] = Field(default_factory=dict)

    def concept_prompts(self) -> list[tuple[str, str]]:
        """Return (text_prompt, canonical_class) concepts for the fixed backend.

        text_prompt seeds SAM 3.1 concept detection; canonical_class is the
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
