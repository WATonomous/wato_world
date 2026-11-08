"""Pydantic config for perception_2d, sourced from perception_2d.yaml.

A per-keyframe box detector (``detection.backend``: YOLO26 or GroundingDINO)
feeds SAM2's video predictor, which tracks each box into a masklet. The class
vocabulary comes from ``discovery.backend``: ``fixed`` (the closed-set
taxonomy) or ``florence2`` (open-vocab phrases, GroundingDINO only). DINOv2
embeddings are extracted per masklet for the downstream ``tracking`` component.
"""

from __future__ import annotations

import os
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DiscoveryConfig(BaseModel):
    """Source of the class vocabulary fed to the detector."""

    model_config = ConfigDict(extra="allow")

    # "fixed"     → prompt the detector with the closed-set taxonomy (default).
    # "florence2" → open-vocabulary noun phrases from Florence-2, deduped into a
    #               concept set per camera stream, then fed to the detector.
    backend: str = "fixed"
    model: str = "microsoft/Florence-2-large-ft"
    min_confidence: float = 0.3
    # Run Florence-2 on every k-th frame when backend == "florence2".
    sample_every_k: int = 10
    # Closed-set class list used when backend == "fixed".  Empty → falls back to
    # the prompts.yaml taxonomy (concept_prompts()).
    fixed_classes: list[str] = Field(default_factory=list)


def _default_yolo_class_map() -> dict[str, str]:
    return {
        "person": "pedestrian",
        "bicycle": "bicycle",
        "car": "car",
        "motorcycle": "motorcycle",
        "bus": "bus",
        "truck": "truck",
    }


class YoloConfig(BaseModel):
    """Closed-set YOLO detector (detection.backend: yolo)."""

    model_config = ConfigDict(extra="forbid")

    # Pinned weights (model_registry.RAW_CHECKPOINTS["yolo26"]), loaded by path.
    checkpoint: str = "/data/models/yolo26x.pt"
    conf: float = 0.25  # min box confidence
    iou: float = 0.7  # NMS IoU
    imgsz: int = 1280  # inference size (long side); frames are 1280x1024
    # Weights' class name → taxonomy class. Unmapped classes are dropped.
    class_map: dict[str, str] = Field(default_factory=_default_yolo_class_map)


class DetectionConfig(BaseModel):
    """Per-keyframe box source for SAM2."""

    model_config = ConfigDict(extra="allow")

    # "yolo": closed-set COCO weights mapped onto the taxonomy (yolo.class_map).
    # "grounding_dino": open-vocabulary, text-prompted by the taxonomy (and by
    # Florence-2 discovery when discovery.backend is florence2).
    backend: Literal["yolo", "grounding_dino"] = "yolo"
    yolo: YoloConfig = Field(default_factory=YoloConfig)

    # --- grounding_dino ---------------------------------------------------
    # HuggingFace zero-shot detection checkpoint. The Transformers backend needs
    # no CUDA custom-op compile (unlike the standalone groundingdino package).
    model: str = "IDEA-Research/grounding-dino-base"
    box_threshold: float = 0.35  # min box confidence
    text_threshold: float = 0.25  # min text-token match score
    nms_iou: float = 0.5  # per-class NMS IoU to drop duplicate boxes

    # --- both backends ----------------------------------------------------
    # Run the detector every k frames to introduce objects entering mid-clip.
    # Large value → effectively detect-once at frame 0.
    redetect_every_k: int = 2  # images; see config/perception_2d.yaml


class MaskRefinementConfig(BaseModel):
    """Experimental image-mask seeds; thresholds need validation on WATO."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    # SAM2 automatic-mask-generator defaults, not calibrated WATO probabilities.
    min_predicted_iou: float = Field(default=0.8, ge=0, le=1)
    min_stability: float = Field(default=0.95, ge=0, le=1)
    stability_offset: float = Field(default=1.0, gt=0)
    # False follows the image API's advice for unambiguous box prompts.
    multimask_output: bool = False
    # At most one extra decoder pass; ablate points and logit recurrence separately.
    interior_point: bool = True
    mask_input_refinement: bool = True
    # Start with seed / existing box-drift refinement; ablate shape correction separately.
    correct_shape_disagreement: bool = False
    # A fresh mask must retain this much agreement before correcting an ID.
    # These are engineering guardrails, not published optimal thresholds.
    correction_min_mask_iou: float = Field(default=0.5, ge=0, le=1)
    correction_below_mask_iou: float = Field(default=0.8, ge=0, le=1)

    @model_validator(mode="after")
    def correction_range(self) -> MaskRefinementConfig:
        if self.correction_min_mask_iou >= self.correction_below_mask_iou:
            raise ValueError(
                "correction_min_mask_iou must be below correction_below_mask_iou"
            )
        return self


class Dam4samConfig(BaseModel):
    """DAM4SAM memory parameters the adapter sets (models/dam4sam.py)."""

    model_config = ConfigDict(extra="forbid")

    # Memory update spacing Δ, in images. The authors use 5 at 30 fps and 1 at
    # 5 fps (DAM4SAM extended, §3.2.1 / §5.5): one update per ~170-200 ms,
    # which is 2 at WATO's ~10 Hz and nuScenes' 12 Hz cameras.
    update_delta: int = Field(default=2, ge=1)
    # SAMURAI's memory-bank gate (its code's memory_bank_iou_threshold): a
    # frame is kept as an older recent-appearance memory only if SAM2's
    # predicted IoU exceeds this; the previous frame is always attended to.
    # Object pointers are not gated, as in SAMURAI. 0 disables.
    memory_min_predicted_iou: float = Field(default=0.5, ge=0, le=1)


class SegmentationConfig(BaseModel):
    """SAM2.1 weights, propagation backend and optional image-mask refinement."""

    model_config = ConfigDict(extra="allow")

    # Published training-free memory policy, using the same SAM2.1 weights.
    backend: Literal["sam2", "dam4sam"] = "dam4sam"
    # backend dam4sam only; stock SAM2 keeps Meta's memory.
    dam4sam: Dam4samConfig = Field(default_factory=Dam4samConfig)

    # Local SAM2.1 checkpoint path. data/models is bind-mounted read-only at
    # /data/models in the container; the checkpoint is a loose .pt (placed there
    # by scripts/fetch_models.py), loaded by the selected backend with no download.
    checkpoint: str = "/data/models/sam2.1_hiera_large.pt"
    # Hydra config name bundled in the `sam2` package (matches the checkpoint).
    config: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    refinement: MaskRefinementConfig = Field(default_factory=MaskRefinementConfig)
    # Write per-camera mask diagnostics (mask_quality.py). Off: computing them
    # costs a GPU sync per frame and 10-20 MB of JSON per WATO chunk.
    mask_quality: bool = False


class TrackingConfig(BaseModel):
    """How detector boxes become tracked masklets via SAM2 propagation."""

    model_config = ConfigDict(extra="allow")

    # A re-detected box whose IoU with an already-tracked object's mask bbox is
    # >= this is treated as the existing object (not a new track).
    iou_match_threshold: float = 0.5
    # Stock SAM2 frame stack on CPU. DAM4SAM always loads frames lazily on CPU.
    offload_video_to_cpu: bool = True
    # OOM fallback: if the whole clip OOMs, retry in windows of this many frames
    # (fresh session each — object ids reset between windows; `tracking`
    # associates 3D boxes, not masklet ids). A window that itself OOMs is
    # recursively halved to the floor.
    sub_clip_frames: int = 150
    min_sub_clip_frames: int = 16
    # SAM2 doesn't let go of an object it has lost: within 0-30 frames it tends
    # to latch the lost object's id onto a similar object nearby. A track ends
    # (no more masks recorded; removed from the session at the next keyframe)
    # once its object has had no mask for this many consecutive frames. 0 = never.
    drop_lost_after_frames: int = 1  # images
    # Drop a mask's specks: 8-connected components smaller than this fraction
    # of its largest one (the largest always stays). 0 disables.
    island_fraction: float = 0.1
    # Independent masks retain contested pixels; overlap, containment and
    # collapse are resolved after lifting and geometric refinement.
    # Detector support, on keyframes (a box matching a track's mask bbox at
    # iou_match_threshold supports it). A track's keep-alive rises by one per
    # supported keyframe (up to support_keyframes) and falls by one per
    # keyframe its mask's bbox covers >= support_min_area of the image with no
    # support; at 0 it ends (0 disables). A box scoring >= birth_score starts a
    # track; a weaker one starts it tentatively, its masks held back until
    # boxes support it at confirm_keyframes keyframes in a row, dropped at the
    # first without (birth_score 0: every box starts a track). A track's class
    # is the one its boxes' scores sum highest for.
    birth_score: float = 0.0
    confirm_keyframes: int = 3
    support_keyframes: int = 5
    support_min_area: float = 0.01
    # Whether a box no track accounts for may start one, tested against the
    # tracked masks (tracks still live at the keyframe, plus births already
    # accepted there, highest score first):
    #   untracked_area: SAM2MOT (arXiv 2504.04519, "object addition"): more
    #     than birth_untracked_fraction (its 0.5) of the box lies outside
    #     every tracked mask;
    #   mask_iou: SAM 3 (arXiv 2511.16719, App. C.3): SAM2's image-decoder
    #     mask for the box has IoU below birth_max_mask_iou (its 0.1) with
    #     every tracked mask; costs one image encoding per keyframe with births;
    #   none: every such box starts a track.
    # A refused box neither starts nor supports a track.
    birth_gate: Literal["untracked_area", "mask_iou", "none"] = "untracked_area"
    birth_untracked_fraction: float = Field(default=0.5, ge=0, lt=1)
    birth_max_mask_iou: float = Field(default=0.1, gt=0, le=1)
    # Reverse back-fill ("once detected, never lost": CTRL arXiv 2304.12315,
    # OC-SORT 2203.14360, SAM4D 2506.21547): each track born after its
    # window's first image is propagated backwards from its birth prompt for
    # up to this many images (CTRL's 20 frames for short tracks, at 10 Hz).
    # It stops at drop_lost_after_frames empty masks, on the ego vehicle, or
    # where its mask's bbox fails the untracked_area test against the tracks
    # recorded there. 0 disables.
    backfill_frames: int = Field(default=20, ge=0)  # images
    # A track backed by exactly one box scoring >= reprompt_score that backs no
    # other track, whose mask bbox has drifted below reprompt_below_iou from
    # it, is re-prompted with the box. 0 disables.
    reprompt_score: float = 0.8
    reprompt_below_iou: float = 0.7
    # Precision for SAM2 calls on CUDA: "bfloat16" (upstream's setting; enables
    # flash attention), "float16", or "float32". Detector + DINOv2 stay float32.
    autocast_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"


class EgoMaskConfig(BaseModel):
    """The vehicle's own body (rack, hood, antenna) as each camera sees it.

    Without it the detector boxes the rack as a truck on every keyframe and
    SAM2 tracks it for the whole chunk. Per vehicle, so it lives in the
    vehicle's config (perception_2d.wato.yaml); empty = no camera sees the car.
    """

    model_config = ConfigDict(extra="forbid")

    # (width, height) of the rectified images the polygons are drawn on; a
    # camera whose images differ is an error, not a silently misplaced mask.
    image_size: tuple[int, int] = (0, 0)
    # A detection box, or a track's mask, at least this fraction inside the
    # region is the vehicle itself.
    max_fraction: float = 0.5
    # cam_id → polygons, each a list of [x, y] pixel vertices. Cameras not
    # listed don't see the vehicle.
    polygons: dict[str, list[list[tuple[int, int]]]] = Field(default_factory=dict)


class DepthConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    model: str = "depth-anything-v2-large"
    # Frames per DA-V2 GPU batch in the depth pass. 1 = one frame at a time.
    batch_size: int = 1
    min_lidar_anchors: int = 30  # min anchor pairs to attempt a disparity fit
    ransac_n_iter: int = 200
    ransac_inlier_threshold_m: float = 0.5
    fallback_window: int = 5
    sky_mask_top_fraction: float = 0.3
    output_dtype: str = "float16"  # stored depth-map dtype (disparity_to_depth)


class EmbeddingConfig(BaseModel):
    """DINOv2 appearance embeddings, kept for a later `tracking` ReID (unread today)."""

    model_config = ConfigDict(extra="allow")

    model: str = "dinov2_vitl14"
    every_k_frames: int = 1  # images


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
]


class ComponentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    ego_mask: EgoMaskConfig = Field(default_factory=EgoMaskConfig)
    depth: DepthConfig = Field(default_factory=DepthConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    prompts_path: str = "/config/prompts.yaml"

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
        """Cached {synonym(lowercased): canonical class} from prompts.yaml."""
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
