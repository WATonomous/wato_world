"""Pydantic config for proposal_generation, sourced from proposal_generation.yaml."""

from __future__ import annotations

import math
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class FramesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # The LiDAR lidar_preprocessing's frame_sync anchored frames on. No
    # artifact records it, so it is restated here; a run fails when it doesn't
    # match how the chunk's frames were grouped.
    canonical_lidar: Optional[str] = None


class LidarMosConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Step F clusters become proposals only past these gates
    # (docs/lidar_preprocessing_measurements.md, "Step F against ground truth").
    min_motion_score: float = 1.0
    min_n_sources: int = 2
    max_frac_persistent: float = 0.5
    # A cluster takes a class only when this fraction of its points carry a
    # lifted label; below it cls is None.
    min_labeled_fraction: float = 0.3
    # Shape gates, on the median box of the cluster's Step F track
    # (track_hint_id). A track longer than this many widths is wall-shaped: a
    # strip of structure whose visible part slid as the ego passed (LISO's
    # length/width limit; ~20% of WATO's gated tracks,
    # docs/proposal_generation_measurements.md). Pedestrians and cyclists sit
    # well under it.
    max_aspect_ratio: float = 4.0
    # A row no lifted label classes is dropped when its track is smaller than
    # these (LISO's floors). A pedestrian is smaller, so a classed row keeps
    # its box; 0 disables.
    min_footprint_m2: float = 0.35
    min_volume_m3: float = 0.5


class ScanRefinementConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # SAL Appendix A.1/Table A.1's six density thresholds, in published order.
    # Interpreting them as Euclidean XYZ radii in metres is an implementation
    # choice: the paper does not name the metric, units, or DBSCAN backend.
    radii_m: tuple[float, ...] = (1.2488, 0.8136, 0.6952, 0.594, 0.4353, 0.3221)
    # SAL does not report this. Five is the standard sklearn DBSCAN default;
    # each point counts itself. It is configurable, not a claimed SAL value.
    min_samples: int = Field(default=5, ge=1)
    # Strictly greater, as in SAL. A failed match keeps the ORIGINAL seed.
    min_iou: float = Field(default=0.5, ge=0.0, le=1.0)
    # Disable for an original-support ablation, without largest-group cleanup.
    enabled: bool = True

    @field_validator("radii_m")
    @classmethod
    def valid_radii(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if not values or any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("radii_m must contain finite, positive radii")
        return values


class LiftedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Keep the two algorithms selectable for frozen-input comparisons. This
    # chooses support before the same existing box postfilters are applied.
    # SAL remains opt-in: the frozen nuScenes comparison currently loses
    # proposal precision/recall (docs/research/sal_scan_cluster_refinement.md).
    support_method: Literal["scan_clusters", "largest_group"] = "largest_group"
    refinement: ScanRefinementConfig = Field(default_factory=ScanRefinementConfig)
    # Resolve ownership only after independent 3D refinement and box gates.
    # SAL-4D Appendix A.1.1 uses strict IoM > 0.5. Here support size ranks
    # hypotheses within one frame; temporal volume is not accumulated.
    max_support_iom: float = Field(default=0.5, ge=0.0, le=1.0)
    # Only the legacy largest_group method reads these fields.
    cluster_radius_m: float = Field(default=2.0, gt=0)
    cluster_radius_by_class_m: dict[str, float] = Field(default_factory=dict)
    # Points higher than this above the ground grid are left out of the box,
    # per class (CPD's vehicle window stops at 3 m): bleed onto a pole or wall
    # behind a car made car boxes 3.3 m tall at p90 on WATO. Without a ground
    # grid, height is over the refined support's lowest point. This is a box
    # postfilter, applied AFTER SAL matching; it never changes the seed IoU.
    max_height_m: dict[str, float] = Field(default_factory=dict)
    # A box with at least this share of its BEV footprint inside a same-class
    # box with more points in the frame is dropped (DetZero's nested-box
    # filter): a car hidden behind another is boxed on its occluder's edge
    # returns, at the occluder's depth (docs/proposal_generation_measurements.md,
    # "Occlusion"), so point count, not depth, tells them apart. > 1 disables.
    max_nested_fraction: float = 0.3
    # A box with at least fragment_min_fraction of its points within
    # fragment_radius_m of the points of a box with at least
    # fragment_min_ratio times as many is dropped, whatever either's class: a
    # small mask beside a big one claims a few of the big object's edge
    # points (thin "car" columns beside a car, docs/tracking_measurements.md
    # "Duplicate tracks"). min_fraction > 1 disables.
    fragment_radius_m: float = 0.3
    fragment_min_fraction: float = 0.8
    fragment_min_ratio: float = 5.0
    # Fewer points than this in a frame: no box.
    min_points: int = Field(default=5, ge=1)


class FusionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # A lidar_mos and a lifted box are one object when they share this many
    # LiDAR points in the frame, and this fraction of the smaller one's.
    min_shared_points: int = 5
    min_shared_fraction: float = 0.3


class ComponentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frames: FramesConfig = Field(default_factory=FramesConfig)
    lidar_mos: LidarMosConfig = Field(default_factory=LidarMosConfig)
    lifted: LiftedConfig = Field(default_factory=LiftedConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)


def load_config(path: str) -> ComponentConfig:
    with open(path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    section = data.get("proposal_generation", data)
    return ComponentConfig.model_validate(section)
