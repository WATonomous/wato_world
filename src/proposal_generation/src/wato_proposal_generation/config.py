"""Pydantic config for proposal_generation, sourced from proposal_generation.yaml."""

from __future__ import annotations

from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field


class FramesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # The LiDAR lidar_preprocessing's frame_sync anchored frames on. No
    # artifact records it, so it is restated here; a run fails when it doesn't
    # match how the chunk's frames were grouped.
    canonical_lidar: Optional[str] = None


class LidarMosConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Step F clusters become proposals only past these gates
    # (docs/research/lidar_mos_guidance.md, "How downstream should consume it").
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


class LiftedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Points of one instance chained within this distance are one group; only
    # the largest group is boxed (masklets bleed onto background). Lifted
    # objects are far and sparse (median 34-51 m on WATO): 0.75 m splits real
    # cars (docs/proposal_generation_measurements.md).
    cluster_radius_m: float = 2.0
    # Per-class override of cluster_radius_m, about the class's width (VESPA):
    # at 2 m a pedestrian's group also takes in the wall or car beside it.
    cluster_radius_by_class_m: dict[str, float] = Field(default_factory=dict)
    # Points higher than this above the ground grid are left out of the box,
    # per class (CPD's vehicle window stops at 3 m): bleed onto a pole or wall
    # behind a car made car boxes 3.3 m tall at p90 on WATO. Without a ground
    # grid, height is over the group's lowest point.
    max_height_m: dict[str, float] = Field(default_factory=dict)
    # A box with at least this share of its BEV footprint inside a same-class
    # box with more points in the frame is dropped (DetZero's nested-box
    # filter): a car hidden behind another is boxed on its occluder's edge
    # returns, at the occluder's depth (docs/proposal_generation_measurements.md,
    # "Occlusion"), so point count, not depth, tells them apart. > 1 disables.
    max_nested_fraction: float = 0.3
    # Fewer points than this in a frame: no box.
    min_points: int = 5


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
