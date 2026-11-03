"""Pydantic config for label_refinement, sourced from label_refinement.yaml."""

from __future__ import annotations

from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field


class FramesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # The LiDAR lidar_preprocessing's frame_sync anchored frames on (as
    # proposal_generation's frames.canonical_lidar): a track frame's points are
    # every sweep of its frame.
    canonical_lidar: Optional[str] = None


class ParkedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # A parked track's points are gathered from every frame inside the
    # rectangle enclosing all its boxes, grown by this much: each frame's box
    # covers the face visible then, and the car doesn't move (3DAL).
    region_margin_m: float = 0.3
    # Only boxes centred within this of the track's median centre shape the
    # rectangle: a parked track moved under 1 m, and one stray box (a frame
    # where it took a neighbour's) would otherwise pull the neighbour in.
    region_max_centre_offset_m: float = 1.0
    # Accumulated points are dense, so a short chain radius keeps the car and
    # leaves a neighbour parked a metre away.
    cluster_radius_m: float = 0.5
    # Deduplication grid for the accumulated points (runtime only).
    voxel_m: float = 0.05
    # Fewer accumulated points than this: refined like a moving track.
    min_points: int = 30
    # Only these classes are aggregated. A standing person still moves (3DAL
    # treats every pedestrian as moving), and a small object's region collects
    # its neighbours' points; class-less tracks have no shape to trust.
    aggregate_classes: list[str] = Field(
        default_factory=lambda: ["car", "truck", "bus"]
    )


class MovingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # One size per track: this percentile of l, w and h over its densest
    # frames (the share densest_fraction by point count). Far frames see one
    # face, so their boxes are small; the percentile takes the near ones
    # (LISO's 90th, OYSTER's 95-100th).
    size_percentile: float = 90.0
    densest_fraction: float = 0.5
    # Per-class override. A person's box isn't one face of a larger object,
    # and its high percentile is its frames with the most bleed.
    size_percentile_by_class: dict[str, float] = Field(
        default_factory=lambda: {
            "pedestrian": 50.0,
            "bicycle": 50.0,
            "motorcycle": 50.0,
        }
    )
    # Heading from the smoothed velocity when the track moves at least this
    # fast at that instant (VESPA's 0.5 m/s); slower, the boxes' own heading.
    heading_min_speed_mps: float = 0.5
    # Centres are smoothed by a local linear fit over this window.
    smooth_window_s: float = 1.0


class ComponentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frames: FramesConfig = Field(default_factory=FramesConfig)
    parked: ParkedConfig = Field(default_factory=ParkedConfig)
    moving: MovingConfig = Field(default_factory=MovingConfig)


def load_config(path: str) -> ComponentConfig:
    with open(path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    section = data.get("label_refinement", data)
    return ComponentConfig.model_validate(section)
