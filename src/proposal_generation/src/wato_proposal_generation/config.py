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


class LiftedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Points of one instance chained within this distance are one group; only
    # the largest group is boxed (masklets bleed onto background). Lifted
    # objects are far and sparse (median 34-51 m on WATO): 0.75 m splits real
    # cars (docs/proposal_generation_measurements.md).
    cluster_radius_m: float = 2.0
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
