"""Pydantic config for tracking, sourced from tracking.yaml."""

from __future__ import annotations

from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class AssociationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Chen et al.'s gates (wato_common.tracking.AssociationGates): centre
    # distance, 1 − IoU, 1 − volume ratio. A pair over any of them starts a
    # new track.
    max_centre_dist_m: float = 2.0
    max_one_minus_iou: float = 0.95
    max_one_minus_volume_ratio: float = 0.7
    # A track that misses frames keeps coasting on its prediction this long
    # (Immortal Tracker: ending tracks early causes most ID switches); only
    # the frames it matched are output.
    max_coast_s: float = 2.0
    # Cost taken off a pair whose box comes from the same lifted instance as
    # the track's last box: SAM2 already followed that object across frames.
    instance_bonus: float = 1.0
    # Such a pair is held only to the distance gate: a one-sided box's IoU and
    # volume swing as its visible face changes, and the shape gates exist to
    # stop jumps to a neighbour, which the shared identity already rules out.
    instance_opens_shape_gates: bool = True


class StrongConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # A box is strong when two sources agree on it (a fused lidar_mos + lifted
    # pair), or when its lifted box has at least this confidence and this many
    # points. Strong boxes are matched to tracks first (SimpleTrack), so a weak
    # box can't take a track a strong one fits.
    min_lift_confidence: float = 0.5
    min_lift_points: int = 20


class MotionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Fewer frames than this: motion state "unknown" (3DAL classifies tracks
    # of ≥ 7 boxes).
    min_frames: int = 7
    # Moving iff the Kalman-filtered centre moved at least
    # moving_displacement_m first → last (3DAL's and MS3D's 1 m), or, on a
    # fragment too short to cover that, at least fast_displacement_m at an
    # average of at least fast_speed_mps: a walker covers 0.6 m in a 0.4 s
    # fragment. MS3D's centre-variance test is left out: a one-sided box's
    # centre shifts as its visible face changes, which read parked cars and
    # trucks as moving on nuScenes (docs/tracking_measurements.md).
    moving_displacement_m: float = 1.0
    fast_displacement_m: float = 0.5
    fast_speed_mps: float = 0.5
    # Frames medianed at each end for the net displacement.
    endpoint_frames: int = 3


class FilterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # A track is dropped only when it is both short and weak (Offline-Poly):
    # fewer frames than min_frames AND fewer strong frames than min_strong.
    min_frames: int = 5
    min_strong: int = 1


class ComponentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    association: AssociationConfig = Field(default_factory=AssociationConfig)
    strong: StrongConfig = Field(default_factory=StrongConfig)
    motion: MotionConfig = Field(default_factory=MotionConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)


def load_config(path: str) -> ComponentConfig:
    with open(path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    section = data.get("tracking", data)
    return ComponentConfig.model_validate(section)
