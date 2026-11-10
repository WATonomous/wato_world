"""Pydantic config for tracking, sourced from tracking.yaml."""

from __future__ import annotations

from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class AssociationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # "chen": Chen et al.'s cost and gates (wato_common.tracking, Step F's):
    # centre distance, 1 − IoU, 1 − volume ratio; a pair over any of them
    # starts a new track. "giou_3d": SimpleTrack's 1 − GIoU (3D) with one gate,
    # max_one_minus_giou (GIoU > −0.5 for every class on nuScenes and WOD):
    # IoU is 0 for a one-sided sliver beside the track's whole box, GIoU still
    # ranks it by how far apart they are.
    cost: Literal["chen", "giou_3d"] = "giou_3d"
    max_centre_dist_m: float = 2.0
    max_one_minus_iou: float = 0.95
    max_one_minus_volume_ratio: float = 0.7
    max_one_minus_giou: float = 1.5
    # A track that misses frames keeps coasting on its prediction this long;
    # None: until the chunk ends (Immortal Tracker: ending tracks early causes
    # most ID switches). Only the frames it matched are output.
    max_coast_s: Optional[float] = None
    # Cost taken off a pair whose box comes from the same lifted instance as
    # the track's last box: SAM2 already followed that object across frames.
    instance_bonus: float = 1.0
    # Such a pair is held only to the distance gate (max_centre_dist_m): a
    # one-sided box's IoU and volume swing as its visible face changes, and
    # the shape gates (or the GIoU gate) exist to stop jumps to a neighbour,
    # which the shared identity already rules out.
    instance_opens_shape_gates: bool = True


class DuplicateGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classes: list[str]
    min_radius: float  # squared BEV centre distance, m² (CenterPoint's unit)


class DuplicatesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # CenterPoint's circle NMS (det3d/core/utils/circle_nms_jit.py), before
    # association, per frame: within one group, a fused box whose BEV centre
    # lies within sqrt(min_radius) of a higher-ranked kept box is dropped. The
    # radii are CenterPoint's nuScenes task heads
    # (nusc_centerpoint_pp_02voxel_two_pfn_10sweep_circular_nms.py) for the
    # vehicle classes: [car] 4, [truck] 12, [bus] 10. Its [motorcycle,
    # bicycle] 0.85 and [pedestrian] 0.175 heads are left out: on scene-1100
    # the pedestrian one removed real neighbours (docs/tracking_measurements.md).
    # A box with no class (lidar_mos only) is in no group. Ranked strong
    # first, then by LiDAR points: CenterPoint ranks by detector score, and the
    # fused score is the same for every box.
    enabled: bool = True
    groups: list[DuplicateGroup] = Field(
        default_factory=lambda: [
            DuplicateGroup(classes=["car"], min_radius=4.0),
            DuplicateGroup(classes=["truck"], min_radius=12.0),
            DuplicateGroup(classes=["bus"], min_radius=10.0),
        ]
    )

    @model_validator(mode="after")
    def _one_group_per_class(self) -> "DuplicatesConfig":
        seen: set[str] = set()
        for g in self.groups:
            twice = seen.intersection(g.classes)
            if twice:
                raise ValueError(f"classes {sorted(twice)} are in two groups")
            seen.update(g.classes)
        return self


class RelinkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Offline-Poly's STWO (arXiv 2602.13772 §IV-C) after the forward pass
    # (relink.py): two of a chunk's tracks that never exist at once are fused
    # when, at a frame, their observed or predicted boxes overlap, one to one
    # by the blossom algorithm.
    enabled: bool = True
    # "Predictions exceeding 1 second are discarded".
    horizon_s: float = 1.0
    # θ_blo: fuse at cost 1 − similarity below this; "optimal performance
    # occurs under a moderately relaxed threshold (around 0.9)" (§V-E).
    max_cost: float = 0.9
    # Eq. 3's IoU, in BEV or 3D; giou_3d is the alternative of Table VIII.
    metric: Literal["iou_bev", "iou_3d", "giou_3d"] = "iou_bev"
    # Never fuse two tracks whose classes are both known and differ.
    class_consistent: bool = True
    # Tracks sharing a lifted instance fuse whatever their IoU when their
    # states are within association.max_centre_dist_m.
    shared_instance: bool = True


class StrongConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # A box is strong when two sources agree on it (a fused lidar_mos + lifted
    # pair), or when its lifted box has at least this confidence and this many
    # retained original seed points, excluding geometry acquired by scan
    # refinement. Legacy proposals use their total labeled count. Strong
    # boxes are matched to tracks first (SimpleTrack), so a weak
    # box can't take a track a strong one fits.
    min_lift_confidence: float = 0.5
    min_lift_points: int = 20
    # A box backed by a fusion_detector row scored at least this is strong:
    # ByteTrack's high-score split (track_thresh), the value the car's tracker
    # runs on BEVFusion (wato_monorepo tracking params.yaml). Lower-scored
    # detector boxes are matched only after the strong ones, as ByteTrack's
    # low-score boxes are.
    min_detector_score: float = 0.5


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


class StitchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # A track of chunk k and one of chunk k+1 are one object when, at the
    # instants of their shared span both saw, their boxes' centres are within
    # max_centre_dist_m at no fewer than min_agreeing_frames instants, and at
    # no less than min_agreement of the instants the shorter of the two saw
    # there (SUSHI's shared-frame link); one to one, the pairs agreeing at the
    # most instants in total win. Both chunks decode the same sweeps there, so
    # the same object's box is usually identical in both
    # (docs/tracking_measurements.md).
    max_centre_dist_m: float = 1.0
    min_agreement: float = 0.5
    min_agreeing_frames: int = 3


class ComponentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    association: AssociationConfig = Field(default_factory=AssociationConfig)
    duplicates: DuplicatesConfig = Field(default_factory=DuplicatesConfig)
    relink: RelinkConfig = Field(default_factory=RelinkConfig)
    strong: StrongConfig = Field(default_factory=StrongConfig)
    motion: MotionConfig = Field(default_factory=MotionConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
    stitch: StitchConfig = Field(default_factory=StitchConfig)


def load_config(path: str) -> ComponentConfig:
    with open(path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    section = data.get("tracking", data)
    return ComponentConfig.model_validate(section)
