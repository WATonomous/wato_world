"""Pydantic config for label_refinement, sourced from label_refinement.yaml."""

from __future__ import annotations

from typing import Any, Literal, Optional

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
    # Boxes of these classes keep their observed centre instead of their
    # corner nearest the ego: a person's returns surround their centre (the
    # observed box's centre is within 0.1 m of the ground truth's, radially,
    # on nuScenes), and growing the box away from the ego put it 0.2-0.3 m
    # behind them (docs/label_refinement_measurements.md).
    centred_classes: list[str] = Field(default_factory=lambda: ["pedestrian"])
    # Heading from the smoothed velocity when the track moves at least this
    # fast at that instant (VESPA's 0.5 m/s); slower, the boxes' own heading.
    heading_min_speed_mps: float = 0.5
    # ...and at least heading_speed_z standard errors of that velocity (the
    # local linear fit's): a speed the centres' jitter explains is no heading.
    # 0 disables. Unlike a higher min speed, it adapts to each recording's
    # centre noise.
    heading_speed_z: float = Field(default=3.0, ge=0)
    # Centres are smoothed by a local linear fit over this window.
    smooth_window_s: float = 1.0


class SlfConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # SLF (slf.py): a box shape fit per track to its masks and LiDAR points,
    # replacing Phase 3's size where the fit passes the gates below.
    enabled: bool = True
    classes: list[str] = Field(
        default_factory=lambda: [
            "car",
            "truck",
            "bus",
            "pedestrian",
            "bicycle",
            "motorcycle",
        ]
    )
    # "auto" (CUDA if torch sees a GPU, else CPU), "cuda" or "cpu".
    device: str = "auto"

    # Views: images where one of the track's masklets has a mask, at most
    # max_views of them spread over the track's life (the largest projected
    # box in each stretch); a track with fewer than min_views is not fit.
    max_views: int = 16
    min_views: int = 3
    min_mask_px: int = 64
    # Each view is a crop_px x crop_px grid of rays over the box's and the
    # mask's bounding rectangle, grown by crop_margin of its size on each side.
    crop_px: int = Field(default=48, ge=2)
    crop_margin: float = 0.2
    # Samples along each ray, spanning the box's half diagonal + 0.5 m about
    # its centre.
    ray_samples: int = Field(default=32, ge=2)
    # The silhouette's soft edge, in crop pixels.
    edge_px: float = Field(default=1.0, gt=0)
    # Edges rounded by this share of the box's smallest half extent.
    corner_radius_frac: float = 0.2
    # Another object's mask hides the track where its box is this much nearer
    # the camera (SLF's occlusion map); those pixels are left out of the Dice.
    occluder_margin_m: float = 0.5
    # LiDAR returns inside each view frame's observed box grown by
    # point_margin_m, at most max_points of them.
    point_margin_m: float = 0.15
    max_points: int = 2048

    # Fit: Adam over log size and a shared offset (SLF's settings), this many
    # tracks per fit: each step's cost is in launching it, not its size, so a
    # batch takes about as long as one track (on the GPU).
    batch_tracks: int = 32
    iterations: int = 150
    lr: float = 0.1
    weight_mask: float = 1.0
    # |SDF| at the points, robust (Charbonnier) below lidar_robust_m. The
    # returns of enclosing_classes need only lie inside the box: a person's
    # box encloses arms and legs, and pulling its faces onto the front
    # surface's returns put it 0.2 m behind them (radially, on nuScenes).
    weight_lidar_per_m: float = 2.0
    lidar_robust_m: float = 0.05
    enclosing_classes: list[str] = Field(default_factory=lambda: ["pedestrian"])
    # A weak pull of log size to the class's mean size (l, w, h; the
    # nuScenes means CenterPoint's anchors use), sigma in log units.
    weight_size_prior: float = 0.01
    size_prior_sigma: float = 0.25
    size_prior_m: dict[str, list[float]] = Field(
        default_factory=lambda: {
            "car": [4.63, 1.97, 1.74],
            "truck": [6.94, 2.51, 2.84],
            "bus": [11.19, 2.94, 3.47],
            "pedestrian": [0.73, 0.67, 1.77],
            "bicycle": [1.70, 0.60, 1.28],
            "motorcycle": [2.11, 0.77, 1.47],
        }
    )
    weight_offset: float = 0.01
    offset_sigma_m: float = 0.3
    min_size_m: float = 0.1
    max_size_m: float = 20.0

    # Heading (slf.py): Phase 3's plus a correction fit with the shape, for
    # tracks of fit_yaw_classes (an elongated footprint's silhouette says
    # which way it points; a pedestrian's doesn't). A parked vehicle's one
    # box gets one angle; any other track's correction is piecewise linear in
    # time between knots at most yaw_knot_spacing_s apart (null: one angle per
    # track). Adam steps it at yaw_lr (radians), it is clamped to
    # ±max_yaw_correction_deg, and weak priors pull each knot to zero
    # (yaw_prior_sigma_deg) and neighbouring knots together
    # (yaw_step_sigma_deg). Measurements: docs/label_refinement_measurements.md.
    fit_yaw_classes: list[str] = Field(
        default_factory=lambda: ["car", "truck", "bus", "bicycle", "motorcycle"]
    )
    yaw_knot_spacing_s: Optional[float] = Field(default=2.0, gt=0)
    yaw_lr: float = Field(default=0.02, gt=0)
    weight_yaw_prior: float = Field(default=0.01, ge=0)
    yaw_prior_sigma_deg: float = Field(default=10.0, gt=0)
    yaw_step_sigma_deg: float = Field(default=5.0, gt=0)
    max_yaw_correction_deg: float = Field(default=30.0, ge=0, le=90)

    # A fit replaces Phase 3's size only if it explains the masks better than
    # Phase 3's box by at least min_silhouette_gain (the drop in the mean
    # 1 − IoU over the views): where Phase 3's box already matches them, the
    # fit mostly shrinks a full-size box to masks drawn tight
    # (docs/label_refinement_measurements.md). A pedestrian's masks are a few
    # pixels wide and its LiDAR returns a sliver, so its fit is kept whatever
    # the gain: the prior gives the depth and width neither sees. Every fit
    # also needs the median |SDF| at its points and its silhouette residual
    # at most these, against masks of another object.
    min_silhouette_gain: float = 0.15
    min_silhouette_gain_by_class: dict[str, float] = Field(
        default_factory=lambda: {"pedestrian": -1.0}
    )
    max_residual_silhouette: float = 0.8
    max_residual_lidar_m: float = 0.3


class ComponentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calibration_source: Literal["recorded", "validated"] = "recorded"
    frames: FramesConfig = Field(default_factory=FramesConfig)
    parked: ParkedConfig = Field(default_factory=ParkedConfig)
    moving: MovingConfig = Field(default_factory=MovingConfig)
    slf: SlfConfig = Field(default_factory=SlfConfig)


def load_config(path: str) -> ComponentConfig:
    with open(path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    section = data.get("label_refinement", data)
    return ComponentConfig.model_validate(section)
