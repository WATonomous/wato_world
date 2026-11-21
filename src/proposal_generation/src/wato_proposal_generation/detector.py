"""fusion_detector: detection_3d's BEVFusion boxes as proposals.

BEVFusion predicts each object's whole box — the sides LiDAR never hit and
the extent a partial mask can't show — so its rows keep the model's geometry
rather than a box refit to the points inside. Which points those are is still
recorded (members), because fusion matches sources by the points they share.

The model fuses camera and LiDAR, so its rows are not independent of the
lifted source's camera evidence; provenance `fusion_detector` keeps it apart
from a LiDAR-only detector (`lidar_detector`).
"""

from __future__ import annotations

import math

from wato_common.artifacts.detections_3d import Detection3D
from wato_common.artifacts.lidar_sweeps import Frame
from wato_common.schemas import Box3D
from wato_common.tracking import points_in_box
from wato_common.tracking.boxes import wrap_heading_half_pi
from wato_proposal_generation.boxes import SourceBox, proposal
from wato_proposal_generation.config import FusionDetectorConfig
from wato_proposal_generation.frame_points import FramePoints

PROVENANCE = "fusion_detector"


def kept(d: Detection3D, cfg: FusionDetectorConfig) -> bool:
    """A detection proposal_generation proposes: scored, in the taxonomy,
    outside the ego zone (measured in the model's input frame, as the car)."""
    return (
        d.score >= cfg.min_score
        and d.cls in cfg.class_map
        and math.hypot(d.model_x, d.model_y) >= cfg.ego_suppression_radius_m
    )


def pipeline_box(d: Detection3D) -> Box3D:
    """The detection in the proposals' box convention: l the longer BEV side,
    heading undirected in (-pi/2, pi/2] (wato_common.tracking.boxes). The
    directed heading stays in detections_3d."""
    length, width, heading = d.l, d.w, d.heading
    if width > length:
        length, width, heading = width, length, heading + math.pi / 2.0
    return Box3D(
        cx=d.cx,
        cy=d.cy,
        cz=d.cz,
        w=width,
        l=length,
        h=d.h,
        heading=wrap_heading_half_pi(heading),
    )


def proposals(
    frame: Frame,
    detections: list[Detection3D],
    fp: FramePoints,
    cfg: FusionDetectorConfig,
) -> list[SourceBox]:
    """One fusion_detector row per kept detection of the frame, with the
    frame's non-ground points inside its box."""
    out = []
    for d in detections:
        if not kept(d, cfg):
            continue
        box = pipeline_box(d)
        members = points_in_box(fp.xyz, box)
        out.append(
            SourceBox(
                proposal(
                    d.detection_id,
                    frame,
                    box,
                    cls=cfg.class_map[d.cls],
                    score=d.score,
                    provenance=PROVENANCE,
                    lidar_point_count=int(members.sum()),
                ),
                members,
            )
        )
    return out
