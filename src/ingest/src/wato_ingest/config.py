"""Ingest configuration loaded from config/pipeline.yaml.

Schema mirrors the `ingest` block of pipeline.yaml plus a few derived
fields (per-bag input path, calibration source path, topic mapping) that are
populated from the CLI or per-bag overrides.
"""

from __future__ import annotations

from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class TopicMap(BaseModel):
    """Maps the bag's actual topic names to the logical names ingest uses.

    Override per bag if your recording uses different topic names.
    """
    model_config = ConfigDict(extra="forbid")

    cameras: dict[str, str] = Field(
        default_factory=lambda: {
            "CAM_FRONT":       "/cam_front/image_rect_compressed",
            "CAM_FRONT_LEFT":  "/cam_front_left/image_rect_compressed",
            "CAM_FRONT_RIGHT": "/cam_front_right/image_rect_compressed",
            "CAM_LEFT":        "/cam_left/image_rect_compressed",
            "CAM_RIGHT":       "/cam_right/image_rect_compressed",
            "CAM_BACK":        "/cam_back/image_rect_compressed",
            "CAM_BACK_LEFT":   "/cam_back_left/image_rect_compressed",
            "CAM_BACK_RIGHT":  "/cam_back_right/image_rect_compressed",
        }
    )
    lidars: dict[str, str] = Field(
        default_factory=lambda: {"LIDAR_TOP": "/lidar_top/points"}
    )
    tf: str = "/tf"
    tf_static: str = "/tf_static"
    odom: str = "/odom"


class IngestConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    chunk_seconds: float = 30.0
    chunk_overlap_seconds: float = 2.0
    reference_clock: str = "lidar"
    max_cam_offset_ms: float = 50.0
    storage_id: str = "sqlite3"
    topics: TopicMap = Field(default_factory=TopicMap)
    quality_thresholds: dict[str, float] = Field(default_factory=lambda: {
        "low_light_v_max": 50.0,
        "stationary_speed_mps": 0.5,
        "min_pose_availability": 0.95,
        "max_camera_drop_fraction": 0.10,
        "min_lidar_points": 10000,
    })
    upstream_versions: dict[str, str] = Field(default_factory=dict)

def load_config(path: str) -> IngestConfig:
    with open(path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    section = data.get("ingest", {})
    return IngestConfig.model_validate(section)
