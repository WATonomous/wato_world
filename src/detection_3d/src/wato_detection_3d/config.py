"""Pydantic config for detection_3d, sourced from detection_3d.yaml."""

from __future__ import annotations

from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# The ONNX files BEVFusion's TensorRT engines are built from, the car's
# (wato_monorepo bevfusion_core.cpp): four engines and the LiDAR backbone,
# which libspconv reads as ONNX.
MODEL_FILES = (
    "camera.backbone.onnx",
    "camera.vtransform.onnx",
    "fuser.onnx",
    "head.bbox.onnx",
    "lidar.backbone.xyz.onnx",
)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # The car's model directory (wato_monorepo bevfusion params.yaml
    # model_dir), mounted read-only.
    dir: str = "/data/models/bevfusion/resnet50_trt11_int8"
    # The LiDAR backbone's precision; the car's default (BEVFusionInputConfig).
    precision: Literal["int8", "fp16"] = "int8"
    # Must match the exported camera backbone and the runtime's buffers.
    num_cameras: int = Field(default=6, ge=1)
    # sha256 of each MODEL_FILES file. A different file is a different
    # labeler, so a mismatch stops the run rather than labelling with it.
    sha256: dict[str, str] = Field(default_factory=dict)
    # Writable. Engines are built here once per (model files, TensorRT
    # version, GPU) and reused.
    engine_cache_dir: str = "/data/models/.cache/tensorrt"

    @field_validator("sha256")
    @classmethod
    def every_file_pinned(cls, value: dict[str, str]) -> dict[str, str]:
        missing = [f for f in MODEL_FILES if f not in value]
        if missing:
            raise ValueError(f"sha256 must pin {missing}")
        return value


class FramesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # lidar_preprocessing's frame_sync.canonical_lidar for the bag (restated:
    # no artifact records it). Each frame is run once, at its reference sweep.
    canonical_lidar: Optional[str] = None


class PointsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Whose sweeps feed the model. None: the canonical LiDAR only (the frame's
    # own LiDAR on a bag without one), as the car feeds lidar_cc alone. Adding
    # WATO's VLP-16 lidar_ne cut agreement with lifted objects from 0.46 to
    # 0.04 (docs/detection_3d_measurements.md).
    lidars: Optional[list[str]] = None
    # BEVFusion's nuScenes config accumulates the current sweep and 9 earlier
    # ones (mmdetection3d LoadPointsFromMultiSweeps, sweeps_num 9), each point
    # tagged with its sweep's age in seconds (the 5th input feature). Offline
    # we have those sweeps; the car feeds one, with age 0.
    past_sweeps: int = Field(default=9, ge=0)
    # Ages the model was trained on: 9 earlier nuScenes sweeps at 20 Hz reach
    # ~0.45 s. A 10 Hz LiDAR gets the sweeps within this age, not 9.
    max_time_lag_s: float = Field(default=0.5, ge=0.0)
    # LoadPointsFromMultiSweeps' remove_close: an earlier sweep's points
    # within this |x| and |y| of its own sensor (the ego vehicle) are dropped.
    remove_close_m: float = Field(default=1.0, ge=0.0)


class InputFrameConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # nuscenes_lidar: the frame BEVFusion was trained in, rebuilt on this rig
    # (geometry.py). sensor: the canonical LiDAR's own frame, as the car runs.
    kind: Literal["nuscenes_lidar", "sensor"] = "nuscenes_lidar"
    # nuScenes LIDAR_TOP's height above its ego frame (on the ground).
    height_m: float = 1.84


class ImageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # The model's camera input is a 1600x900 image (nuScenes'), which the
    # vendor resizes by 0.48 and crops from the bottom. A camera of another
    # shape is scaled to 1600 wide and a 900-high band of it is cut, placed
    # so its principal point lands on this row: the mean of the six nuScenes
    # cameras' cy (calibration of scenes 0061 / 1100).
    principal_row_px: float = 490.4


class CameraSyncConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Each camera's image nearest the frame's reference time, at most this
    # far from it: the car's sync_max_time_diff_ms (wato_monorepo bevfusion
    # params.yaml). Wider than ingest's frame_index pairing, which leaves a
    # 20 Hz LiDAR's sweep between two 12 Hz images without one; the ego's
    # motion over the gap is compensated (inputs.py), an object's is not.
    max_offset_ms: float = Field(default=200.0, gt=0.0)


class ComponentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calibration_source: Literal["recorded", "validated"] = "recorded"
    model: ModelConfig
    camera_sync: CameraSyncConfig = Field(default_factory=CameraSyncConfig)
    frames: FramesConfig = Field(default_factory=FramesConfig)
    points: PointsConfig = Field(default_factory=PointsConfig)
    input_frame: InputFrameConfig = Field(default_factory=InputFrameConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    # Ordered physical cameras, exactly model.num_cameras distinct IDs.
    cameras: list[str]

    @field_validator("cameras")
    @classmethod
    def distinct_cameras(cls, value: list[str]) -> list[str]:
        if not value or len(set(value)) != len(value):
            raise ValueError(f"cameras must name distinct cameras, got {value}")
        return value

    @model_validator(mode="after")
    def cameras_match_model(self):
        if len(self.cameras) != self.model.num_cameras:
            raise ValueError(
                f"cameras must name {self.model.num_cameras} distinct cameras "
                "to match model.num_cameras"
            )
        return self


def load_config(path: str) -> ComponentConfig:
    with open(path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    section = data.get("detection_3d", data)
    return ComponentConfig.model_validate(section)
