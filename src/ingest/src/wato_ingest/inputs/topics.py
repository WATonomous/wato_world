"""Validates that a bag has the topics ingest needs, with usable message types.

A configured topic that is missing, or that carries a type ingest can't
decode, would otherwise come out as an empty camera, LiDAR or pose stream with
no error, so both are checked up front against the bag's metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from wato_ingest.config import IngestConfig

# Message types each configured role accepts (ROS 2 "pkg/msg/Type" names).
IMAGE_TYPES = frozenset({"sensor_msgs/msg/CompressedImage", "sensor_msgs/msg/Image"})
CAMERA_INFO_TYPES = frozenset({"sensor_msgs/msg/CameraInfo"})
LIDAR_TYPES = frozenset({"sensor_msgs/msg/PointCloud2"})
POSE_TYPES = frozenset(
    {
        "nav_msgs/msg/Odometry",
        "geometry_msgs/msg/PoseStamped",
        "geometry_msgs/msg/PoseWithCovarianceStamped",
    }
)
TF_TYPES = frozenset({"tf2_msgs/msg/TFMessage"})


@dataclass
class TopicValidationResult:
    ok: bool
    missing: list[str]
    found_camera_image_topics: list[str]
    found_camera_info_topics: list[str]
    found_lidar_topics: list[str]
    found_pose_topics: list[str]
    found_tf_static_topics: list[str] = field(default_factory=list)
    # "<topic> is <type>, expected one of [...]" per wrongly typed topic.
    wrong_type: list[str] = field(default_factory=list)

    def describe(self) -> str:
        parts = []
        if self.missing:
            parts.append(f"missing topics {self.missing}")
        if self.wrong_type:
            parts.append("unsupported message types: " + "; ".join(self.wrong_type))
        return ", ".join(parts)


def normalize_type(type_str: str) -> str:
    """ "sensor_msgs/Image" (ROS 1 style) → "sensor_msgs/msg/Image"."""
    parts = type_str.split("/")
    if len(parts) == 2:
        return f"{parts[0]}/msg/{parts[1]}"
    return type_str


def validate(
    bag_topics: dict[str, str],
    cfg: IngestConfig,
    *,
    calibration_from_bag: bool = True,
) -> TopicValidationResult:
    """Check that every required topic exists in the bag with a usable type.

    `bag_topics` is {topic_name: type_str} pulled from the bag's metadata; an
    empty type string skips the type check for that topic.

    Required:
      - every camera's `image` topic
      - every lidar topic
      - the pose topic
      - when `calibration_from_bag`: every camera's `info` topic and the
        `tf_static` topic (they drive calibration.json).  With an authored
        calibration file neither is read, so neither is required.
    """
    missing: list[str] = []
    wrong_type: list[str] = []

    def _check(topics: list[str], accepted: frozenset[str]) -> list[str]:
        found = []
        for t in topics:
            if t not in bag_topics:
                missing.append(t)
                continue
            found.append(t)
            got = bag_topics[t]
            if got and normalize_type(got) not in accepted:
                wrong_type.append(f"{t} is {got}, expected one of {sorted(accepted)}")
        return found

    cameras = cfg.topics.cameras
    found_image = _check([c.image for c in cameras.values()], IMAGE_TYPES)
    found_lidars = _check(list(cfg.topics.lidars.values()), LIDAR_TYPES)
    found_poses = _check([cfg.topics.pose], POSE_TYPES)

    found_info: list[str] = []
    found_tf: list[str] = []
    if calibration_from_bag:
        for cam_id, c in cameras.items():
            if c.info is None:
                missing.append(f"<topics.cameras.{cam_id}.info not configured>")
        found_info = _check(
            [c.info for c in cameras.values() if c.info is not None],
            CAMERA_INFO_TYPES,
        )
        if cfg.topics.tf_static is None:
            missing.append("<topics.tf_static not configured>")
        else:
            found_tf = _check([cfg.topics.tf_static], TF_TYPES)

    return TopicValidationResult(
        ok=not missing and not wrong_type,
        missing=missing,
        found_camera_image_topics=found_image,
        found_camera_info_topics=found_info,
        found_lidar_topics=found_lidars,
        found_pose_topics=found_poses,
        found_tf_static_topics=found_tf,
        wrong_type=wrong_type,
    )
