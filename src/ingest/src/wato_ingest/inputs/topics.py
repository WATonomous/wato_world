"""Validates that a bag has the topics ingest needs."""

from __future__ import annotations

from dataclasses import dataclass, field

from wato_ingest.config import IngestConfig


@dataclass
class TopicValidationResult:
    ok: bool
    missing: list[str]
    found_camera_image_topics: list[str]
    found_camera_info_topics: list[str]
    found_lidar_topics: list[str]
    found_pose_topics: list[str]
    found_tf_static_topics: list[str] = field(default_factory=list)


def validate(bag_topics: dict[str, str], cfg: IngestConfig) -> TopicValidationResult:
    """Check that every required topic exists in the bag.

    `bag_topics` is {topic_name: type_str} pulled from the bag's metadata.
    Returns a result describing what was found and what is missing.

    Required:
      - every camera's `image` and `info` topic
      - every lidar topic
      - the pose topic
      - /tf_static (drives calibration extrinsics)
    """
    image_topics = [c.image for c in cfg.topics.cameras.values()]
    info_topics = [c.info for c in cfg.topics.cameras.values()]
    lidars = list(cfg.topics.lidars.values())
    pose_topics = [cfg.topics.pose]
    tf_static = [cfg.topics.tf_static]

    missing: list[str] = []
    found_image: list[str] = []
    found_info: list[str] = []
    found_lidars: list[str] = []
    found_poses: list[str] = []
    found_tf: list[str] = []

    def _check(topics, found, label):
        for t in topics:
            (found if t in bag_topics else missing).append(t)

    _check(image_topics, found_image, "camera image")
    _check(info_topics, found_info, "camera info")
    _check(lidars, found_lidars, "lidar")
    _check(pose_topics, found_poses, "pose")
    _check(tf_static, found_tf, "tf_static")

    return TopicValidationResult(
        ok=len(missing) == 0,
        missing=missing,
        found_camera_image_topics=found_image,
        found_camera_info_topics=found_info,
        found_lidar_topics=found_lidars,
        found_pose_topics=found_poses,
        found_tf_static_topics=found_tf,
    )
