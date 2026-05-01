"""Validates that a bag has the topics ingest needs."""

from __future__ import annotations

from dataclasses import dataclass

from wato_ingest.config import IngestConfig


@dataclass
class TopicValidationResult:
    ok: bool
    missing: list[str]
    found_camera_topics: list[str]
    found_lidar_topics: list[str]
    found_pose_topics: list[str]


def validate(bag_topics: dict[str, str], cfg: IngestConfig) -> TopicValidationResult:
    """Check that every required topic exists in the bag.

    `bag_topics` is {topic_name: type_str} pulled from the bag's metadata.
    Returns a result describing what was found and what is missing.
    """
    cameras = list(cfg.topics.cameras.values())
    lidars = list(cfg.topics.lidars.values())
    pose_topics = [cfg.topics.tf]  # tf_static and odom are nice-to-have, not required

    missing: list[str] = []
    found_cameras: list[str] = []
    found_lidars: list[str] = []
    found_poses: list[str] = []

    for t in cameras:
        (found_cameras if t in bag_topics else missing).append(t)
    for t in lidars:
        (found_lidars if t in bag_topics else missing).append(t)
    for t in pose_topics:
        (found_poses if t in bag_topics else missing).append(t)

    return TopicValidationResult(
        ok=len(missing) == 0,
        missing=missing,
        found_camera_topics=found_cameras,
        found_lidar_topics=found_lidars,
        found_pose_topics=found_poses,
    )
