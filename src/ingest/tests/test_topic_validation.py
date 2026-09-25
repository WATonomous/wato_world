"""Tests for inputs/topics.validate: required topics must exist AND carry a
message type ingest can decode, checked against the bag's metadata."""

from __future__ import annotations

from pathlib import Path

from wato_ingest.config import load_config
from wato_ingest.inputs.topics import normalize_type, validate

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def _cfg():
    cfg = load_config(str(CONFIG_DIR / "ingest.yaml"))
    topics = cfg.topics.model_copy(
        update={
            "cameras": {"CAM": cfg.topics.cameras["CAM_FRONT"]},
            "lidars": {"LIDAR": "/LIDAR_TOP"},
        }
    )
    return cfg.model_copy(update={"topics": topics})


def _bag(**overrides) -> dict[str, str]:
    topics = {
        "/CAM_FRONT/image_rect_compressed": "sensor_msgs/msg/CompressedImage",
        "/CAM_FRONT/camera_info": "sensor_msgs/msg/CameraInfo",
        "/LIDAR_TOP": "sensor_msgs/msg/PointCloud2",
        "/odom": "nav_msgs/msg/Odometry",
        "/tf": "tf2_msgs/msg/TFMessage",
    }
    topics.update(overrides)
    return {t: ty for t, ty in topics.items() if ty is not None}


def test_matching_bag_passes():
    assert validate(_bag(), _cfg()).ok


def test_missing_topic_is_reported():
    r = validate(_bag(**{"/LIDAR_TOP": None}), _cfg())
    assert not r.ok and r.missing == ["/LIDAR_TOP"]


def test_wrong_message_type_is_reported():
    # A LaserScan would otherwise decode to zero sweeps without an error.
    r = validate(_bag(**{"/LIDAR_TOP": "sensor_msgs/msg/LaserScan"}), _cfg())
    assert not r.ok and not r.missing
    assert r.wrong_type[0].startswith("/LIDAR_TOP is sensor_msgs/msg/LaserScan")
    assert "unsupported message types" in r.describe()


def test_raw_image_and_every_pose_type_are_accepted():
    for pose_type in (
        "nav_msgs/msg/Odometry",
        "geometry_msgs/msg/PoseStamped",
        "geometry_msgs/msg/PoseWithCovarianceStamped",
    ):
        bag = _bag(
            **{
                "/odom": pose_type,
                "/CAM_FRONT/image_rect_compressed": "sensor_msgs/msg/Image",
            }
        )
        assert validate(bag, _cfg()).ok, pose_type


def test_ros1_style_type_names_are_normalized():
    assert normalize_type("sensor_msgs/Image") == "sensor_msgs/msg/Image"
    assert validate(_bag(**{"/odom": "nav_msgs/Odometry"}), _cfg()).ok


def test_unknown_type_skips_the_type_check():
    assert validate({t: "" for t in _bag()}, _cfg()).ok


def test_calibration_file_makes_info_and_tf_optional():
    bag = _bag(**{"/CAM_FRONT/camera_info": None, "/tf": None})
    assert not validate(bag, _cfg()).ok
    assert validate(bag, _cfg(), calibration_from_bag=False).ok


def test_unconfigured_info_topic_fails_only_when_calibrating_from_bag():
    cfg = _cfg()
    cam = cfg.topics.cameras["CAM"].model_copy(update={"info": None})
    topics = cfg.topics.model_copy(update={"cameras": {"CAM": cam}, "tf_static": None})
    cfg = cfg.model_copy(update={"topics": topics})
    r = validate(_bag(), cfg)
    assert r.missing == [
        "<topics.cameras.CAM.info not configured>",
        "<topics.tf_static not configured>",
    ]
    assert validate(_bag(), cfg, calibration_from_bag=False).ok
