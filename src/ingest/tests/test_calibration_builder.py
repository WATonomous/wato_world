"""Tests for inputs/calibration: build_calibration_dict (the pure-logic part
that assembles calibration.json from already-decoded inputs) and
require_complete.  Doesn't touch the rosbag or the filesystem.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wato_common.geometry import make_se3
from wato_ingest.config import load_config
from wato_ingest.inputs import calibration
from wato_ingest.inputs.calibration import (
    CalibrationError,
    _resolve_chain,
    build_calibration_dict,
    require_complete,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def _identity():
    return np.eye(4)


def _xyz(x: float, y: float = 0.0, z: float = 0.0):
    return make_se3(np.array([x, y, z]), (0.0, 0.0, 0.0, 1.0))


def _camera_info(frame_id: str, fx: float = 1000.0):
    return {
        "frame_id": frame_id,
        "K": [fx, 0, 960, 0, fx, 540, 0, 0, 1],
        "D": [0.0, 0.0, 0.0, 0.0],
        "distortion_model": "plumb_bob",
        "width": 1920,
        "height": 1080,
    }


def test_resolves_direct_extrinsics():
    static = {
        ("base_footprint", "camera_lower_ne"): _xyz(2.0, 1.0, 1.5),
        ("base_footprint", "lidar_cc"): _xyz(0.5, 0.0, 1.8),
    }
    calib = build_calibration_dict(
        camera_infos={"CAM_LOWER_NE": _camera_info("camera_lower_ne", fx=1200)},
        lidar_frame_ids={"LIDAR_CC": "lidar_cc"},
        static_transforms=static,
        ego_frame="base_footprint",
        bag_id="b",
    )

    cam = calib["cameras"]["CAM_LOWER_NE"]
    assert cam["frame_id"] == "camera_lower_ne"
    assert cam["K"] == [[1200.0, 0, 960], [0, 1200.0, 540], [0, 0, 1]]
    assert cam["distortion_model"] == "plumb_bob"
    assert cam["ego_T_cam"][0][3] == 2.0  # x translation

    lid = calib["lidars"]["LIDAR_CC"]
    assert lid["frame_id"] == "lidar_cc"
    assert lid["ego_T_lidar"][0][3] == 0.5

    assert calib["checks"]["sanity"] == "auto"
    assert "all extrinsics resolved" in calib["checks"]["notes"]


def test_resolves_two_hop_extrinsic():
    # base -> roof_rack -> camera_pano_nn (composed via two static transforms).
    static = {
        ("base_footprint", "roof_rack"): _xyz(0.0, 0.0, 1.5),
        ("roof_rack", "camera_pano_nn"): _xyz(0.0, 0.0, 0.3),
    }
    calib = build_calibration_dict(
        camera_infos={"CAM_PANO_NN": _camera_info("camera_pano_nn")},
        lidar_frame_ids={},
        static_transforms=static,
        ego_frame="base_footprint",
        bag_id="b",
    )
    # 1.5 + 0.3 = 1.8 m above base.
    assert calib["cameras"]["CAM_PANO_NN"]["ego_T_cam"][2][3] == 1.8


def test_warns_when_extrinsic_unresolved():
    calib = build_calibration_dict(
        camera_infos={"CAM_LOWER_NE": _camera_info("camera_lower_ne")},
        lidar_frame_ids={"LIDAR_CC": "lidar_cc"},
        static_transforms={},  # no /tf_static at all
        ego_frame="base_footprint",
        bag_id="b",
    )
    assert calib["cameras"]["CAM_LOWER_NE"]["ego_T_cam"] is None
    assert calib["lidars"]["LIDAR_CC"]["ego_T_lidar"] is None
    assert calib["checks"]["sanity"] == "warn"
    assert "no /tf_static path" in calib["checks"]["notes"]


def test_static_transforms_dump_preserves_all_links():
    static = {
        ("base_footprint", "camera_lower_ne"): _xyz(2.0),
        ("base_footprint", "lidar_cc"): _xyz(0.5),
        ("base_footprint", "imu"): _xyz(0.1, 0.0, 0.5),
    }
    calib = build_calibration_dict(
        camera_infos={"CAM_LOWER_NE": _camera_info("camera_lower_ne")},
        lidar_frame_ids={"LIDAR_CC": "lidar_cc"},
        static_transforms=static,
        ego_frame="base_footprint",
        bag_id="b",
    )
    # Even imu is preserved in static_transforms even though no camera/lidar
    # references it — debugging breadcrumb.
    assert "base_footprint__imu" in calib["static_transforms"]
    assert "base_footprint__camera_lower_ne" in calib["static_transforms"]
    assert "base_footprint__lidar_cc" in calib["static_transforms"]


def test_resolve_chain_returns_none_beyond_max_hops():
    # 5 hops; default max_hops is 4.
    static = {
        ("a", "b"): _identity(),
        ("b", "c"): _identity(),
        ("c", "d"): _identity(),
        ("d", "e"): _identity(),
        ("e", "f"): _identity(),
    }
    assert _resolve_chain(static, "a", "f", max_hops=4) is None
    assert _resolve_chain(static, "a", "f", max_hops=5) is not None


def test_intrinsics_round_trip_K_matrix_shape():
    calib = build_calibration_dict(
        camera_infos={"CAM": _camera_info("cam_frame", fx=1500)},
        lidar_frame_ids={},
        static_transforms={("base_footprint", "cam_frame"): _identity()},
        ego_frame="base_footprint",
        bag_id="b",
    )
    K = calib["cameras"]["CAM"]["K"]
    assert len(K) == 3 and len(K[0]) == 3
    assert K[0][0] == 1500.0  # fx
    assert K[1][1] == 1500.0  # fy
    assert K[2][2] == 1.0


# ---------------------------------------------------------------------------
# require_complete: every configured sensor must be calibrated
# ---------------------------------------------------------------------------
def _one_of_each_cfg():
    cfg = load_config(str(CONFIG_DIR / "ingest.yaml"))
    topics = cfg.topics.model_copy(
        update={
            "cameras": {"CAM": cfg.topics.cameras["CAM_FRONT"]},
            "lidars": {"LIDAR": "/LIDAR_TOP"},
        }
    )
    return cfg.model_copy(update={"topics": topics})


def _calib(static):
    return build_calibration_dict(
        camera_infos={"CAM": _camera_info("cam")},
        lidar_frame_ids={"LIDAR": "lidar"},
        static_transforms=static,
        ego_frame="base_link",
        bag_id="b",
    )


def test_complete_calibration_passes():
    static = {("base_link", "cam"): _xyz(1.0), ("base_link", "lidar"): _xyz(0.5)}
    require_complete(_calib(static), _one_of_each_cfg())


def test_unresolved_extrinsic_fails_ingest():
    calib = _calib({("base_link", "cam"): _xyz(1.0)})
    with pytest.raises(CalibrationError, match="lidar LIDAR: no transform"):
        require_complete(calib, _one_of_each_cfg())


def test_sensor_missing_from_calibration_fails_ingest():
    calib = _calib({("base_link", "cam"): _xyz(1.0), ("base_link", "lidar"): _xyz(0)})
    del calib["cameras"]["CAM"]
    with pytest.raises(CalibrationError, match="camera CAM: no entry"):
        require_complete(calib, _one_of_each_cfg())


def test_freeze_from_bag_needs_info_and_tf_topics(monkeypatch):
    cfg = _one_of_each_cfg()
    topics = cfg.topics.model_copy(update={"tf_static": None})
    cfg = cfg.model_copy(update={"topics": topics})

    def no_bag(*_a, **_kw):
        raise AssertionError("must fail before reading the bag")

    monkeypatch.setattr(calibration, "messages", no_bag)
    with pytest.raises(CalibrationError, match="topics.tf_static"):
        calibration.freeze_from_bag("bag", "b", cfg)
