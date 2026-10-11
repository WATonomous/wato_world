"""Tests for inputs/calibration: build_calibration (the pure-logic part
that assembles calibration.json from already-decoded inputs) and
require_complete.  Doesn't touch the rosbag or the filesystem.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from wato_common.geometry import make_se3
from wato_ingest.config import load_config
from wato_ingest.inputs import calibration
from wato_ingest.inputs.calibration import (
    CalibrationError,
    _resolve_chain,
    build_calibration,
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
    calib = build_calibration(
        camera_infos={"CAM_LOWER_NE": _camera_info("camera_lower_ne", fx=1200)},
        lidar_frame_ids={"LIDAR_CC": "lidar_cc"},
        static_transforms=static,
        ego_frame="base_footprint",
        bag_id="b",
    )

    cam = calib.cameras["CAM_LOWER_NE"]
    assert cam.frame_id == "camera_lower_ne"
    assert cam.K.tolist() == [[1200.0, 0, 960], [0, 1200.0, 540], [0, 0, 1]]
    assert cam.distortion_model == "plumb_bob"
    assert cam.ego_T_cam[0][3] == 2.0  # x translation

    lid = calib.lidars["LIDAR_CC"]
    assert lid.frame_id == "lidar_cc"
    assert lid.ego_T_lidar[0][3] == 0.5

    assert calib.checks["sanity"] == "auto"
    assert "all extrinsics resolved" in calib.checks["notes"]


def test_resolves_two_hop_extrinsic():
    # base -> roof_rack -> camera_pano_nn (composed via two static transforms).
    static = {
        ("base_footprint", "roof_rack"): _xyz(0.0, 0.0, 1.5),
        ("roof_rack", "camera_pano_nn"): _xyz(0.0, 0.0, 0.3),
    }
    calib = build_calibration(
        camera_infos={"CAM_PANO_NN": _camera_info("camera_pano_nn")},
        lidar_frame_ids={},
        static_transforms=static,
        ego_frame="base_footprint",
        bag_id="b",
    )
    # 1.5 + 0.3 = 1.8 m above base.
    assert calib.cameras["CAM_PANO_NN"].ego_T_cam[2][3] == 1.8


def test_warns_when_extrinsic_unresolved():
    calib = build_calibration(
        camera_infos={"CAM_LOWER_NE": _camera_info("camera_lower_ne")},
        lidar_frame_ids={"LIDAR_CC": "lidar_cc"},
        static_transforms={},  # no /tf_static at all
        ego_frame="base_footprint",
        bag_id="b",
    )
    assert calib.cameras["CAM_LOWER_NE"].ego_T_cam is None
    assert calib.lidars["LIDAR_CC"].ego_T_lidar is None
    assert calib.checks["sanity"] == "warn"
    assert "no /tf_static path" in calib.checks["notes"]


def test_static_transforms_dump_preserves_all_links():
    static = {
        ("base_footprint", "camera_lower_ne"): _xyz(2.0),
        ("base_footprint", "lidar_cc"): _xyz(0.5),
        ("base_footprint", "imu"): _xyz(0.1, 0.0, 0.5),
    }
    calib = build_calibration(
        camera_infos={"CAM_LOWER_NE": _camera_info("camera_lower_ne")},
        lidar_frame_ids={"LIDAR_CC": "lidar_cc"},
        static_transforms=static,
        ego_frame="base_footprint",
        bag_id="b",
    )
    # Even imu is preserved in static_transforms even though no camera/lidar
    # references it — debugging breadcrumb.
    assert "base_footprint__imu" in calib.static_transforms
    assert "base_footprint__camera_lower_ne" in calib.static_transforms
    assert "base_footprint__lidar_cc" in calib.static_transforms


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
    calib = build_calibration(
        camera_infos={"CAM": _camera_info("cam_frame", fx=1500)},
        lidar_frame_ids={},
        static_transforms={("base_footprint", "cam_frame"): _identity()},
        ego_frame="base_footprint",
        bag_id="b",
    )
    K = calib.cameras["CAM"].K
    assert K.shape == (3, 3)
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
    return build_calibration(
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
    calib = replace(calib, cameras={})
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


# ---------------------------------------------------------------------------
# freeze_from_file: an authored calibration is parsed by the shared reader
# ---------------------------------------------------------------------------
def _authored(tmp_path, monkeypatch, content: dict) -> str:
    import json

    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path / 'artifacts'}")
    src = tmp_path / "authored.json"
    src.write_text(json.dumps(content))
    return str(src)


def test_freeze_from_file_writes_canonical_calibration(tmp_path, monkeypatch):
    from wato_common.artifacts.calibration import Calibration

    src = _authored(
        tmp_path,
        monkeypatch,
        {
            "cameras": {
                "CAM": {"K": np.eye(3).tolist(), "ego_T_cam": np.eye(4).tolist()}
            },
            "lidars": {"LIDAR": {"ego_T_lidar": _xyz(0.5).tolist()}},
        },
    )
    calibration.freeze_from_file("b", src, version="hand_v2")
    calib = Calibration.load("b")
    assert calib.version == "hand_v2"
    assert calib.checks["sanity"] == "skipped"
    assert calib.ego_T_lidar("LIDAR")[0][3] == 0.5


def test_freeze_from_file_rejects_a_missing_extrinsic(tmp_path, monkeypatch):
    from wato_common.artifacts import ArtifactFormatError

    src = _authored(
        tmp_path,
        monkeypatch,
        {"cameras": {}, "lidars": {"LIDAR": {"frame_id": "lidar"}}},
    )
    with pytest.raises(ArtifactFormatError, match="lidar LIDAR has no 'ego_T_lidar'"):
        calibration.freeze_from_file("b", src)


# ---------------------------------------------------------------------------
# rectified image topics: intrinsics come from CameraInfo.P
# ---------------------------------------------------------------------------
def _rect_info(P=None, R=None):
    info = _camera_info("cam", fx=1046.0)
    info["D"] = [-0.09, 0.09, 0.0, 0.0, 0.0]
    info["P"] = (
        P if P is not None else [1022.0, 0, 645, 0, 0, 1027.0, 536, 0, 0, 0, 1, 0]
    )
    info["R"] = R if R is not None else [1, 0, 0, 0, 1, 0, 0, 0, 1]
    return info


def _build_rect(info):
    return build_calibration(
        camera_infos={"CAM": info},
        lidar_frame_ids={},
        static_transforms={("base_link", "cam"): _identity()},
        ego_frame="base_link",
        bag_id="b",
        rectified_cameras=frozenset({"CAM"}),
    )


def test_rectified_camera_uses_P_and_no_distortion():
    cam = _build_rect(_rect_info()).cameras["CAM"]
    assert cam.rectified
    assert cam.K.tolist() == [[1022.0, 0, 645], [0, 1027.0, 536], [0, 0, 1]]
    assert cam.distortion == ()
    # The raw CameraInfo is kept for traceability.
    assert cam.camera_info_K[0, 0] == 1046.0
    assert cam.camera_info_D[0] == -0.09
    assert cam.camera_info_P.shape == (3, 4)


def test_raw_camera_keeps_K_and_D():
    calib = build_calibration(
        camera_infos={"CAM": _rect_info()},
        lidar_frame_ids={},
        static_transforms={("base_link", "cam"): _identity()},
        ego_frame="base_link",
        bag_id="b",
    )
    cam = calib.cameras["CAM"]
    assert not cam.rectified
    assert cam.K[0, 0] == 1046.0 and cam.distortion[0] == -0.09


@pytest.mark.parametrize(
    "P, R, match",
    [
        ([0.0] * 12, None, "CameraInfo.P is unset"),
        (
            [1022.0, 0, 645, -60.0, 0, 1027.0, 536, 0, 0, 0, 1, 0],
            None,
            "stereo baseline",
        ),
        (None, [0, -1, 0, 1, 0, 0, 0, 0, 1], "CameraInfo.R is not identity"),
    ],
)
def test_rectification_ingest_cannot_represent_is_refused(P, R, match):
    with pytest.raises(CalibrationError, match=match):
        _build_rect(_rect_info(P=P, R=R))


def test_wato_profile_marks_its_rectified_cameras():
    cfg = load_config(str(CONFIG_DIR / "ingest.wato.yaml"))
    assert all(c.rectified for c in cfg.topics.cameras.values())
    assert all(
        c.image.endswith("image_rect_compressed") for c in cfg.topics.cameras.values()
    )
