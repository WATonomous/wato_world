"""wato_common.artifacts.calibration: what ingest writes, consumers read back."""

from __future__ import annotations

import json

import numpy as np
import pytest

from wato_common.artifact_store import calibration_path, local_path
from wato_common.artifacts import ArtifactFormatError, calibration
from wato_common.artifacts.calibration import (
    Calibration,
    CameraCalibration,
    LidarCalibration,
)

BAG = "b"


@pytest.fixture(autouse=True)
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")


def _se3(x: float) -> np.ndarray:
    T = np.eye(4)
    T[0, 3] = x
    return T


def _calib(**overrides) -> Calibration:
    fields = dict(
        cameras={
            "CAM": CameraCalibration(
                cam_id="CAM",
                K=np.array([[1000.0, 0, 960], [0, 1000, 540], [0, 0, 1]]),
                ego_T_cam=_se3(2.0),
                frame_id="cam",
                distortion=(0.1, -0.2, 0.0, 0.0, 0.01),
                distortion_model="plumb_bob",
                width=1920,
                height=1080,
            )
        },
        lidars={"LIDAR": LidarCalibration("LIDAR", _se3(0.5), frame_id="lidar")},
        version="auto_from_b",
        ego_frame="base_link",
        static_transforms={"base_link__cam": _se3(2.0)},
        checks={"sanity": "auto", "notes": ""},
    )
    fields.update(overrides)
    return Calibration(**fields)


def _assert_same(a: Calibration, b: Calibration) -> None:
    assert a.to_dict() == b.to_dict()


def test_round_trip():
    original = _calib()
    calibration.write(BAG, original)
    loaded = Calibration.load(BAG)
    _assert_same(loaded, original)
    cam = loaded.camera("CAM")
    assert cam.distortion == (0.1, -0.2, 0.0, 0.0, 0.01)
    assert (cam.width, cam.height) == (1920, 1080)
    np.testing.assert_array_equal(loaded.ego_T_lidar("LIDAR"), _se3(0.5))


def test_unresolved_extrinsics_round_trip_as_none_and_raise_on_access():
    cam = CameraCalibration("CAM", np.eye(3), None)
    calibration.write(
        BAG,
        _calib(cameras={"CAM": cam}, lidars={"LIDAR": LidarCalibration("LIDAR", None)}),
    )
    loaded = Calibration.load(BAG)
    assert loaded.cameras["CAM"].ego_T_cam is None
    assert loaded.lidars["LIDAR"].ego_T_lidar is None
    with pytest.raises(ValueError, match="null ego_T_cam"):
        loaded.camera("CAM")
    with pytest.raises(ValueError, match="null ego_T_lidar"):
        loaded.ego_T_lidar("LIDAR")


def test_unknown_sensors_raise_key_error():
    calibration.write(BAG, _calib())
    loaded = Calibration.load(BAG)
    with pytest.raises(KeyError, match="CAM_X"):
        loaded.camera("CAM_X")
    with pytest.raises(KeyError, match="LIDAR_X"):
        loaded.ego_T_lidar("LIDAR_X")


def _write_raw(d: dict) -> None:
    path = local_path(calibration_path(BAG))
    calibration.write(BAG, _calib())  # creates the bag directory
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(d, fh)


@pytest.mark.parametrize(
    "mutate, reason",
    [
        # semantic_lifting once read the per-LiDAR table under this key.
        (lambda d: d.update(lidar=d.pop("lidars")), "has no 'lidars'"),
        (lambda d: d["cameras"]["CAM"].pop("K"), "camera CAM has no 'K'"),
        (lambda d: d["cameras"]["CAM"].pop("ego_T_cam"), "has no 'ego_T_cam'"),
        (lambda d: d["lidars"]["LIDAR"].pop("ego_T_lidar"), "has no 'ego_T_lidar'"),
        (lambda d: d["cameras"]["CAM"].update(K=[1, 2, 3]), "shape"),
    ],
)
def test_malformed_files_are_rejected(mutate, reason):
    d = _calib().to_dict()
    mutate(d)
    _write_raw(d)
    with pytest.raises(ArtifactFormatError, match=reason):
        Calibration.load(BAG)


def test_authored_file_may_omit_metadata():
    """Only what projection needs is required; the rest reads as empty."""
    _write_raw(
        {
            "cameras": {
                "CAM": {"K": np.eye(3).tolist(), "ego_T_cam": np.eye(4).tolist()}
            },
            "lidars": {"LIDAR": {"ego_T_lidar": np.eye(4).tolist()}},
        }
    )
    loaded = Calibration.load(BAG)
    cam = loaded.camera("CAM")
    assert (cam.frame_id, cam.distortion, cam.width) == ("", (), None)
    assert loaded.version == "" and loaded.static_transforms == {}
