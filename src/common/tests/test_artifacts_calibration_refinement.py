"""Complete evidence, status and source identity are required for refinement."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from wato_common.artifacts import calibration as recorded
from wato_common.artifacts import calibration_refinement as refinement
from wato_common.artifacts.calibration import (
    Calibration,
    CameraCalibration,
    LidarCalibration,
)


def publish(tmp_path, monkeypatch, status="refined", qualified=True):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", tmp_path.as_uri())
    original = Calibration(
        {"CAM": CameraCalibration("CAM", np.eye(3), np.eye(4))},
        {"LIDAR": LidarCalibration("LIDAR", np.eye(4))},
    )
    recorded.write("B", original)
    new = np.eye(4)
    new[0, 3] = 0.1
    candidate = replace(
        original, cameras={"CAM": replace(original.camera("CAM"), ego_T_cam=new)}
    )
    source = tmp_path / "source.txt"
    source.write_text("static observations")
    report = {
        "schema_version": 1,
        "bag_id": "B",
        "validation_qualified": qualified,
        "cameras": {"CAM": {"status": status}},
    }
    refinement.write(
        "B",
        candidate,
        candidate if status == "refined" else original,
        report,
        [refinement.fingerprint(source)],
        {},
    )
    return original, candidate, source


def test_recorded_source_remains_unchanged(tmp_path, monkeypatch):
    original, candidate, _ = publish(tmp_path, monkeypatch)
    np.testing.assert_equal(
        refinement.resolve("B").calibration.camera("CAM").ego_T_cam,
        original.camera("CAM").ego_T_cam,
    )
    result = refinement.resolve("B", "validated", ["CAM"])
    np.testing.assert_equal(
        result.calibration.camera("CAM").ego_T_cam, candidate.camera("CAM").ego_T_cam
    )
    assert result.report_digest


@pytest.mark.parametrize(
    "status", ["insufficient_evidence", "rejected", "error", "not_requested"]
)
def test_failed_or_unevaluated_cameras_do_not_fall_back(tmp_path, monkeypatch, status):
    publish(tmp_path, monkeypatch, status=status)
    with pytest.raises(ValueError, match="unavailable"):
        refinement.resolve("B", "validated", ["CAM"])


def test_inputs_outputs_and_completion_are_verified(tmp_path, monkeypatch):
    _, _, source = publish(tmp_path, monkeypatch)
    source.write_text("changed")
    with pytest.raises(ValueError, match="stale"):
        refinement.resolve("B", "validated")
    publish(tmp_path, monkeypatch)
    root = refinement.directory("B")
    (root / "report.json").write_text("{}")
    with pytest.raises(ValueError, match="incomplete"):
        refinement.resolve("B", "validated")
    (root / "manifest.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        refinement.resolve("B", "validated")


def test_unqualified_profile_and_changed_baseline_are_rejected(tmp_path, monkeypatch):
    original, _, _ = publish(tmp_path, monkeypatch, qualified=False)
    with pytest.raises(ValueError, match="not qualified"):
        refinement.resolve("B", "validated")
    recorded.write("B", replace(original, version="new"))
    with pytest.raises(ValueError, match="recorded calibration changed"):
        refinement.resolve("B", "validated")


def test_existing_output_cannot_mix_calibrations(tmp_path, monkeypatch):
    publish(tmp_path, monkeypatch)
    selection = refinement.resolve("B", "validated")
    output = tmp_path / "manifest.json"
    output.write_text(
        json.dumps({"camera_calibration": refinement.resolve("B").identity()})
    )
    with pytest.raises(ValueError, match="--force"):
        refinement.check_output_identity(output, selection)
    output.write_text(json.dumps({"camera_calibration": selection.identity()}))
    refinement.check_output_identity(output, selection)


def test_resumable_outputs_and_transitive_manifests_keep_identity(
    tmp_path, monkeypatch
):
    from wato_common import manifest

    publish(tmp_path, monkeypatch)
    selected = refinement.resolve("B", "validated")
    with pytest.raises(ValueError, match="legacy"):
        refinement.bind_outputs(
            "B", "0", "perception_2d", selected, force=False, existing_outputs=True
        )
    refinement.bind_outputs(
        "B", "0", "perception_2d", selected, force=True, existing_outputs=True
    )
    upstream = manifest.write(
        component="perception_2d",
        bag_id="B",
        chunk_id="0",
        filename="manifest_perception_2d.json",
    )
    data = json.loads(Path(upstream.removeprefix("file://")).read_text())
    assert data["camera_calibration"] == selected.identity()
    assert data["inputs"]["calibration"]["sha256"] == selected.digest
    refinement.bind_inherited_outputs(
        "B", None, "tracking", [upstream], force=False, existing_outputs=False
    )
    refinement.bind_from_bag_stage(
        "B", "label_refinement", "tracking", force=False, existing_outputs=False
    )
    assert (
        refinement.output_binding("B", "0", "label_refinement")["identity"]
        == selected.identity()
    )
    with pytest.raises(ValueError, match="different"):
        refinement.bind_outputs(
            "B",
            "0",
            "perception_2d",
            refinement.resolve("B"),
            force=False,
            existing_outputs=True,
        )


def test_failed_forced_switch_cannot_bless_old_cached_outputs(tmp_path, monkeypatch):
    publish(tmp_path, monkeypatch)
    old = refinement.resolve("B")
    selected = refinement.resolve("B", "validated")
    refinement.bind_outputs(
        "B", "0", "perception_2d", old, force=False, existing_outputs=False
    )
    refinement.bind_outputs(
        "B", "0", "perception_2d", selected, force=True, existing_outputs=True
    )
    with pytest.raises(ValueError, match="replacement did not complete"):
        refinement.bind_outputs(
            "B", "0", "perception_2d", selected, force=False, existing_outputs=True
        )
    refinement.finish_output_binding("B", "0", "perception_2d")
    refinement.bind_outputs(
        "B", "0", "perception_2d", selected, force=False, existing_outputs=True
    )
