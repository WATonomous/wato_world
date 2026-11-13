"""Coordinate, evidence and real-backend controls; no downstream labels."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import os
import shutil

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

from wato_common.artifacts.calibration import CameraCalibration
from wato_common.artifacts.frame_index import CameraFrame
from wato_common.artifacts.lidar_sweeps import WorldSweep, write_world
from wato_camera_calibration.config import (
    SamplingConfig,
    SolverConfig,
    ValidationConfig,
)
from wato_camera_calibration.inputs import Sample, build_sample, world_to_ego
from wato_camera_calibration.sampling import scene_split
from wato_camera_calibration.solver import (
    backend_identity,
    camera_parameters,
    fit,
    prepare,
)
from wato_camera_calibration.validation import evaluate


def camera(T=None):
    return CameraCalibration(
        "CAM",
        np.array([[180.0, 0, 160], [0, 180.0, 120], [0, 0, 1]]),
        np.eye(4) if T is None else T,
        width=320,
        height=240,
    )


def synthetic_pairs(count=9, points=4000):
    """Analytic texture observed at independent pixel/depth samples, with a
    planted camera mount. Each image has a different texture phase.
    """
    rng = np.random.default_rng(12)
    truth = np.eye(4)
    truth[:3, :3] = Rotation.from_rotvec([0.02, -0.03, 0.01]).as_matrix()
    truth[:3, 3] = [0.23, -0.12, 1.6]
    samples = []
    cam = camera(truth)
    yy, xx = np.mgrid[:240, :320]
    for i in range(count):

        def texture(x, y):
            return np.clip(
                0.5
                + 0.15 * np.sin(x * 0.083 + i)
                + 0.13 * np.cos(y * 0.107 - i * 0.7)
                + 0.1 * np.sin(x * 0.039 + y * 0.061 + i * 0.3),
                0,
                1,
            )

        gray = np.rint(texture(xx, yy) * 255).astype(np.uint8)
        uv = rng.uniform([15, 15], [305, 225], (points, 2))
        depth = rng.uniform(5, 25, points)
        optical = np.column_stack(
            ((uv[:, 0] - 160) / 180 * depth, (uv[:, 1] - 120) / 180 * depth, depth)
        )
        ego = optical @ truth[:3, :3].T + truth[:3, 3]
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec([0, 0, i * 0.08]).as_matrix()
        pose[:3, 3] = [i * 3, i * 0.5, 0]
        split = ("train", "validation", "test")[i % 3]
        samples.append(
            Sample(
                i * 10**9, i, split, "", pose, ego, texture(uv[:, 0], uv[:, 1]), gray
            )
        )
    return cam, samples


def test_moving_pose_adapter_uses_image_time():
    cam, samples = synthetic_pairs(3, 500)
    for s in samples:
        world = s.xyz @ s.world_T_ego[:3, :3].T + s.world_T_ego[:3, 3]
        np.testing.assert_allclose(
            world_to_ego(world, s.world_T_ego), s.xyz, atol=1e-12
        )
        wrong_pose = s.world_T_ego.copy()
        wrong_pose[:3, 3] += [1, 0, 0]
        assert not np.allclose(world_to_ego(world, wrong_pose), s.xyz)


def test_guarded_splits_do_not_share_accumulation_windows():
    cfg = SamplingConfig(accumulation_s=0.4, guard_s=0.5, block_s=4)
    assert scene_split(0, 0, cfg) is None
    assert scene_split(3_700_000_000, 0, cfg) is None
    assert scene_split(12_600_000_000, 0, cfg) == (3, "validation")
    assert scene_split(16_600_000_000, 0, cfg) == (4, "test")
    with pytest.raises(ValueError, match="guard_s"):
        SamplingConfig(accumulation_s=1, guard_s=0.5)


def test_backend_model_is_stored_rectified_geometry():
    cam = replace(
        camera(),
        rectified=True,
        distortion_model="equidistant",
        camera_info_D=(1, 2, 3, 4),
    )
    assert camera_parameters(cam) == {
        "model": "plumb_bob",
        "intrinsics": [180.0, 180.0, 160.0, 120.0],
        "distortion": [0.0] * 5,
    }
    K = cam.K.copy()
    K[0, 1] = 2
    with pytest.raises(ValueError, match="skew"):
        camera_parameters(replace(cam, K=K))


def test_build_sample_preserves_static_intensity_and_requires_intensity(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", tmp_path.as_uri())
    cam, samples = synthetic_pairs(1, 4000)
    sample = samples[0]
    img = tmp_path / "image.png"
    Image.fromarray(sample.gray).save(img)
    world = sample.xyz @ sample.world_T_ego[:3, :3].T + sample.world_T_ego[:3, 3]
    (tmp_path / "raw/B/chunks/0/lidar_proc").mkdir(parents=True)
    world_uri = write_world(
        "B",
        "0",
        0,
        WorldSweep(*world.T, intensity=(sample.intensity * 255).astype("f4")),
    )
    mask = np.zeros(len(world), bool)
    mask[:1000] = True
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, mask)
    row = SimpleNamespace(world_path=world_uri, dynamic_mask_path=str(mask_path))
    frame = CameraFrame("CAM", 0, 0, str(img), (0,), sample.world_T_ego)
    result, used = build_sample(
        frame, [row], cam, SamplingConfig(voxel_m=0.001), 0, "train"
    )
    assert len(result.xyz) == 3000
    assert result.intensity.min() == 0 and result.intensity.max() == 1
    assert img in used
    write_world("B", "0", 0, WorldSweep(*world.T))
    with pytest.raises(ValueError, match="intensity is missing"):
        build_sample(frame, [row], cam, SamplingConfig(), 0, "train")


def annotations(cam, samples):
    rows = []
    for sample in samples:
        if sample.split == "train":
            continue
        pts = sample.xyz[::100]
        uv, _, _ = cam.model.project(pts, np.linalg.inv(cam.ego_T_cam))
        world = pts @ sample.world_T_ego[:3, :3].T + sample.world_T_ego[:3, 3]
        rows.append(
            {
                "timestamp_ns": sample.timestamp_ns,
                "world_points": world.tolist(),
                "pixels": uv.tolist(),
                "provenance": "independent_static",
            }
        )
    return {"CAM": rows}


def test_validation_requires_independent_evidence_and_preserves_correct_baseline():
    cam, samples = synthetic_pairs()
    cfg = ValidationConfig(
        qualified=True, profile_id="synthetic-control", correspondences="fixture"
    )
    evidence = annotations(cam, samples)
    result = evaluate(
        samples, cam, cam.ego_T_cam, [cam.ego_T_cam] * 4, cfg, SolverConfig(), evidence
    )
    assert result["status"] == "recorded_valid"
    result = evaluate(
        samples, cam, cam.ego_T_cam, [cam.ego_T_cam] * 4, cfg, SolverConfig(), {}
    )
    assert result["status"] == "insufficient_evidence"
    result = evaluate(
        samples,
        cam,
        cam.ego_T_cam,
        [cam.ego_T_cam] * 4,
        ValidationConfig(),
        SolverConfig(),
        evidence,
    )
    assert result["status"] == "insufficient_evidence"


def test_stable_true_correction_passes_but_unstable_translation_does_not():
    truth, samples = synthetic_pairs()
    recorded = truth.ego_T_cam.copy()
    recorded[:3, :3] = (
        recorded[:3, :3] @ Rotation.from_rotvec([0.02, -0.02, 0]).as_matrix()
    )
    cam = replace(truth, ego_T_cam=recorded)
    cfg = ValidationConfig(
        qualified=True, profile_id="synthetic-control", correspondences="fixture"
    )
    evidence = annotations(truth, samples)
    result = evaluate(
        samples,
        cam,
        truth.ego_T_cam,
        [truth.ego_T_cam] * 4,
        cfg,
        SolverConfig(),
        evidence,
    )
    assert result["status"] == "refined", result
    bad = truth.ego_T_cam.copy()
    bad[0, 3] += 0.2
    result = evaluate(
        samples,
        cam,
        truth.ego_T_cam,
        [truth.ego_T_cam, bad],
        cfg,
        SolverConfig(),
        evidence,
    )
    assert result["status"] == "insufficient_evidence"


def test_native_backend_recovers_full_rigid_transform(tmp_path):
    if (
        shutil.which(os.environ.get("WATO_KOIDE_EXECUTABLE", "wato_koide_calibrate"))
        is None
    ):
        pytest.skip("native backend is tested in the camera_calibration image")
    identity = backend_identity(SolverConfig())
    truth, samples = synthetic_pairs(4, 4000)
    recorded = truth.ego_T_cam.copy()
    recorded[:3, :3] = (
        recorded[:3, :3] @ Rotation.from_rotvec([0.012, -0.015, 0.008]).as_matrix()
    )
    recorded[:3, 3] += [0.04, -0.03, 0.02]
    cam = replace(truth, ego_T_cam=recorded)
    pairs = prepare(samples, tmp_path / "pairs")
    result, info = fit(
        cam,
        pairs,
        recorded,
        SolverConfig(max_iterations=120, timeout_s=90),
        identity,
        tmp_path / "fit",
    )
    angle = np.rad2deg(
        Rotation.from_matrix(truth.ego_T_cam[:3, :3].T @ result[:3, :3]).magnitude()
    )
    error = np.linalg.norm(result[:3, 3] - truth.ego_T_cam[:3, 3])
    assert angle < 0.2, (angle, error, info)
    assert error < 0.04, (angle, error, info)


def test_prepare_publication_resume_and_stale_input(tmp_path, monkeypatch):
    from wato_common.artifacts import calibration as recorded
    from wato_common.artifacts.calibration import Calibration, LidarCalibration
    from wato_common.artifacts.calibration_refinement import read, resolve
    from wato_camera_calibration import pipeline
    from wato_camera_calibration.config import ComponentConfig

    monkeypatch.setenv("ARTIFACT_ROOT_URI", tmp_path.as_uri())
    cam, samples = synthetic_pairs()
    recorded.write(
        "B", Calibration({"CAM": cam}, {"top": LidarCalibration("top", np.eye(4))})
    )
    observation = tmp_path / "observation.txt"
    observation.write_text("first")
    calls = []

    def observations(*args):
        calls.append(True)
        return samples, {observation}, {}

    monkeypatch.setattr(pipeline, "collect", observations)
    cfg = ComponentConfig(
        enabled=True, mode="prepare", sampling=SamplingConfig(lidar_id="top")
    )
    report = pipeline.run(cfg, bag_id="B")
    assert report["cameras"]["CAM"]["status"] == "insufficient_evidence"
    read("B")
    with pytest.raises(ValueError, match="not qualified"):
        resolve("B", "validated")
    assert pipeline.run(cfg, bag_id="B") == report
    assert len(calls) == 1
    observation.write_text("changed")
    with pytest.raises(ValueError, match="stale"):
        pipeline.run(cfg, bag_id="B")
    pipeline.run(cfg, bag_id="B", force=True)
    assert len(calls) == 2


def test_backend_timeout_publishes_error_without_changing_baseline(
    tmp_path, monkeypatch
):
    import subprocess
    from wato_common.artifacts import calibration as recorded
    from wato_common.artifacts.calibration import Calibration, LidarCalibration
    from wato_common.artifacts.calibration_refinement import read
    from wato_camera_calibration import pipeline
    from wato_camera_calibration.config import ComponentConfig

    monkeypatch.setenv("ARTIFACT_ROOT_URI", tmp_path.as_uri())
    cam, samples = synthetic_pairs()
    recorded.write(
        "B", Calibration({"CAM": cam}, {"top": LidarCalibration("top", np.eye(4))})
    )
    monkeypatch.setattr(pipeline, "collect", lambda *args: (samples, set(), {}))
    monkeypatch.setattr(pipeline, "backend_identity", lambda *args: {"fixture": True})

    def timeout(*args):
        raise subprocess.TimeoutExpired("backend", 1)

    monkeypatch.setattr(pipeline, "run_fits", timeout)
    cfg = ComponentConfig(enabled=True, sampling=SamplingConfig(lidar_id="top"))
    report = pipeline.run(cfg, bag_id="B")
    assert report["cameras"]["CAM"]["status"] == "error"
    effective, _, _ = read("B")
    np.testing.assert_equal(effective.camera("CAM").ego_T_cam, cam.ego_T_cam)


def test_native_rotation_only_fixes_camera_centre_during_optimization(tmp_path):
    if (
        shutil.which(os.environ.get("WATO_KOIDE_EXECUTABLE", "wato_koide_calibrate"))
        is None
    ):
        pytest.skip("native backend is tested in the camera_calibration image")
    identity = backend_identity(SolverConfig())
    truth, samples = synthetic_pairs(4, 4000)
    recorded = truth.ego_T_cam.copy()
    recorded[:3, :3] = (
        recorded[:3, :3] @ Rotation.from_rotvec([0.015, -0.02, 0.01]).as_matrix()
    )
    cam = replace(truth, ego_T_cam=recorded)
    pairs = prepare(samples, tmp_path / "pairs")
    result, info = fit(
        cam,
        pairs,
        recorded,
        SolverConfig(motion_model="rotation_only", max_iterations=120),
        identity,
        tmp_path / "fit",
    )
    np.testing.assert_allclose(result[:3, 3], recorded[:3, 3], atol=1e-14, rtol=0)
    angle = np.rad2deg(
        Rotation.from_matrix(truth.ego_T_cam[:3, :3].T @ result[:3, :3]).magnitude()
    )
    assert angle < 0.2, (angle, info)


def test_picker_exports_held_out_views_in_world_coordinates(tmp_path, monkeypatch):
    import json

    from wato_common.artifacts import calibration as recorded
    from wato_common.artifacts.calibration import Calibration, LidarCalibration
    from wato_camera_calibration import picker, pipeline
    from wato_camera_calibration.config import ComponentConfig
    from wato_camera_calibration.validation import correspondence_residuals

    monkeypatch.setenv("ARTIFACT_ROOT_URI", tmp_path.as_uri())
    cam, samples = synthetic_pairs(6, 300)
    recorded.write(
        "B", Calibration({"CAM": cam}, {"top": LidarCalibration("top", np.eye(4))})
    )
    monkeypatch.setattr(pipeline, "collect", lambda *args: (samples, set(), {}))
    pipeline.run(
        ComponentConfig(
            enabled=True, mode="prepare", sampling=SamplingConfig(lidar_id="top")
        ),
        bag_id="B",
    )
    (tmp_path / "raw/B/chunks/0/lidar_proc").mkdir(parents=True)
    frames, rows, worlds = [], [], {}
    for s in samples:
        image = tmp_path / f"{s.timestamp_ns}.png"
        Image.fromarray(s.gray).save(image)
        frames.append(
            CameraFrame("CAM", s.block, s.timestamp_ns, str(image), (), s.world_T_ego)
        )
        world = s.xyz @ s.world_T_ego[:3, :3].T + s.world_T_ego[:3, 3]
        worlds[s.timestamp_ns] = world
        mask = tmp_path / f"{s.timestamp_ns}.npy"
        np.save(mask, np.zeros(len(world), bool))
        uri = write_world(
            "B",
            "0",
            s.block,
            WorldSweep(*world.T, intensity=(s.intensity * 255).astype("f4")),
        )
        rows.append(
            SimpleNamespace(
                world_path=uri,
                dynamic_mask_path=str(mask),
                reference_timestamp_ns=s.timestamp_ns,
            )
        )
    monkeypatch.setattr(picker, "observations", lambda *args: (frames, rows, set()))

    root = picker.export("B", window_s=0.4, voxel_m=1e-4)

    index = json.loads((root / "views.json").read_text())
    views = index["cameras"]["CAM"]["views"]
    assert {v["split"] for v in views} == {"validation", "test"}
    assert len(views) == sum(s.split != "train" for s in samples)
    assert (root / "index.html").exists()
    for view in views:
        n = view["n_points"]
        data = (root / view["cloud"]).read_bytes()
        assert len(data) == n * 13
        xyz = np.frombuffer(data[: n * 12], "<f4").reshape(n, 3) + view["origin"]
        expected = worlds[int(view["timestamp_ns"])]
        assert n == len(expected)
        np.testing.assert_allclose(
            np.sort(xyz, axis=0), np.sort(expected, axis=0), atol=1e-4
        )
        assert (root / view["image"]).exists()

    # A pick written the way the page writes it scores zero against the truth.
    by_time = {s.timestamp_ns: s for s in samples}
    picks = []
    for view in views:
        s = by_time[int(view["timestamp_ns"])]
        world = worlds[s.timestamp_ns][:5]
        uv, _, _ = cam.model.project(s.xyz[:5], np.linalg.inv(cam.ego_T_cam))
        picks.append(
            {
                "timestamp_ns": int(view["timestamp_ns"]),
                "provenance": "independent_static",
                "world_points": world.tolist(),
                "pixels": uv.tolist(),
            }
        )
    residuals = correspondence_residuals(samples, cam, cam.ego_T_cam, {"CAM": picks})
    for rows_ in residuals.values():
        for _, errors in rows_:
            assert np.all(errors < 1e-6)
