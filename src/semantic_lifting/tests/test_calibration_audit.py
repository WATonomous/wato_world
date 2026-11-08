"""The calibration diagnostic must improve unseen projections without dropping misses."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation
from wato_common.artifacts.calibration import CameraCalibration
from wato_semantic_lifting.calibration_audit import (
    OFFSETS,
    Sample,
    fit_family,
    load_samples,
    parameters,
    project,
    save_samples,
    score,
)
from wato_semantic_lifting.projection import compute_cam_T_world, project_and_clip


def scene(centre, depth, correction=(0.8, -1.2, 0.5)):
    K = np.array([[300.0, 0, 200], [0, 300.0, 150], [0, 0, 1]])
    cam = CameraCalibration("cam", K, np.eye(4), width=400, height=300)
    mask = np.zeros((300, 400), bool)
    x, y = centre
    mask[y - 20 : y + 21, x - 30 : x + 31] = True
    u, v = np.meshgrid(np.linspace(x - 29, x + 29, 20), np.linspace(y - 19, y + 19, 15))
    true_points = np.column_stack(
        [
            (u.ravel() - 200) * depth / 300,
            (v.ravel() - 150) * depth / 300,
            np.full(u.size, depth),
        ]
    )
    R, *_ = parameters("rotation", correction)
    xyz = true_points @ R.T
    sample = Sample(
        "cam",
        "track",
        "0000",
        0,
        0,
        0,
        "car",
        "parked",
        True,
        xyz,
        xyz.copy(),
        np.full(len(xyz), "lidar"),
        np.repeat(np.eye(4)[None], len(OFFSETS), axis=0),
        5.0,
        mask,
        distance_transform_edt(~mask).astype(np.float32),
        np.zeros(2),
        np.array(centre),
        np.array([60.0, 40.0]),
        "image.jpg",
    )
    return cam, sample


def test_fit_recovers_alignment_on_separate_depth_and_image_position():
    cam, a = scene((120, 100), 10)
    _, b = scene((270, 200), 25)
    _, unseen = scene((220, 125), 17)
    assert score(unseen, cam)[0] < 0.85
    p = fit_family([a, b], cam, "rotation")
    assert score(unseen, cam, "rotation", p)[0] > 0.95
    assert score(unseen, cam, "rotation", p)[1] < score(unseen, cam)[1]


def test_exported_local_rotation_matches_the_lifting_projection():
    cam, sample = scene((200, 150), 12)
    cam.K[0, 1] = 7.0  # the diagnostic must preserve the full projection matrix
    world_T_ego = np.eye(4)
    world_T_ego[:3, :3] = Rotation.from_euler("xyz", [0.1, 0.2, 0.3]).as_matrix()
    world_T_ego[:3, 3] = [10.0, -5.0, 1.0]
    world_points = sample.ego_xyz @ world_T_ego[:3, :3].T + world_T_ego[:3, 3]
    R, *_ = parameters("rotation", [0.8, -1.2, 0.5])
    corrected = cam.ego_T_cam.copy()
    corrected[:3, :3] = corrected[:3, :3] @ R
    uv, ok = project(sample, cam, "rotation", [0.8, -1.2, 0.5])
    actual, _, valid = project_and_clip(
        world_points, cam.K, compute_cam_T_world(world_T_ego, corrected), (400, 300)
    )
    np.testing.assert_allclose(uv, actual, atol=1e-9)
    np.testing.assert_array_equal(ok, valid)


def test_points_outside_image_and_behind_camera_stay_in_denominator():
    cam, sample = scene((200, 150), 12, correction=(0.0, 0.0, 0.0))
    xyz = np.array([[0.0, 0.0, 12.0], [50.0, 0.0, 12.0], [0.0, 0.0, -12.0]])
    sample = replace(sample, xyz=xyz, ego_xyz=xyz)
    assert score(sample, cam)[0] == pytest.approx(1 / 3)


def test_cache_preserves_correspondences_and_rejects_changed_inputs(tmp_path):
    _, sample = scene((200, 150), 12)
    metadata = {"bag": "b", "source_sha256": {"calibration": "original"}}
    save_samples(tmp_path, [sample], metadata)
    [restored] = load_samples(tmp_path, metadata)
    for name in ("xyz", "ego_xyz", "mask", "distance", "offset_poses"):
        np.testing.assert_array_equal(getattr(restored, name), getattr(sample, name))
    assert restored.track_id == sample.track_id
    with pytest.raises(ValueError, match="inputs changed"):
        load_samples(tmp_path, {"bag": "another"})


def test_collection_keeps_restarted_frame_numbers_in_different_chunks(monkeypatch):
    """A linked track's image 0 in its next chunk is a different physical image."""
    from types import SimpleNamespace

    from wato_common.artifacts.calibration import Calibration
    from wato_common.artifacts.frame_index import CameraFrame, FrameIndex
    from wato_common.artifacts.masklets import MaskletRecord
    from wato_semantic_lifting import calibration_audit as audit

    cam, sample = scene((200, 150), 12, correction=(0.0, 0.0, 0.0))
    frames = [
        SimpleNamespace(
            track_id="linked",
            owned=True,
            chunk_id=chunk,
            sweep_id=0,
            supporting_masklet_ids=("mask",),
            cx=0.0,
            cy=0.0,
            cz=12.0,
            w=10.0,
            l=10.0,
            h=5.0,
            heading=0.0,
        )
        for chunk in ("0000", "0001")
    ]
    monkeypatch.setattr(
        audit.tracks,
        "read",
        lambda bag: (
            frames,
            [SimpleNamespace(track_id="linked", cls="car", motion_state="parked")],
        ),
    )
    monkeypatch.setattr(
        audit.Calibration, "load", lambda bag: Calibration({"cam": cam}, {})
    )
    physical_images = {
        chunk: CameraFrame(
            "cam", 0, (i + 1) * 1_000_000_000, "image.jpg", (0,), np.eye(4)
        )
        for i, chunk in enumerate(("0000", "0001"))
    }
    monkeypatch.setattr(
        audit.FrameIndex,
        "load",
        lambda bag, chunk: FrameIndex([physical_images[chunk]]),
    )
    masks = SimpleNamespace(
        records=[MaskletRecord("mask", "cam", "car", 1.0, (0,))],
        mask=lambda rec, seq: sample.mask,
    )
    monkeypatch.setattr(audit.MaskletIndex, "load", lambda bag, chunk: masks)
    sweeps = SimpleNamespace(
        frames=lambda canon: [SimpleNamespace(reference_sweep_id=0, sweep_ids=(0,))],
        get=lambda sid: SimpleNamespace(world_path="world.npz", lidar_id="lidar"),
    )
    monkeypatch.setattr(audit.ProcessedSweeps, "load", lambda bag, chunk: sweeps)
    monkeypatch.setattr(
        audit,
        "read_world",
        lambda path: SimpleNamespace(xyz=sample.xyz, n_points=len(sample.xyz)),
    )
    lookup = SimpleNamespace(
        at=lambda timestamp: SimpleNamespace(valid=True, world_T_ego=np.eye(4))
    )
    monkeypatch.setattr(audit.PoseLookup, "load", lambda bag, chunk: lookup)
    collected = audit.collect("bag", 10, None)
    assert {s.chunk_id for s in collected} == {"0000", "0001"}
    assert len({s.timestamp_ns for s in collected}) == 2
