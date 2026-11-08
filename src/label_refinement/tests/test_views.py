"""Camera/artifact adapter checks, independent of the SLF optimizer."""

from dataclasses import asdict

import numpy as np
import pytest

from wato_common.artifacts.calibration import Calibration, CameraCalibration
from wato_common.artifacts.frame_index import CameraFrame, FrameIndex
from wato_common.artifacts.masklets import MaskletIndex, MaskletRecord, write_mask
from wato_common.artifacts.tracks import TrackFrame
from wato_common.geometry import invert_se3
from wato_common.schemas import Box3D
from wato_label_refinement import views
from wato_label_refinement.config import SlfConfig


def test_slf_forward_projection_uses_the_full_intrinsic_matrix():
    K = np.array([[720.0, 65.0, 320.0], [0.0, 810.0, 240.0], [0.0, 0.0, 1.0]])
    transform = np.eye(4)
    transform[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    transform[:3, 3] = [0.5, -0.2, 1.0]
    points = np.array([[1.0, 0.5, 5.0], [-2.0, 1.5, 10.0]])
    camera = points @ transform[:3, :3].T + transform[:3, 3]
    homogeneous = camera @ K.T
    expected = homogeneous[:, :2] / homogeneous[:, 2:]
    actual, depth = views.project(points, K, transform)
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    np.testing.assert_allclose(depth, camera[:, 2])


@pytest.fixture
def camera_problem(tmp_path, monkeypatch, request):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", tmp_path.as_uri())
    K = np.array([[500.0, 90.0, 320.0], [0.0, 570.0, 240.0], [0.0, 0.0, 1.0]])
    ego_T_cam = np.eye(4)
    ego_T_cam[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    ego_T_cam[:3, 3] = [0.2, -0.3, 1.8]
    world_T_ego = np.eye(4)
    c, s = np.cos(0.31), np.sin(0.31)
    world_T_ego[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    world_T_ego[:3, 3] = [4, 2, 0.5]
    world_T_cam = world_T_ego @ ego_T_cam
    centre = (world_T_cam @ [0, 0, 12, 1])[:3]
    box = Box3D(
        cx=centre[0], cy=centre[1], cz=centre[2], l=4, w=1.8, h=1.6, heading=0.31
    )
    frame = TrackFrame(
        track_id="t",
        chunk_id="0000",
        sweep_id=0,
        reference_timestamp_ns=1_000_000_000,
        proposal_id="p",
        cls="car",
        score=1,
        owned=True,
        **asdict(box),
    )
    lens, coefficients = getattr(request, "param", ("", ()))
    camera = CameraCalibration(
        "CAM",
        K,
        ego_T_cam,
        width=640,
        height=480,
        distortion_model=lens,
        distortion=coefficients,
    )
    image = CameraFrame("CAM", 0, 1_000_000_000, "unused.jpg", (0,), world_T_ego)
    record = MaskletRecord("m", "CAM", "car", 1, (0,))
    mask = np.zeros((480, 640), bool)
    mask[190:290, 260:380] = np.random.default_rng(4).random((100, 120)) > 0.1
    write_mask("bag", "0000", "m", 0, mask)
    masks = MaskletIndex("bag", "0000", [record])
    chunk = views.Chunk(FrameIndex([image]), masks, {"m": record}, {})
    view = views.View(0, image, (record,), world_T_cam, float(mask.sum()))
    cfg = SlfConfig(crop_px=12)
    problem = views.problem(
        "t",
        "car",
        [frame],
        [box],
        np.zeros((1, 2)),
        np.zeros((1, 2)),
        False,
        [view],
        {},
        {"0000": chunk},
        Calibration({"CAM": camera}, {}),
        cfg,
    )
    return problem, camera, world_T_cam, cfg, mask


@pytest.mark.parametrize(
    "camera_problem",
    [
        ("", ()),
        ("plumb_bob", (-0.12, 0.025, 0.003, -0.004, 0.005)),
        (
            "rational_polynomial",
            (-0.12, 0.025, 0.003, -0.004, 0.005, 0.01, -0.003, 0.001),
        ),
        ("equidistant", (-0.03, 0.004, -0.0005, 0.0001)),
    ],
    indirect=True,
)
def test_actual_slf_crop_rays_reproject_to_a_rectangular_pixel_grid(camera_problem):
    problem, camera, world_T_cam, cfg, _ = camera_problem
    # The shared forward model has independent OpenCV-observation tests.
    # Here it checks the artifact → SLF-ray adapter for every lens model.
    pixels, _, valid = camera.model.project(
        problem.origin[0] + 10 * problem.dirs[0], invert_se3(world_T_cam)
    )
    assert valid.all()
    pixels = pixels.reshape(cfg.crop_px, cfg.crop_px, 2)
    np.testing.assert_allclose(
        pixels[:, :, 0], np.broadcast_to(pixels[0, :, 0], pixels.shape[:2]), atol=1e-9
    )
    np.testing.assert_allclose(
        pixels[:, :, 1],
        np.broadcast_to(pixels[:, 0, 1, None], pixels.shape[:2]),
        atol=1e-9,
    )
    np.testing.assert_allclose(np.linalg.norm(problem.dirs[0], axis=1), 1)


def test_mask_targets_sample_the_pixel_nearest_the_actual_ray(camera_problem):
    problem, camera, world_T_cam, _, mask = camera_problem
    uv, _, _ = camera.model.project(
        problem.origin[0] + 10 * problem.dirs[0], invert_se3(world_T_cam)
    )
    ui = np.clip(np.rint(uv[:, 0]).astype(int), 0, mask.shape[1] - 1)
    vi = np.clip(np.rint(uv[:, 1]).astype(int), 0, mask.shape[0] - 1)
    np.testing.assert_array_equal(problem.target[0] > 0.5, mask[vi, ui])
