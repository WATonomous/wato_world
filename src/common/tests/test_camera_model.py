"""Independent camera equations, inverse checks, and invalid-model boundaries."""

import numpy as np
import pytest

from wato_common.artifacts.calibration import CameraCalibration
from wato_common.geometry import CameraModel, lift_pixel_to_world, project_points


K = np.array([[700.0, 43.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]])
POINTS = np.array(
    [[0.4, -0.2, 1.0], [0.8, 0.6, 1.0], [-0.5, 0.7, 1.0], [0.0, 0.0, 1.0]]
)

# Normalized image observations independently generated with OpenCV 4.11's
# projectPoints / fisheye.projectPoints and identity K. OpenCV's Brown
# projector omits skew, so full K is applied separately in these checks.
OBSERVATIONS = [
    (
        "plumb_bob",
        (-0.12, 0.025, 0.003, -0.004, 0.005),
        [
            [0.388256, -0.193928],
            [0.72176, 0.54732],
            [-0.47051806, 0.656801284],
            [0.0, 0.0],
        ],
    ),
    (
        "rational_polynomial",
        (-0.12, 0.025, 0.003, -0.004, 0.005, 0.01, -0.003, 0.001),
        [
            [0.38751952984764765, -0.19355976492382382],
            [0.7159822222222223, 0.5429866666666668],
            [-0.4676795271708934, 0.6528273380392507],
            [0.0, 0.0],
        ],
    ),
    (
        "equidistant",
        (-0.03, 0.004, -0.0005, 0.0001),
        [
            [0.37418780967414, -0.18709390483707],
            [0.6175828478653465, 0.4631871358990097],
            [-0.4070595097789133, 0.5698833136904786],
            [0.0, 0.0],
        ],
    ),
]


@pytest.mark.parametrize("name,coefficients,observed", OBSERVATIONS)
def test_full_intrinsics_and_declared_lens_match_independent_observations(
    name, coefficients, observed
):
    model = CameraModel(K, coefficients, name)
    expected_h = np.column_stack([observed, np.ones(len(POINTS))]) @ K.T
    expected = expected_h[:, :2] / expected_h[:, 2:]
    actual, depth, valid = model.project(POINTS, np.eye(4))
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    np.testing.assert_allclose(depth, 1)
    assert valid.all()
    # Optical depth is not distance along a normalized ray.
    np.testing.assert_allclose(
        model.unproject(expected, [2, 3, 4, 5]),
        POINTS * np.array([2, 3, 4, 5])[:, None],
        atol=1e-9,
    )
    rays = model.rays(expected)
    np.testing.assert_allclose(
        rays, POINTS / np.linalg.norm(POINTS, axis=1, keepdims=True), atol=1e-10
    )


def test_fisheye_zero_polynomial_is_still_equidistant_not_pinhole():
    model = CameraModel(np.eye(3), (0.0, 0.0, 0.0, 0.0), "equidistant")
    uv, _, _ = model.project(np.array([[1.0, 0.0, 1.0]]), np.eye(4))
    np.testing.assert_allclose(uv, [[np.pi / 4, 0]])
    np.testing.assert_allclose(model.unproject(uv), [[1, 0, 1]])


def test_shared_pixel_lifting_inverts_skew_and_world_extrinsics():
    transform = np.eye(4)
    transform[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    transform[:3, 3] = [4, -2, 1]
    optical = np.array([1.2, -0.7, 5.0])
    h = K @ optical
    pixel = h[:2] / h[2]
    world = lift_pixel_to_world(*pixel, optical[2], K, transform, np.eye(4))
    np.testing.assert_allclose(world, (transform @ np.r_[optical, 1])[:3])


def test_rectified_model_uses_image_intrinsics_not_raw_camera_info():
    c = CameraCalibration(
        "cam",
        K,
        np.eye(4),
        rectified=True,
        distortion_model="equidistant",
        camera_info_K=np.eye(3),
        camera_info_D=(0.3, 0.2, 0.1, 0.04),
    )
    uv, _, _ = c.model.project(POINTS, np.eye(4))
    expected = POINTS @ K.T
    np.testing.assert_allclose(uv, expected[:, :2] / expected[:, 2:])


def test_rectified_image_cannot_apply_active_distortion_again():
    c = CameraCalibration(
        "cam",
        K,
        np.eye(4),
        rectified=True,
        distortion=(0.1, 0, 0, 0, 0),
        distortion_model="plumb_bob",
    )
    with pytest.raises(ValueError, match="rectified image has active distortion"):
        _ = c.model


def test_calibrated_image_size_is_checked_before_projection():
    camera = CameraCalibration("cam", K, np.eye(4), width=640, height=480)
    assert camera.model_for_image((640, 480)) is camera.model
    with pytest.raises(ValueError, match="update image intrinsics"):
        camera.model_for_image((320, 240))


@pytest.mark.parametrize(
    "model,d",
    [
        ("unknown_lens", ()),
        ("", (0.1,)),
        ("plumb_bob", (0.1, 0.2, 0.3)),
        ("equidistant", ()),
        ("rational_polynomial", (0,) * 5),
    ],
)
def test_unsupported_or_incomplete_lens_is_rejected(model, d):
    with pytest.raises(ValueError):
        CameraModel(K, d, model)


@pytest.mark.parametrize(
    "bad_K",
    [
        np.zeros((3, 3)),
        np.eye(4),
        [[1, 1, 0], [1, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0.1, 1]],
        [[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]],
    ],
)
def test_invalid_intrinsics_are_rejected(bad_K):
    with pytest.raises(ValueError):
        CameraModel(bad_K)


def test_invalid_points_cannot_be_marked_visible():
    pts = np.array([[np.nan, 0, 5], [0, np.inf, 5], [0, 0, -1], [0, 0, 0], [1, 2, 5]])
    uv, depth, valid = CameraModel(np.eye(3)).project(pts, np.eye(4))
    assert valid.tolist() == [False, False, False, False, True]
    assert np.isnan(uv[:4]).all()
    assert depth.tolist() == [0, 0, 0, 0, 5]
    _, common_valid = project_points(pts, np.eye(3), np.eye(4))
    np.testing.assert_array_equal(common_valid, valid)


@pytest.mark.parametrize(
    "bad", [np.diag([2.0, 1.0, 1.0, 1.0]), np.diag([-1.0, 1.0, 1.0, 1.0]), np.eye(3)]
)
def test_nonrigid_camera_transforms_are_rejected(bad):
    with pytest.raises(ValueError, match="transform"):
        CameraModel(K).project(POINTS, bad)


@pytest.mark.parametrize(
    "model",
    [
        CameraModel(np.eye(3), (-1.0, 0, 0, 0, 0), "plumb_bob"),
        CameraModel(np.eye(3), (0.0, 0, 0, 0), "equidistant"),
    ],
)
def test_noninvertible_or_outside_forward_hemisphere_pixels_do_not_silently_make_rays(
    model,
):
    with pytest.raises(ValueError):
        model.rays(np.array([[2.0, 0.0]]))


def test_empty_and_out_of_image_points_keep_the_projection_contract():
    model = CameraModel(np.eye(3))
    uv, depth, valid = model.project(np.zeros((0, 3)), np.eye(4), (10, 10))
    assert uv.shape == (0, 2) and depth.shape == (0,) and valid.shape == (0,)
    uv, depth, valid = model.project(np.array([[20.0, 0.0, 2.0]]), np.eye(4), (10, 10))
    np.testing.assert_allclose(uv, [[10, 0]])
    assert depth[0] == 2 and not valid[0]
