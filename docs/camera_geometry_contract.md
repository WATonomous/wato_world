# Camera geometry contract

`CameraCalibration.model` is the shared camera model for depth anchors,
semantic lifting, SLF view construction and proposal-camera overlays.
`wato_common/geometry/camera_model.py` owns projection and inverse rays.
Consumers pass the model rather than extracting focal lengths themselves.
The older helpers accept a bare K for an explicitly undistorted camera.

## Coordinates and equations

Optical camera axes are right, down, forward. `cam_T_world` maps world points
into that frame. Depth is optical **z**, not distance along a unit ray.
Extrinsics must be finite proper rigid transforms. Camera pose comes from
the image timestamp; object motion compensation remains a separate concern.

For a positive-depth camera point `(x, y, z)`, normalize to `(x/z, y/z)`,
apply the declared distortion D, then multiply `[D_x, D_y, 1]` by the full K
and divide by its homogeneous coordinate. This includes skew and unequal
focal lengths. Inverse rays apply `K⁻¹`, invert D with a convergence check,
then normalize `[x, y, 1]`. Backprojection at optical depth instead multiplies
that vector by z.

| Supported image model | Coefficients |
|---|---|
| Undistorted pinhole (`""`, `none`) | empty |
| `plumb_bob` | empty, 4 or 5: k1, k2, p1, p2, optional k3 |
| `rational_polynomial` | 8: k1, k2, p1, p2, k3, k4, k5, k6 |
| `equidistant` / `fisheye` | 4: k1, k2, k3, k4 in the angle polynomial |

Brown/rational and fisheye equations follow the documented
[OpenCV calibration model](https://docs.opencv.org/4.11.0/d9/d0c/group__calib3d.html)
and [fisheye model](https://docs.opencv.org/4.11.0/db/d58/group__calib3d__fisheye.html).
Tests use independent OpenCV-generated observations; OpenCV is not a runtime
dependency. Fisheye support is limited to the forward optical hemisphere.
Other models, singular intrinsics, folded lens regions and nonconvergent
inverse rays fail explicitly. Another lens requires forward and inverse
equations and independent reference tests.

## Stored image versus raw metadata

K and D describe the stored image. For raw images, ingest uses CameraInfo.K/D.
For rectified images it uses P's intrinsic block and empty distortion. Raw
CameraInfo is provenance and is never applied again. This distinction follows
the [ROS CameraInfo definition](https://github.com/ros2/common_interfaces/blob/rolling/sensor_msgs/msg/CameraInfo.msg).

`model_for_image((width, height))` checks dimensions where calibration has
them; SLF checks mask dimensions too. A resize or crop needs intrinsics for
the new image. Ingest rejects unnormalized CameraInfo binning/ROI and
unsupported rectification transforms/stereo baselines. These need an authored
`--calibration` describing the stored images.

Invalid forward points return NaN pixels, zero depth and false visibility.
Finite off-image points retain coordinates/depth but have false visibility
when bounds are requested. Inverse projection rejects nonfinite pixels and
nonpositive optical depths.

## SLF consistency and limits

SLF uses this model for crops, occluders and rays. Mask sampling uses the
nearest pixel centre, matching lifting. Pixel metric footprints come from
adjacent backprojected samples, including skew and distortion. Optimization
subtracts a local origin before float32 conversion and restores world
coordinates in float64.

The zero-offset initial model reproduces Phase 3's smoothed positions.
Accepted fits export exactly the evaluated positions for every track frame,
including those without a selected image. There is no second smoothing or
ground resampling after acceptance. Heading and bottom remain Phase 3's.

This is a rounded-box SLF adaptation; see the
[primary-source review](research/slf_projection_implementation_review.md).
Correct camera equations do not establish object ownership, correct masks,
ground or amodal dimensions. Overlays join projected corners with straight
lines; raw distorted images can have curved box edges. That display
approximation and a lower fitting loss do not validate 3D size.
