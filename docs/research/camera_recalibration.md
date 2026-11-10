# Camera recalibration for the WATO rig

Researched 2026-11-09. This note proposes a measurement and import workflow;
it does not change calibration artifacts, ingest configuration, or the car.

## What needs measuring

Treat **intrinsics** (the lens and image mapping) separately from
**extrinsics** (the optical camera's rigid pose relative to the LiDAR/ego).
Start by checking existing intrinsics, then measure extrinsics against
independent static geometry. The current evidence points most clearly to
orientation error; it does not establish that every intrinsic must change.
[The existing WATO measurements](../semantic_lifting_measurements.md#wato-camera-alignment-july-7-bag-2026-11-03)
record nominal ring mounts and independent epipolar estimates of roughly
2–3° errors, strongest for NN. Those estimates depend on pose and exposure
timing and are **not certified calibration**.
Start with PANO_NN against independent static LiDAR geometry, then extend to
the other cameras once the capture, frame conversion and validation agree.

Do not promote `calibration_audit`'s label/mask-containment candidate into
production. Its point correspondences were selected using labels made with
the original calibration: an independently better NN rotation lowered that
proxy. The measured disagreement and held-out checks are documented in the
same [measurement section](../semantic_lifting_measurements.md#wato-camera-alignment-july-7-bag-2026-11-03).

## Where calibration lives

| Role | Authoritative location and behavior |
| --- | --- |
| Car intrinsic files | Sibling monorepo [`camera_calib/config/camera_*.yaml`](../../../wato_monorepo/src/interfacing/calibration/camera_calib/config/), installed as package configuration. [`DEVELOPING.md`](../../../wato_monorepo/src/interfacing/calibration/camera_calib/DEVELOPING.md) describes checkerboard capture and YAML export. |
| Car image production | [`template_camera_load.yaml`](../../../wato_monorepo/src/perception/perception_bringup/launch/template_camera_load.yaml) loads those files through `camera_info_urls`; Isaac ROS `RectifyNode` emits 1280 × 1024 rectified images; JPEG publishing wraps `image_rect`. |
| Car extrinsics | [`eve_roof_mount.xacro`](../../../wato_monorepo/src/interfacing/eve_description/urdf/eve_roof_mount.xacro) generates ideal ring mounts and optical-frame rotations; lower-camera mounts are explicitly nominal. [`eve_kia_soul_ev.xacro`](../../../wato_monorepo/src/interfacing/eve_description/urdf/eve_kia_soul_ev.xacro) connects this mount to `base_link`/`base_footprint`. |
| Offline input selection | [`ingest.wato.yaml`](../../src/ingest/config/ingest.wato.yaml) maps eight `image_rect_compressed`/`camera_info` pairs, three per-scanner clouds, and `base_footprint`; some bags need the six-camera or missing-NW-LiDAR profile, or raw-message `image_rect` topic names. |
| Offline calibration construction | [`inputs/calibration.py`](../../src/ingest/src/wato_ingest/inputs/calibration.py) reads the first CameraInfo and LiDAR frame IDs, resolves ego-to-sensor TF paths, or imports an authored JSON. It does not estimate calibration. |
| Frozen artifact contract | [`artifacts/calibration.py`](../../src/common/src/wato_common/artifacts/calibration.py) owns `raw/<bag_id>/calibration.json`: image-space `K`, distortion, dimensions, `ego_T_cam`, and `ego_T_lidar`; raw CameraInfo is provenance. The contract is here, rather than in the general [`schemas.py`](../../src/common/src/wato_common/schemas.py). |
| Projection and inverse rays | [`camera_model.py`](../../src/common/src/wato_common/geometry/camera_model.py), with the cross-component [camera geometry contract](../camera_geometry_contract.md). |

The monorepo's [`visual_odometry calibrate_pairs`](../../../wato_monorepo/src/perception/visual_odometry/README.md#calibration)
is a useful relative-pair diagnostic, but keeps NN/SS anchors and all
translations nominal in [`all6.yaml`](../../../wato_monorepo/src/perception/visual_odometry/config/all6.yaml).
Its override is local to that VO package; `wato_world` does not consume it.

## Match the model to the images

The repository supports these raw-image models; unknown models fail instead
of silently becoming pinhole. All projection/rays use the optical forward
hemisphere, with depth equal to optical z.
[Source: `CameraModel`](../../src/common/src/wato_common/geometry/camera_model.py).

| Image model | Accepted distortion coefficients |
| --- | --- |
| Undistorted (`""`, `none`) | Empty |
| `plumb_bob` | Empty, 4, or 5: `k1,k2,p1,p2[,k3]` |
| `rational_polynomial` | 8: `k1,k2,p1,p2,k3,k4,k5,k6` |
| `equidistant` / `fisheye` | 4: `k1,k2,k3,k4` |
| Omnidirectional / equirectangular | Unsupported by this pipeline |

BEVFusion's current [`detection_3d` input adapter](../../src/detection_3d/src/wato_detection_3d/inputs.py)
additionally rejects active distortion; raw-image support in the shared
camera model does not establish compatibility with that consumer.

For `rectified: true`, the artifact's `K` is **`CameraInfo.P[:3,:3]`** and
distortion is empty. `CameraInfo.K/D` describe raw optics and must not be
applied again. ROS defines that distinction and the optical axes as right,
down, forward in its [CameraInfo message](https://raw.githubusercontent.com/ros2/common_interfaces/rolling/sensor_msgs/msg/CameraInfo.msg).
For example, the July 7 NN artifact has rectified `fx≈1021.70`, while the raw
CameraInfo has `fx≈1046.42`; substituting the latter changes geometry.
[Repo measurements](../semantic_lifting_measurements.md#wato-camera-alignment-july-7-bag-2026-11-03).

Ingest rejects nonidentity rectification `R`, stereo translation in `P[:,3]`,
and unnormalized ROI/binning. Cropping/resizing also requires updated
image-space intrinsics and dimensions; an authored JSON must describe the
actual stored image and its actual optical frame.
[Builder](../../src/ingest/src/wato_ingest/inputs/calibration.py),
[dimension checks](../../src/common/src/wato_common/artifacts/calibration.py).

An old bag containing only rectified images cannot be re-rectified using new
raw lens parameters: the original pixels were already warped and may have
been discarded. An extrinsic-only correction can still use its existing
rectified `K`. A raw-optics correction needs original raw images and
regenerated images/calibration, or a separately established model for the
existing warped images. This follows from the
[ROS raw/rectified image contract](https://raw.githubusercontent.com/ros2/common_interfaces/rolling/sensor_msgs/msg/CameraInfo.msg).

## Practical intrinsic capture

Use ROS 2 `camera_calibration` on **`image_raw`** at the operational sensor
resolution, focus and lens settings. The existing car instructions specify
a flat board with **7 × 10 inner corners and 0.08686 m squares**; verify the
physical board before using these numbers. Capture full-field, near/far and
tilted views, holding still for accepted samples, then Calibrate and Save.
Keep independent views for reprojection and rectification checks.
[Car instructions](../../../wato_monorepo/src/interfacing/calibration/camera_calib/DEVELOPING.md),
[official ROS tutorial](https://github.com/ros-perception/image_pipeline/blob/rolling/camera_calibration/doc/tutorial_mono.rst).

```bash
# Example for the board specified above, in a sourced ROS 2 environment.
ros2 run camera_calibration cameracalibrator \
  --size 7x10 --square 0.08686 --no-service-check \
  --ros-args -r image:=/camera_pano_nn/image_raw -r camera:=/camera_pano_nn
```

Use a ROS namespace remap for `camera`, which prefixes the tool's
`camera/set_camera_info` client; the sibling instructions' `-p camera:=...`
sets a parameter instead. `--no-service-check` permits capture/export without
that service. Check installed `--help`: current source supports pinhole
radial-coefficient controls, separate fisheye controls, and a GUI model
selection. Do not assume “pano” means a fisheye model or that passing fisheye
flags alone selects it.
[ROS node source](https://github.com/ros-perception/image_pipeline/blob/rolling/camera_calibration/src/camera_calibration/nodes/cameracalibrator.py),
[service and GUI source](https://github.com/ros-perception/image_pipeline/blob/rolling/camera_calibration/src/camera_calibration/camera_calibrator.py).

Review the exported model, dimensions, K/D/R/P and held-out reprojection
residuals before replacing the car YAML. The sibling document's pixel-error
bands are local heuristics, not a guarantee of correct edge geometry or
extrinsics. Loading YAML alone does not rectify pixels; verify the actual
rectifier output and its P after deployment.
[Car instructions](../../../wato_monorepo/src/interfacing/calibration/camera_calib/DEVELOPING.md),
[ROS tutorial](https://github.com/ros-perception/image_pipeline/blob/rolling/camera_calibration/doc/tutorial_mono.rst).

## Practical LiDAR–camera extrinsics

Use Koide's [direct_visual_lidar_calibration](https://github.com/koide3/direct_visual_lidar_calibration)
as a practical ROS 2 candidate: it registers image texture against LiDAR
intensity without a target, including spinning scanners. It remains a tool
to validate on this rig, not a demonstrated WATO result. Pin the inspected
revision [`02a0dc0`](https://github.com/koide3/direct_visual_lidar_calibration/commit/02a0dc039f5509708f384be4ff3228e0ae09352d)
(main checked 2026-11-09; commit dated 2026-01-02) and its dependencies.

1. Establish intrinsics first; rigidly fix the camera and scanner. Collect
   several independent static textured scenes with overlapping geometry at
   different depths. The author recommends 5–10 bags, starting each with the
   rig still. For 16/32-line scanners, their collection procedure adds gentle
   motion to accumulate coverage; use the dynamic integrator and verify its
   registration instead of naively stacking scans. A vehicle-scale rigid rig
   may need a safely adapted capture procedure.
   [Official collection guidance](https://koide3.github.io/direct_visual_lidar_calibration/collection/).
   For existing WATO hardware, start with **fully stationary captures** and
   static integration (no `-d`). WATO point `time` fields are all zero; the
   tool synthesizes times only when the field is absent, and expects scan
   headers at the first point, whereas WATO uses sweep end. Validate/fix
   these conventions before any moving-scan accumulation.
   [WATO convention](../../CLAUDE.md),
   [tool timing source](https://github.com/koide3/direct_visual_lidar_calibration/blob/02a0dc039f5509708f384be4ff3228e0ae09352d/src/vlcal/common/time_keeper.cpp).
2. Choose one camera and one physical LiDAR topic explicitly. Avoid automatic
   topic selection on this multi-sensor rig, and avoid merged point clouds.
   For rectified WATO images explicitly supply **P intrinsics plus zero
   distortion**: the tool's ROS 2 CameraInfo reader extracts K/D and ignores P.
   It selects compressed decoding from `compressed` in the topic name.
   [Tool ROS 2 reader](https://github.com/koide3/direct_visual_lidar_calibration/blob/02a0dc039f5509708f384be4ff3228e0ae09352d/src/preprocess_ros2.cpp).
3. Preprocess, select independent 2D/3D correspondences for a manual initial
   guess, refine, and inspect. For the parameter substitution below, obtain
   `FX,FY,CX,CY` from the exact recorded image's P; they are placeholders.
   [Official programs](https://koide3.github.io/direct_visual_lidar_calibration/programs/),
   [CLI source](https://github.com/koide3/direct_visual_lidar_calibration/blob/02a0dc039f5509708f384be4ff3228e0ae09352d/src/vlcal/preprocess/preprocess.cpp).

```bash
ros2 run direct_visual_lidar_calibration preprocess CAPTURE_PARENT PREPROCESSED \
  --image_topic /camera_pano_nn/image_rect_compressed \
  --points_topic /lidar_cc/velodyne_points \
  --camera_model plumb_bob --camera_intrinsics FX,FY,CX,CY \
  --camera_distortion_coeffs 0,0,0,0,0 --intensity_channel intensity -v
ros2 run direct_visual_lidar_calibration initial_guess_manual PREPROCESSED
ros2 run direct_visual_lidar_calibration calibrate PREPROCESSED
ros2 run direct_visual_lidar_calibration viewer PREPROCESSED
```

`CAPTURE_PARENT` contains the capture bags. Check that the ROS 2 environment
can open their storage format. Repeat per camera, using a scanner with
sufficient overlap and a trusted ego mount. With sparse scans, inspect
integrated clouds for drift and corrupted/missing intensity before fitting.

Model compatibility needs explicit review. The inspected factory accepts
`plumb_bob`, `rational_polynomial`, and `fisheye`/`equidistant`, beyond the
older program page's list, but silently pads/truncates coefficient arrays.
The preprocessing CLI's explicit-model whitelist still rejects
`rational_polynomial`; factory support alone does not make that manual
workflow usable. Inspect/pin the actual preprocessing path for raw rational
inputs rather than relabeling coefficients as `plumb_bob`.
Its pinhole model takes only `fx,fy,cx,cy`, so **nonzero K skew is not
represented**. Validate coefficient counts yourself; use compatible
normalization rather than silently dropping skew. Tool-only omnidir or
equirectangular results do not add support to this repository.
[Factory](https://github.com/koide3/direct_visual_lidar_calibration/blob/02a0dc039f5509708f384be4ff3228e0ae09352d/src/camera/create_camera.cpp),
[preprocess whitelist](https://github.com/koide3/direct_visual_lidar_calibration/blob/02a0dc039f5509708f384be4ff3228e0ae09352d/src/vlcal/preprocess/preprocess.cpp),
[pinhole implementation](https://github.com/koide3/direct_visual_lidar_calibration/blob/02a0dc039f5509708f384be4ff3228e0ae09352d/include/camera/pinhole.hpp).

The result `T_lidar_camera = [x,y,z,qx,qy,qz,qw]` maps **camera points into
LiDAR coordinates**. Consequently, for column vectors:

```text
ego_T_cam = ego_T_lidar @ T_lidar_camera
cam_T_lidar = inverse(T_lidar_camera)
cam_T_world = inverse(ego_T_cam) @ inverse(world_T_ego_at_image_time)
```

Convert the quaternion in xyzw order; retain metres and optical axes. Confirm
the fitted camera frame is the same frame used by the image model, including
any rectification rotation.
[Tool result convention](https://koide3.github.io/direct_visual_lidar_calibration/example/#6-calibration-result-file),
[repo projection](../../src/semantic_lifting/src/wato_semantic_lifting/projection.py).

For a future car update, change each camera's **joint** origin, not its
visual/collision origin or the shared ring radius/angle. The xacro parent is
`base_link`; convert from artifact `base_footprint` coordinates (the vehicle
file contributes a 1.76 m vertical offset). Add per-camera measured
parameters/explicit joints to preserve other mounts. Rebuild the generated
URDF and relaunch the publisher; old bag TF will remain unchanged.
[Roof joints](../../../wato_monorepo/src/interfacing/eve_description/urdf/eve_roof_mount.xacro),
[vehicle transform](../../../wato_monorepo/src/interfacing/eve_description/urdf/eve_kia_soul_ev.xacro),
[URDF generation](../../../wato_monorepo/src/interfacing/eve_description/CMakeLists.txt).

## Verification and import

Accept a candidate only after independent static-scene image/point
correspondences improve on held-out scenes across the full field, at near
and far depths and for each camera/scanner combination. Report pixel
residual distribution and spatial patterns, not just an optimizer cost or
mask containment. Inspect occlusion boundaries: foreground/background
returns can be correct geometrically without landing in the same mask.
This is a proposed acceptance procedure addressing the documented
[audit bias](../semantic_lifting_measurements.md#wato-camera-alignment-july-7-bag-2026-11-03).

Then check moving data separately. Header times should represent acquisition,
not receipt; recording-minus-header latency does not establish an exposure
offset. In this repo camera poses are looked up at image time and LiDAR
points are deskewed at point time, but dynamic-object motion between them is
not compensated. The WATO nearest-frame tolerance of 60 ms is a pairing gate,
not synchronization. A rotation that compensates a timing error can look
better on one drive and fail elsewhere.
[CameraInfo timing definition](https://raw.githubusercontent.com/ros2/common_interfaces/rolling/sensor_msgs/msg/CameraInfo.msg),
[projection](../../src/semantic_lifting/src/wato_semantic_lifting/projection.py),
[WATO timing config](../../src/ingest/config/ingest.wato.yaml),
[existing timing findings](../semantic_lifting_measurements.md#wato-camera-alignment-july-7-bag-2026-11-03).

Author a **complete** candidate JSON beside its measurement report, preserving
the required camera/LiDAR entries and updating only measured fields. Record
the rig identity, capture dates, exact input images/model, solver revision,
frame convention, old/new values and held-out residuals in the report and
use an explicit `calibration_version`. `require_complete` checks presence
and supported camera geometry; it does not certify physical accuracy.
[Artifact contract](../../src/common/src/wato_common/artifacts/calibration.py),
[ingest validation](../../src/ingest/src/wato_ingest/inputs/calibration.py).

For a reviewed candidate, preserve the baseline using a fresh comparison ID:

```bash
# July 7 recording: preserve its missing-NW-LiDAR profile.
./watod run ingest --bag /data/bags/sensors_2026_07_07-23_01_24_2_with_eidos.mcap \
  --bag-id CAMERA_CALIBRATION_COMPARISON \
  --config /ws/src/ingest/config/ingest.wato_no_lidar_nw.yaml \
  --calibration /data/artifacts/eval_runs/calibration/CANDIDATE.json
./watod pipeline --bag CAMERA_CALIBRATION_COMPARISON --profile wato_no_lidar_nw \
  --from lidar_preprocessing
```

Use the original appropriate rig profile. `watod pipeline` does not forward
`--calibration`; supply it to ingest explicitly. A later ingest run that
actually processes chunks (forced, or filling missing outputs) rebuilds
calibration from the bag unless the override is supplied again; a run with
every chunk already present skips before writing it. Add `--force` to ingest and
`--force lidar_preprocessing` to the pipeline only when intentionally reusing
an ID whose outputs exist.
[Ingest CLI](../../src/ingest/src/wato_ingest/cli.py),
[pipeline construction](../../src/ingest/src/wato_ingest/pipeline.py),
[pipeline wrapper](../../watod_scripts/watod-pipeline.sh).

For a camera-only correction with decoded data retained, an alternative is
`python -m wato_ingest freeze-calibration --bag-id BAG_ID --from-file CANDIDATE.json`
inside the sourced ingest environment, then
`./watod pipeline --bag BAG_ID --profile wato_no_lidar_nw --from detection_3d --force detection_3d`.
Call that Python subcommand directly: the `watod run` wrapper does not expose
`freeze-calibration`. Recompute perception depth fitting and all later
stages, including BEVFusion: camera calibration affects detector geometry,
depth anchors and lifting. A
LiDAR/ego calibration change requires the earlier LiDAR processing rerun.
[Freeze CLI](../../src/ingest/src/wato_ingest/cli.py),
[run wrapper](../../watod_scripts/watod-run.sh),
[depth anchoring](../../src/perception_2d/README.md),
[rerun contract](../../CLAUDE.md).

## Feasibility of an automatic per-bag stage

**A candidate is feasible; correct extrinsics for every ordinary driving bag
are not established.** Koide's solver can jointly refine one rigid transform
over multiple image/cloud pairs. Its normal bag preprocessing instead takes
the first image per bag and integrates scans; the documented capture starts
at rest. Applying that procedure unchanged to a moving WATO drive is not a
validated solution. Fixed intrinsics, rigid mounts, overlapping static
geometry and informative intensity remain prerequisites.
[Collection requirements](https://koide3.github.io/direct_visual_lidar_calibration/collection/),
[ROS 2 first-image reader](https://github.com/koide3/direct_visual_lidar_calibration/blob/02a0dc039f5509708f384be4ff3228e0ae09352d/src/preprocess_ros2.cpp),
[shared-transform objective](https://github.com/koide3/direct_visual_lidar_calibration/blob/02a0dc039f5509708f384be4ff3228e0ae09352d/src/vlcal/calib/visual_camera_calibration.cpp).

The published [camera-map workflow](https://koide3.github.io/direct_visual_lidar_calibration/example_map/)
accepts one image and an intensity-bearing PCD/PLY map. Its result in that
case is a **camera-to-map pose for that image**, rather than a fixed
camera-to-ego extrinsic. The input structure has only an image and cloud,
with no image-time vehicle pose. Combining many world-map/image pairs
unchanged would incorrectly constrain their different camera poses to one
transform.
[Map preprocessing](https://github.com/koide3/direct_visual_lidar_calibration/blob/02a0dc039f5509708f384be4ff3228e0ae09352d/src/preprocess_map.cpp),
[pair structure](https://github.com/koide3/direct_visual_lidar_calibration/blob/02a0dc039f5509708f384be4ff3228e0ae09352d/include/vlcal/common/visual_lidar_data.hpp).

Recommended integration is a separate **camera-extrinsics refinement stage
after `lidar_preprocessing`**, before camera-dependent perception. Keep
intrinsics and LiDAR mounts fixed initially. An adapter would select diverse
image times, collect nearby deskewed static returns with intensity, and
transform each local cloud from world into **ego at that image's timestamp**.
Then the existing common-transform objective can fit one `ego_T_cam` per
camera across those pairs, without a new optimizer. World sweeps preserve
optional intensity and have aligned dynamic masks; `global_static_map.npz`
contains only voxelized XYZ, so it cannot directly supply this intensity
objective. Require intensity presence/quality rather than inventing values.
[Sweep contract](../../src/common/src/wato_common/artifacts/lidar_sweeps.py),
[map reduction](../../src/lidar_preprocessing/src/wato_lidar_preprocessing/reduce/_core.py).
This is an integration design inferred from the solver's shared objective;
it has not been implemented or tested on WATO data.

The adapter can seed `results.init_T_lidar_camera` from the recorded TF
(`ego_T_cam` when adapted clouds use ego coordinates), then invoke
`calibrate DATA --background --auto_quit`; the output key keeps its upstream
name despite the adapted reference frame. The source accepts this supplied
initial guess, avoiding interactive picking and feature-matcher initialization.
It still constructs a hidden viewer, so graphics-context requirements must
be checked in the intended container before claiming fully headless support.
[Calibration CLI and result writing](https://github.com/koide3/direct_visual_lidar_calibration/blob/02a0dc039f5509708f384be4ff3228e0ae09352d/src/calibrate.cpp).

Emit a candidate JSON and an independent held-out alignment report; promote
only accepted cameras and retain recorded TF when evidence is insufficient.
Low texture, poor overlap, dynamic contamination, pose/map error and exposure
timing can defeat or bias this estimate. Optimization convergence alone does
not establish physical accuracy. Reuse the verification/import procedure
above; camera-only refinement need not rebuild LiDAR geometry, while all
camera-dependent outputs must use the accepted calibration consistently.
