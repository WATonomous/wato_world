# Camera calibration measurements

Initial implementation evaluated 2026-11-09 (America/Toronto). This records
candidate-generation evidence; no WATO calibration has been accepted or
installed into the production pipeline.

## Backend and controls

The implemented backend uses Koide's unchanged NID objective at
`02a0dc039f5509708f384be4ff3228e0ae09352d`, Sophus SE(3) and Ceres BFGS in a
file-only runner. It builds and runs without ROS, PCL, GTSAM or a graphics
context. The dependency image records apt package versions; Python packages
are frozen in `docker/requirements/camera_calibration.txt`.

The native test plants a nonidentity camera mount, perturbs its rotation and
translation, and fits multiple analytic-texture image/cloud observations with
varying point depth. Its predeclared tolerances are 0.2 degrees and 0.04 m.
Separate tests check image-time world-to-ego conversion under moving poses,
preservation of a correct baseline, rejection of unstable translation and
the requirement for qualified independent validation evidence.

Measured native control error was **0.01249 degrees and 0.002083 m**. Tests on
analytic observations do not establish performance on real WATO camera/LiDAR
intensity responses. The native/common module container suite passed 254 tests
at final verification; one optional common progress test was skipped because this image
does not install tqdm. The native recovery test ran successfully.

## July WATO initial PANO_NN experiment

Bag: `sensors_2026_07_07_23_01_24_2_with_eidos_mcap`, using `lidar_cc` only.
Baseline SHA256:
`83afb0469640f976c2f9136d48f43739b6912200b58dfcd312d2fd2d2e862acf`.

Configuration: 24 sampled views, at most 6,000 points per view, accumulation
within 0.3 s of image time, 12 s scene blocks with a 1 s guard, two starts and
two training-block subset fits. Train/validation/test contained 14/5/5 views.
Each of those selected views occupied a distinct scene block. The initial run
took approximately 20 s in the dependency container, including sampling.

The candidate changed rotation by 3.351 degrees and translation by 0.107 m.
Maximum pairwise disagreement across four fits was **2.567 degrees and
0.440 m**, exceeding the experimental stability limits. Mean intensity-edge
distance on validation images changed 24.82 to 23.79 px, and on test images
25.19 to 24.34 px. These are diagnostic distances over intensity-boundary
points, not surveyed reprojection errors.

Outcome: **`insufficient_evidence`**. The validation profile is unqualified,
independent static correspondences have not been supplied, and the fits are
unstable. The effective transform stays recorded. This is a rejection case,
not a calibrated NN camera.

Local outputs, ignored by git:

- `data/artifacts/calibration_backend/wato_nn.log`: initial single-camera run.
- `data/artifacts/calibration_backend/benchmark.yaml`: its exact configuration.
- `data/artifacts/raw/<bag_id>/camera_calibration/`: latest report, backend
  workspaces and side-by-side held-out overlays; later runs can replace it.

## All-camera WATO run and nuScenes control

The same configuration was run on all eight panoramic cameras. Each had
14/5/5 train/validation/test views. All returned `insufficient_evidence`,
including failures of fit stability; no camera was promoted. Values below
are fitted candidates, not calibration measurements to install.

| Camera | Rotation change, degrees | Translation change, m | Rotation spread, degrees | Translation spread, m |
| --- | ---: | ---: | ---: | ---: |
| PANO_EE | 0.926 | 0.071 | 4.450 | 0.076 |
| PANO_NE | 2.921 | 0.231 | 2.915 | 0.360 |
| PANO_NN | 3.351 | 0.107 | 2.567 | 0.440 |
| PANO_NW | 4.920 | 0.039 | 5.393 | 0.119 |
| PANO_SE | 4.623 | 0.084 | 6.916 | 0.178 |
| PANO_SS | 4.818 | 0.054 | 4.704 | 0.301 |
| PANO_SW | 2.102 | 0.086 | 2.995 | 0.219 |
| PANO_WW | 2.785 | 0.077 | 6.198 | 0.157 |

The NN result reproduced the initial single-camera result. NW/SS also approached
the configured rotation limit, and SS regressed on the edge diagnostic. The
correct response is rejection rather than widening gates to accept these fits.
The all-camera log and config are `data/artifacts/calibration_backend/wato_all.log`
and `wato_all.yaml`.

On `NuScenes_v1_0_mini_scene_0061_0_mcap`, CAM_FRONT/LIDAR_TOP used 24 views
(15/5/4 split), 2 s blocks, 0.4 s guards and 0.15 s accumulation. Its candidate
changed rotation by 0.2309 degrees and translation by 0.0220 m. Spread across
four fits was 0.2426 degrees and 0.0327 m, within the experimental stability
limits. It remained `insufficient_evidence` because the profile is unqualified
and independent static correspondences were not supplied. This checks the
behavior on a known-calibration dataset; it does not qualify acceptance on WATO.
The exact config/log are `data/artifacts/calibration_backend/nuscenes.yaml`
and `nuscenes.log`.

Recorded calibration files were unchanged. The nuScenes baseline SHA256 is
`179bc78372302400807404d23f7f87a350cc6e677421b4d7baeee241126931a3`.

## Remaining work before automatic acceptance

Establish independently matched static landmarks across image regions and
depths, validate/freeze the absolute alignment thresholds on controls, and
evaluate a second WATO recording with adequate geometry. If translation remains
weakly constrained, a genuinely rotation-constrained fit is a separate backend
extension; discarding translation from a six-degree-of-freedom fit would require
revalidation and is not implemented as a shortcut.

Shipped profiles remain disabled and unqualified. The current validator needs
independent per-bag annotations; an automatic absolute-quality validator has
not been qualified. Candidate generation and report production are implemented;
production calibration correctness is not claimed.
