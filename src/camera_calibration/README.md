# camera_calibration

Fits one constant camera-to-ego rigid transform per camera per bag using
Koide's direct LiDAR intensity/image registration objective. Runs after
`lidar_preprocessing`, before `detection_3d` and `perception_2d`. Camera
intrinsics, LiDAR mounts, poses and timestamps remain fixed.

Candidate generation works on ordinary driving data with trusted image-time
poses and deskewed static intensity returns. The supplied profiles are
**disabled by default and unqualified for automatic acceptance**. Initial WATO
and nuScenes results are in [the measurements](../../docs/camera_calibration_measurements.md).
A low optimizer cost is not a measured physical calibration.

## Run

Build this module using the repository's normal two-phase build. Add
`camera_calibration:dev` to your local `ACTIVE_MODULES` when developing, or
build its source/dependencies and deploy service explicitly:

```bash
docker compose --env-file modules/.env -f modules/docker-compose.yaml \
  --profile camera_calibration_pre build camera_calibration_source camera_calibration_deps
docker compose --env-file modules/.env -f modules/docker-compose.yaml \
  --profile camera_calibration build camera_calibration
```

Then, from the repository root:

```bash
./watod run camera_calibration --bag BAG_ID \
  --config /ws/src/camera_calibration/config/camera_calibration.wato.yaml \
  --enable --cameras PANO_NN
./watod run camera_calibration --bag BAG_ID \
  --config /ws/src/camera_calibration/config/camera_calibration.wato.yaml \
  --enable --force
./watod run camera_calibration viz --bag BAG_ID
./watod test camera_calibration
```

Repeat `--cameras` to restrict several cameras. Omitting it uses the config's
camera list, or every camera in the recorded calibration. `--prepare` exports
the sampled pairs without invoking the native backend. Both prepared-only and
fitted runs write an explicit decision report. `viz` prints decisions and writes a local HTML
gallery of saved overlays. These are bag-level transforms; there is no per-chunk
calibration flag.

The pipeline includes the stage after LiDAR preprocessing. Disabled profiles
log a skip. `--force camera_calibration` in `watod pipeline` forces every later
stage as well. A changed input/config/backend requires `--force`; completed
runs are reused only when their identity matches.

## Data and coordinates

Use an explicitly configured physical LiDAR (`lidar_cc` in the WATO profile).
Frames and sweeps are deduplicated across overlapping chunks by timestamp and
sensor ID. Sampling spans the bag, with temporal blocks assigned to train,
validation and test; guarded block edges prevent shared accumulation windows
crossing splits.

Each local cloud uses static returns from the processed sweeps, preserving
intensity and useful ground texture. Voxel reduction averages XYZ and
intensity; intensity is normalized by each view's 1st/99th percentiles. The
final `global_static_map.npz` has no intensity and is not used. Invalid poses,
missing intensity, weak intensity variation and insufficient overlap are
reported rather than replaced with invented values.

```text
p_ego_at_image = inverse(world_T_ego_at_image) @ p_world
p_cam          = inverse(ego_T_cam) @ p_ego_at_image
```

The same `ego_T_cam` can therefore fit all image/cloud pairs while the car
moves. The native result is already `ego_T_cam`; do not compose the LiDAR
mount a second time. Stored rectified images use their P-derived K with zero
distortion. The backend rejects nonzero skew. Supported upstream projection
models are pinhole/plumb_bob, rational_polynomial and equidistant/fisheye;
shared image-model validation runs first.

## Backend

`native/` is a small file-based runner around **unchanged upstream `NIDCost`**
at `02a0dc039f5509708f384be4ff3228e0ae09352d`, with Ceres BFGS on a Sophus SE(3)
manifold. It has no ROS or graphics runtime and does not use SuperGlue or
downstream labels. Native visibility is refreshed between outer iterations;
translation and rotation are bounded relative to the recorded mount.

The Docker build pins upstream and submodule commits, freezes Python packages,
and records native apt versions in its build directory. The apt repository's
transitive dependency resolution is not fully frozen; the exact package-list
hash is recorded with backend identity. Preserve the built image for exact
reproduction. Upstream states MIT licensing in its README; its source and the
Sophus license remain in the dependency image.

Python exports grayscale uint8 image buffers and little-endian float64 Nx4
XYZ/intensity buffers. These are internal workspace files, not public pipeline
artifacts. Solver invocations have a wall-clock timeout and log separately for
each start/subset. Selection among initial starts uses training objective only.
Independent training-block subset fits expose unstable solutions.

`solver.motion_model` is `rigid` by default. `rotation_only` genuinely freezes
the camera centre in ego coordinates during optimization using a quaternion
manifold with constant translation. It estimates rotation only and explicitly
assumes the recorded position; it does not measure that position. The same
independent validation gates still apply. The native tests verify that the
nonzero camera centre remains fixed while rotation is recovered.

## Decisions and independent validation

The report distinguishes `refined`, `recorded_valid`, `insufficient_evidence`,
`rejected`, `error` and `not_requested`. Unaccepted cameras keep their recorded
transform in the effective calibration, but that does **not** make them eligible
for validated downstream use.

Acceptance requires independent views/blocks, sufficient image/depth coverage,
absolute held-out reprojection quality, improvement with scene-block bootstrap
support, plausible corrections, stable starts/subset fits and no edge-metric
regression. The intensity-boundary-to-image-edge distance is diagnostic, not an
absolute correctness reference. Intrinsic/pose/time errors can still confound
the rigid fit. The initial numerical limits are experimental and have not been
qualified for WATO production use.

A qualified profile requires independently checked static correspondences in
an authored JSON under the mounted artifact tree:

```json
{
  "schema_version": 1,
  "bag_id": "BAG_ID",
  "cameras": {
    "PANO_NN": [
      {
        "timestamp_ns": 1783465857908281600,
        "provenance": "independent_static",
        "world_points": [[1.0, 2.0, 3.0]],
        "pixels": [[640.0, 512.0]]
      }
    ]
  }
}
```

Coordinates above illustrate the format only. Use actual matched static
landmarks and exact sampled validation/test image timestamps from `sample_index`
in the report. Points and pixels are Nx3/Nx2 arrays; the image-time pose comes
from the sampled frame. Training observations do not contribute to validation.
Do not derive correspondences from labels made with the original calibration.
Pixels use K's convention: integer coordinates are pixel centres.

### Picking correspondences

`pick` writes a browser picker for every validation/test view in the current
report. Run the stage first (`--prepare` is enough), with the sampling config
the scored run will use: picks are keyed on the sampled image timestamps.

```bash
./watod run camera_calibration pick --bag BAG_ID [--cameras PANO_NN]
python3 -m http.server 8000 --bind 127.0.0.1 \
  --directory data/artifacts/raw/BAG_ID/camera_calibration
# open http://localhost:8000/picker/ (VS Code forwards the port)
```

Each view shows the image beside the static `lidar_cc` returns within
`--window-s` (default 1 s) of it, in world coordinates. Click a pixel, click
the same feature in 3D (shift-click several returns to average them), then
Enter. The 3D view starts at the recorded camera pose for orientation only;
LiDAR is never drawn on the image, so no pick is steered by the calibration it
judges. Good features: pole bases, sign and building corners, lane-marking
corners, spread across the image and over near and far depths. The header
counts picks against `min_correspondences` and `min_blocks_per_split` per
split.

Picks autosave in the browser. Download `correspondences.json` into
`camera_calibration/`, where the page also reloads it from, and point
`validation.correspondences` at `/data/artifacts/raw/BAG_ID/camera_calibration/correspondences.json`.
Rerun with `--force`: the report then scores the recorded and candidate
transforms on the picks, which is meaningful even while the profile stays
unqualified. A 32-beam scanner resolves corners to a few centimetres, about
2 px at 15 m, so the 2 px median gate also measures picking error.

After a benchmark establishes the procedure and thresholds, set
`validation.qualified: true`, a documented `validation.profile_id`, and
`validation.correspondences` in a dedicated config. This initial version still
requires those independent annotations for each bag; fully automatic absolute
validation remains future work. Profile qualification is not inferred from an
optimizer success flag.

## Artifacts and downstream selection

`raw/<bag_id>/calibration.json` remains ingest's recorded/imported baseline.
The stage writes `camera_calibration/{candidate,effective,report,manifest}.json`,
an internal `workspace/`, and held-out overlays in `diagnostics/<camera>/`.
Only `refined` cameras change in `effective.json`. Intrinsics, LiDAR transforms,
sensor IDs and the raw TF dump remain baseline provenance. One bag writer is
allowed; publication uses atomic files and a completion manifest written last.

`wato_common.artifacts.calibration_refinement` owns the writer and readers.
The resolver verifies hashes, sampled inputs, baseline identity and camera
decisions. Set `calibration_source: validated` in detection, perception, lifting
and label-refinement configs to request this source explicitly. Proposal camera
overlays use the same config field. Generic and shipped WATO configs currently
keep `recorded`; a partial-camera experiment cannot validate the other cameras.
Rebuild affected deploy images before switching source; dev images need their
updated common/module source mounts.

Validated reads fail on incomplete/stale reports, unqualified profiles, or any
required camera without `refined`/`recorded_valid`. There is no silent fallback.
Calibration identities bind resumable output caches and propagate through
proposal/tracking/refinement manifests. Changing calibration requires rerunning
all affected stages with `--force`, including depth fitting. Changing LiDAR
mounts or poses also requires LiDAR preprocessing first.

See the [implementation plan](../../docs/camera_calibration_plan.md) for remaining
benchmark and rollout requirements.
