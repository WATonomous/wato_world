# Camera projection audit: WATO July 7 bag

**Updated:** the [independent image-motion calibration check](wato_alignment_root_cause.md)
finds a substantially larger NN orientation error and recovers actual moving
SUV returns during lifting. This audit uses correspondences inherited from
the original labels; those can preserve wrong foreground associations. Its
small mask-containment gain does not bound the physical calibration error.

Measured 2026-11-03 on
`sensors_2026_07_07_23_01_24_2_with_eidos_mcap`, all 11 chunks. This is a
calibration experiment, not a replacement production calibration. The bag's
`calibration.json`, masks, lifted labels and refined boxes were preserved.

Small camera rotation changes improve some projections, but do not explain
or resolve the whole WATO alignment problem. On the original 400-view
measurement, median car-return containment moves from **0.70 to 0.73**.
The nuScenes reference of 0.85 is still missed. SLF cars remain too tall in
the small downstream check. The evidence does not establish calibration as
the main cause of the bag-wide 2.24 m SLF car height.

## Calibration and projection checks

- All eight images are 1280 × 1024, with matching mask dimensions. Ingest
  records the rectified intrinsics from `CameraInfo.P[:3, :3]` and empty
  distortion. The raw focal lengths differ by 14–25 px; substituting the raw
  `K` would undo a correction already present. Ingest rejects non-identity
  rectification `R` and nonzero stereo `P[:, 3]`.
- Every camera rotation is orthonormal to numerical precision. The TF links
  describe an ideal ring: exact multiples of 45° in yaw, roll −90°, pitch 0°;
  camera centres are 0.24068 m from the ring centre, 1.977 m above
  `base_footprint`. These are nominal mounts, not evidence of measured
  camera-to-LiDAR alignment.
- Lifting uses `inv(ego_T_cam) @ inv(world_T_ego_at_image_time)` on deskewed
  world points. `FrameIndex` looks the pose up at the image's header timestamp,
  not the sweep timestamp. No transform-order or raw/rectified-image mismatch
  was found in this path. A regression test checks that the diagnostic's local
  correction gives the same pixels as the production projection, including
  nonzero intrinsic skew.
- Camera recording time minus header time has medians of 23–26 ms across
  cameras; the centre/corner LiDAR medians are 1.24/0.87 ms. Recording latency
  does not measure exposure-time bias and is not used as a timestamp correction.

## Fixed-correspondence feedback loop

`wato_semantic_lifting.calibration_audit` caches 2,000 distinct
(track, camera image) views, up to 250 per camera and 12 per track/camera:
1,949 car views and 51 pedestrian views. The points inside each original
observed track box, their supporting masks, and the scoring denominator stay
fixed during tuning. At least 20 projected returns and 100 mask pixels are
required; a seeded sample caps returns at 400 per view. Supporting masks in
one image are unioned. Camera frames are deduplicated per track by image timestamp across chunks
(camera sequence numbers restart in each chunk). Linked track
IDs are SHA-256 partitioned into 70% training / 30% validation, across all
chunks. Cars supply 1,420 training and 529 validation views.

The objective is clipped distance outside the original mask, normalized by
its smaller extent (at least 20 px), averaged per view. Returns leaving the
image remain misses. This prevents an apparent improvement from discarding
misprojected points. Containment uses lifting's nearest-pixel lookup.

These are **proxy correspondences**, not surveyed or annotated calibration
points. Observed boxes and supporting mask IDs were themselves influenced by
the old lifting. Masks can drift, points can include another surface, and
moving-object motion between return time and image time remains uncompensated.
Different LiDARs see different object faces; their raw per-scanner containment
rates are not an extrinsic-error measurement.

This broader sample is deliberately different from the earlier 400 views
(first three chunks, first matching camera per sampled car frame, integer
pixel truncation, denominator restricted to in-image points). Its rates
must not be compared directly to the earlier 0.70. Both before/after pairs
use their own unchanged samples.

## Parameter sweep

Each family is fitted separately on training cars with bounded Powell search:
optical-axis rotation components ±3°, local translation ±0.30 m,
principal point ±40 px, and uniform focal scale 0.95–1.05. No size priors,
masks, occlusion tolerances or observed boxes are tuned.

Rotations below are **local optical-frame rotation vectors**, in degrees:
`ego_T_cam_new = ego_T_cam @ Delta`, with optical x right, y down, z forward.
They are not Euler angles in the ego frame.

| Camera | Validation views / tracks | Median containment, recorded → rotation fit | Mean containment | Mean track gain [95% bootstrap interval] |
|---|---:|---:|---:|---:|
| PANO_EE | 45 / 29 | 0.500 → 0.525 | 0.487 → 0.486 | −0.003 [−0.019, +0.015] |
| PANO_NE | 57 / 25 | 0.590 → 0.547 | 0.540 → 0.515 | −0.028 [−0.065, +0.006] |
| PANO_NN | 69 / 39 | 0.421 → 0.490 | 0.417 → 0.480 | +0.061 [+0.013, +0.106] |
| PANO_NW | 83 / 53 | 0.637 → 0.667 | 0.632 → 0.627 | −0.009 [−0.042, +0.023] |
| PANO_SE | 40 / 13 | 0.488 → 0.405 | 0.546 → 0.497 | −0.092 [−0.180, −0.016] |
| PANO_SS | 84 / 22 | 0.849 → 0.905 | 0.739 → 0.793 | +0.080 [+0.023, +0.133] |
| PANO_SW | 86 / 37 | 0.635 → 0.740 | 0.622 → 0.672 | +0.026 [−0.032, +0.076] |
| PANO_WW | 65 / 35 | 0.671 → 0.705 | 0.613 → 0.642 | −0.002 [−0.048, +0.039] |

Track gains weight tracks equally, after averaging their view gains; the
2,000 bootstrap draws resample tracks. This avoids treating repeated views
of one car as independent evidence. The candidate includes only NN and SS:
at least 20 validation tracks / 25 views, mean track gain ≥0.02, positive
bootstrap lower bound, and lower outside-mask loss. Validation also rejects
camera regressions, so this is not an untouched test-set estimate. Multiple
camera comparisons and fragmented physical identities remain limitations.

| Camera exported | Optical rotation vector x / y / z, degrees |
|---|---|
| PANO_NN | −0.62395 / +0.13837 / −0.12930 |
| PANO_SS | +0.28473 / +0.25132 / −1.20679 |

On all 529 validation car views, this NN/SS candidate changes mean
containment **0.5883 → 0.6050** and median **0.6061 → 0.6276**. On 20
validation pedestrian views, mean is 0.5603 → 0.5539 and median remains
0.5455; there are too few to establish pedestrian quality. The original
400-view measurement gives 0.70 → 0.73 overall, NN 0.73 → 0.78, SS 0.77 →
0.76. Gains do not transfer uniformly even within this bag.

Translation fits commonly approach the 0.30 m search boundary, and several
intrinsic fits degrade validation. They are not exported as measured mount
or lens corrections. All fitted parameters and scores are in `sweep.json`.

Timestamp offsets −200 to +200 ms, in 10 ms steps, were swept **only on
parked tracks while ego speed exceeds 1 m/s**, with trusted poses across the
entire window. This separates ego timing from moving-object velocity as far
as these artifacts allow. Per camera there are only 0–6 validation views.
For example, NN's selected −120 ms has 7 training / 6 validation views and
only changes validation median 0.744 → 0.752; SW's +50 ms degrades 0.955 →
0.900 on three validation views. No reliable timestamp offset is identified
or exported.

## Downstream checks

The actual `_ChunkLifter.sweep` was run twice on the same inputs for 100
seeded validation NN/SS views across all chunks. This includes its original
ground rejection, depth map, LiDAR occluder and visibility rules. The fraction
of fixed non-ground centre-LiDAR returns receiving a vote from their own
supporting masklet in that camera changes as follows:

| Camera | Views | Mean fraction after visibility, recorded → candidate |
|---|---:|---:|
| PANO_NN | 48 | 0.4517 → 0.5125 |
| PANO_SS | 52 | 0.6371 → 0.6811 |

These are mask-consistency proxies; they do not measure semantic precision
against ground truth. No lifting artifacts were overwritten. Depth fitting
was held fixed to isolate projection; it would need refitting before a
production calibration change.

SLF was refit for the five validation moving-car tracks represented in NN/SS
that had an accepted original SLF fit. The original selected images, Phase 3
initialization, masks, points, ground and all optimizer settings were fixed;
only the camera rotations changed. Baseline refits reproduce stored heights.

| Track | Fitted height, recorded → candidate | Silhouette residual | Candidate accepted |
|---|---:|---:|---|
| 0005-122 | 2.590 → 2.549 m | 0.654 → 0.639 | yes |
| 0009-589 | 2.465 → 2.415 m | 0.555 → 0.533 | yes |
| 0009-178 | 2.769 → 2.630 m | 0.618 → 0.591 | yes |
| 0004-348 | 2.869 → 2.416 m | 0.491 → 0.450 | yes |
| 0004-372 | 2.428 → 1.374 m | 0.529 → 0.594 | **no** |

Among the four accepted in both runs, median fitted height falls from 2.680
to 2.482 m and remains excessive. The rejected fifth fit's shorter height
cannot be counted as an improved label. This small, selected subset is not
a new estimate of the bag-wide 2.24 m median.

## Reproduce and inspect

From the repository root, using the host environment:

```bash
export ARTIFACT_ROOT_URI="file://$PWD/data/artifacts"
export PYTHONPATH=src/common/src:src/semantic_lifting/src
export OPENBLAS_NUM_THREADS=1
PROJECTION_BAG=sensors_2026_07_07_23_01_24_2_with_eidos_mcap
PROJECTION_AUDIT=data/artifacts/projection_audit/$PROJECTION_BAG/v2
venv/bin/python -m wato_semantic_lifting.calibration_audit "$PROJECTION_BAG" \
  --canonical-lidar lidar_cc --cache "$PROJECTION_AUDIT/cache" \
  --out "$PROJECTION_AUDIT"
```

The first run collects the correspondences; later runs use compressed numeric
NPZs and JSON, bound to sampling settings and upstream index/manifest hashes.
`--chunks 0000 0001 0002` restricts collection. A cached evaluation takes
seconds and can fail on the remaining alignment symptom:

```bash
venv/bin/python -m wato_semantic_lifting.calibration_audit "$PROJECTION_BAG" \
  --canonical-lidar lidar_cc --cache "$PROJECTION_AUDIT/cache" \
  --out "$PROJECTION_AUDIT" --candidate "$PROJECTION_AUDIT/calibration_candidate.json" \
  --check-min-inside 0.85
# FAIL: validation car containment 0.6275862068965518 < 0.85
```

The experiment directory contains `calibration_candidate.json`,
`selected_rotations.json`, `sweep.json`, `evaluation.json`, `lifting_check.json`,
`slf_check.json` and `overlays.png`. Bag-specific read-only downstream and
original-measurement replay scripts are retained under its `replay/` directory.
The overlay shows lower/upper gain examples, including regressions, on the
same pixels before and after correction. Replays for SLF/original measurements
also need `src/label_refinement/src` on `PYTHONPATH`; the plot needs matplotlib.

The useful next calibration measurement is independent static scene
correspondences spread across each camera's field of view, near and far,
including stationary-ego images. That would separate physical mount error,
timing, moving-object mismatch and mask/track contamination. This experiment
supports an NN/SS candidate for further evaluation, not deploying a rig-wide
calibration or declaring the SLF-height problem fixed.

Validation of the diagnostic and projection/calibration path: **93 tests
passed**, including synthetic rotation recovery on a different depth/image
position, agreement with production projection, fixed-denominator misses,
cache validation, and a regression for camera sequence numbers restarting
across chunks. The 0.85 bag-containment check still fails, as reported above.
