# semantic_lifting: measurements behind the defaults

The measurements behind semantic_lifting's defaults. The behaviour itself is in
the component README. Everything here is `may_30_ring_road_test_5_2` chunk 0000
(WATO rig: 8 rectified pano cameras at 1280×1024, fx ≈ 1030 px; `lidar_cc`
VLP-32C plus two VLP-16 corners at ~20 Hz; 1,884 valid sweeps, ~11.5k points
each; 956 masklets), sampling every 6th sweep in time order unless stated.

## Voting: class first

Masklets are per-camera and no cross-camera id exists until `tracking`,
which runs after lifting. Voting on `(instance_id, class)` therefore split
every point two cameras saw: in 40 sweeps, 2,994 of 15,034 labeled points (20%)
were "disagreements" with confidence ≈ 0.5, and every one was the same class
from two cameras. Voting on class first turns all of them into
`n_supporting_cameras = 2`, confidence 1. Over the whole chunk, 1,790 of
844,436 labeled points (0.2%) have a camera voting another class.

## Visibility: the depth map is only good up close

LiDAR depth vs perception_2d's metric depth at the point's pixel, on patches
with no depth edge in the 7×7 window (so occlusion can't explain the gap):

| Range | Median abs error | Within 0.5 m |
|---|---|---|
| 5–10 m | 0.10 m | 91% |
| 10–20 m | 0.47 m | 52% |
| 20–30 m | 2.1 m | 28% |
| 30–50 m | 20 m | 6% |
| 50–80 m | 9.7 m | 4% |

The fit is one inverse-depth scale + shift per frame, anchored on mostly near
static points, so error grows ~d² (perception_2d reports `rmse_inliers_m` p50
0.22 m, but that is over its near anchors). With Eq. 1 against the map alone,
39–62% of in-mask points beyond 10 m failed; on a row of parked cars at
20–40 m (PANO_SE, frame 227) it kept 10 of 853 points that were visibly on the
car bodies. Only depth-map surfaces nearer than `depth_map_max_range_m` = 10 m
occlude now.

Depth `0` (fitted inverse depth ≤ 0, 19.5% of pixels: sky and far background)
used to pass every point within 3 px of it; as +∞ it flips 164 of 594k
projected points.

## Visibility: the LiDAR occluder

The sweep's own non-ground returns, projected, give the nearest surface around
a pixel: a z-buffer of `lidar_cell_px` cells, min-pooled over
(2·`lidar_neighborhood_cells`+1)² cells, passed when
`d ≤ occluder · (1 + lidar_tolerance_per_m) + tolerance_m`.

Graded against the depth map where it is accurate (in-mask points at 5–10 m;
34,347 the map calls visible, 5,394 occluded):

| cell px, cells, per-m | keeps map-visible | rejects map-occluded | kept at 30–50 m |
|---|---|---|---|
| 8, 1, 0.05 | 99.8% | 4.9% | 65% |
| 8, 2, 0.05 | 99.6% | 12.4% | 45% |
| 16, 1, 0.05 | 99.5% | 16.2% | 38% |
| 24, 2, 0.10 | 99.0% | 25.8% | 17% |

The LiDAR rarely contradicts a visible verdict, and misses most near occluders
the dense map sees (thin structure between rings, parallax), which is why the
map still decides up close. Wider windows reject more, but on camera overlays
of far cars (PANO_SE 227, PANO_EE 109, PANO_NW 194, PANO_NE 172) the extra
rejections are points on car bodies standing behind another car; at 8 px / 1
cell the rejected far points sit in window glass (returns from behind the car)
and above roofs. Ground is excluded as an occluder: it sits in front of an
object's lower edge in the image without hiding it.

## Visibility: the combined test

In-mask points kept, by LiDAR range (every 6th sweep):

| Rule | 0–5 m | 5–10 m | 10–20 m | 20–30 m | 30–50 m | 50 m+ |
|---|---|---|---|---|---|---|
| depth map only (before) | 78% | 86% | 69% | 37% | 25% | 42% |
| LiDAR only | 100% | 99% | 93% | 84% | 65% | 48% |
| map < 10 m and LiDAR (shipped) | 78% | 86% | 88% | 84% | 65% | 48% |
| map < 15 m and LiDAR | 78% | 86% | 75% | 80% | 63% | 48% |
| points | 31k | 40k | 108k | 104k | 109k | 70k |

Whole chunk: 621,699 → 844,436 labeled points; labeled / in any mask 57% →
77%.

## Dynamic points (not compensated)

Velocity per Step F motion cluster (`motion_score > 1`) from a centred line
fit of its track's box centres over ±0.25 s; per-point times recomputed as
deskew does (azimuth, cw, 50 ms, stamp at sweep end).

- 12.1% of labeled points are on moving clusters.
- |t_camera − t_point|: median 26 ms, p90 61–68 ms, p99 85 ms (the sweep's 50
  ms of firing plus up to ~42 ms to the nearest camera frame).
- Object displacement in that gap: median 0.1 m, p90 0.6 m, p99 1.1 m.

Shifting moving-cluster points by v × (t_camera − t_point) before projecting,
share landing inside a mask (velocity scale −1 = shifted the wrong way):

| Displacement | Points | −1 | 0 (shipped) | +1 | +1.5 |
|---|---|---|---|---|---|
| < 0.2 m | 52.6k | 55.0% | 54.9% | 54.4% | 54.2% |
| 0.2–0.5 m | 14.6k | 51.5% | 59.0% | 61.8% | 62.1% |
| > 0.5 m | 10.2k | 44.5% | 54.4% | 59.2% | 58.9% |

The timing model checks out (the wrong sign costs ~10 points, the best scale is
~1), but the gain is ~5 points on the fastest ~13% of moving points, ~1.5% of
labels, and it would need lidar_preprocessing to keep per-point times and
export cluster velocities. Deferred until box fits on fast movers show the
symptom (short or shifted along the direction of travel).

## Runtime

Before the per-frame cache and array votes: 440 ms per sweep (a third of it a
`mask.sum()` per vote whose result was never read; mask PNG decodes and depth
reads repeated for every sweep a frame is paired with). After: ~70 ms per sweep,
2 m 12 s for the chunk including the LiDAR occluder.

## Ground

`sensors_2026_07_07_23_01_24_2_with_eidos_mcap`, all 11 chunks (not the
`may_30` chunk above). With ground labelable, 1,512,549 of 6,935,255 labeled
points (21.8%) were Patchwork++ ground. Their height above the chunk's
`ground.npz` height grid (every 10th sweep of chunks 0000, 0004, 0009; 88,647
points): median 0.00 m, p25–p75 −0.02 to +0.04 m, 8.9% above 0.3 m, 5.0% above
0.5 m — the road surface, with a tail of curbs and the lowest returns on
objects. On camera overlays they come from masks whose lower edge reaches onto
the road and from masks covering the road outright. The visibility and
instance counts in the other sections include ground points.

Leaving ground unlabeled changes no non-ground label (every sweep of chunks
0000 and 0009 keeps the same labeled non-ground points). It splits 37 of chunk
0009's 157 cross-camera instances, none of chunk 0000's 20: 12 whose parts sit
> 5 m apart in the sweeps both claim points in (different objects whose masks
reached the same road), 5 within 2 m over ≥ 5 such sweeps (one object, merge
lost), 20 with too few shared sweeps or 2–5 m apart to judge. The `instances.*`
thresholds were set with ground claims counted.

## Instances

Per-camera masklets of one object, merged from the visible LiDAR points they
both claim (whole chunk, after the visibility change).

- 774 masklet pairs from different cameras share points; 689 same-class, 85
  not. Shared points per pair: median 17, p90 671. Nothing separates true from
  incidental overlap on its own: on BEV renders, a road-wide "car" masklet in
  PANO_EE (1,036 claimed points; perception_2d's background-drift bug) touched
  real cars for 9 and 5 points (shared / smaller claim 0.07, 0.03), while one
  car split across PANO_EE (front) and PANO_SE (rear) shared hundreds at > 0.5.
- Thresholds: ≥ 20 shared points, ≥ 0.3 of the smaller claim: 270 pairs pass.
  Pairs between 0.3 and 0.5 included a pedestrian seen by PANO_NW and PANO_WW.
- Separation: the passing pairs' claimed points sit a median 1.0 m apart (p75
  2.1 m, p90 8.1 m). Without the separation check, transitive merges built
  groups whose members were far apart in the same sweeps; 3 m and 5 m give
  nearly the same groups (705 vs 701 instances), and 5 m leaves room for a
  long vehicle's front and rear views.
- Masklets themselves are noisy: 60% of unmerged masklets have their claimed
  points' median jump > 5 m between consecutive sweeps < 150 ms apart (drift,
  background bleed). Merged groups jump 0.257 times per sweep vs 0.252 for
  their members alone, so merging adds little on top.
- Result: 692 instances from 883 voting masklets; 108 span cameras (up to 6)
  and carry 69% of labeled points.

## WATO camera alignment (July 7 bag, 2026-11-03)

`sensors_2026_07_07_23_01_24_2_with_eidos_mcap`, all 11 chunks. No
production calibration was changed by any of this.

**What the recorded calibration is.** All eight cameras are 1280 × 1024,
rectified (`K = CameraInfo.P[:3,:3]`, no distortion; the raw focal lengths
differ by 14–25 px, so substituting raw `K` would undo the rectification).
The TF mounts are an ideal ring: yaw in exact multiples of 45°, roll −90°,
pitch 0°, centres 0.24068 m from the ring centre and 1.977 m above
`base_footprint`: nominal, not measured. Lifting's transform order and
image-time pose were checked, and the audit's projection matches
production's pixel for pixel (a regression test). Camera recording time
minus header time is 23–26 ms (LiDAR 0.9–1.2 ms); that is latency, not an
exposure-time bias, and is not used as a correction.

**The mask-containment audit** (`calibration_audit`, below) fits per-camera
rotation (±3°), translation (±0.30 m), principal point (±40 px) and focal
scale (0.95–1.05) to 2,000 (track, image) views, the points inside each
original observed box held fixed with their supporting masks, track ids
split 70 / 30 by hash. Validation cars, median containment recorded →
rotation fit:

| Camera | Views / tracks | Median containment | Mean track gain [95% bootstrap] |
|---|---:|---:|---:|
| PANO_EE | 45 / 29 | 0.500 → 0.525 | −0.003 [−0.019, +0.015] |
| PANO_NE | 57 / 25 | 0.590 → 0.547 | −0.028 [−0.065, +0.006] |
| PANO_NN | 69 / 39 | 0.421 → 0.490 | +0.061 [+0.013, +0.106] |
| PANO_NW | 83 / 53 | 0.637 → 0.667 | −0.009 [−0.042, +0.023] |
| PANO_SE | 40 / 13 | 0.488 → 0.405 | −0.092 [−0.180, −0.016] |
| PANO_SS | 84 / 22 | 0.849 → 0.905 | +0.080 [+0.023, +0.133] |
| PANO_SW | 86 / 37 | 0.635 → 0.740 | +0.026 [−0.032, +0.076] |
| PANO_WW | 65 / 35 | 0.671 → 0.705 | −0.002 [−0.048, +0.039] |

Only NN and SS passed (≥ 20 tracks, mean gain ≥ 0.02, positive lower bound):
all 529 validation car views 0.606 → 0.628 median; the earlier 400-view
measurement 0.70 → 0.73, still short of nuScenes' 0.85. Translations ran to
the search bound and intrinsic fits hurt validation, so neither was kept.
Image-time offsets (−200 to +200 ms, parked tracks while the ego moved)
had 0–6 validation views per camera: no offset is supported. Through the
actual lifter, the share of a mask's own returns voted for rose 0.45 → 0.51
(NN) and 0.64 → 0.68 (SS); refit, the four moving cars whose SLF fits
were accepted both times stayed too tall (median 2.68 → 2.48 m).

**An independent check finds a larger error.** Matching SIFT features
between image pairs and fitting one constant rotation per camera to the
epipolar geometry the ego motion predicts (no masks, labels or boxes;
chunks 0008–0010 held out):

| Camera | Held-out pairs | Median epipolar error, px | Optical rotation vector x / y / z, ° |
|---|---:|---:|---|
| NN | 19 | 1.242 → 0.532 | −2.667 / +2.021 / −0.017 |
| NE | 15 | 2.733 → 0.505 | −2.907 / +1.520 / +0.081 |
| EE | 14 | 2.975 → 0.696 | −0.568 / +2.640 / −2.075 |
| SE | 15 | 1.563 → 0.543 | +1.821 / +1.869 / −0.098 |
| SS | 16 | 0.770 → 0.460 | +1.383 / +0.782 / −0.010 |
| SW | 15 | 0.889 → 0.641 | +1.400 / +1.112 / +1.165 |
| WW | 15 | 2.951 → 0.576 | +0.921 / +0.044 / +2.140 |
| NW | 15 | 0.862 → 0.535 | −2.456 / +1.426 / −0.123 |

(`ego_T_cam_new.R = ego_T_cam.R @ Exp(rotvec)`, optical x right, y down,
z forward.) For NN, applying its estimate to a moving SUV whose returns
LiDAR preprocessing selected on its own (no camera) raised the returns
inside its mask from 0.7% to 89% over 14 held-out views. Replaying the
actual lifter and `lifted.proposals()` on chunk 0005 PANO_NN image 50 (masks,
depth and fitting held fixed), the car mask beside the signpost claims 16
returns of that SUV instead of 0, 7 returns nearer than 15 m instead of 38,
and its box grows from a 0.63 × 0.17 m column to 3.65 × 1.17 m; the two near
cars lose their > 20 m background claims (65 → 0, 37 → 0). On chunk 0004's
seven car targets, projected-box / mask IoU goes 0.300 → 0.693; over 165 NN
views, 0.346 → 0.634.

The two audits disagree: the independent NN rotation *lowers* the
containment proxy (0.421 → 0.088), because that proxy's correspondences came
from labels made with the old calibration (in NN50 they hold the foreground
column, not the car). Containment on label-derived points therefore can't
accept or reject a physical calibration. Nor is the epipolar fit a
certification: it depends on ego poses and exposure timing, near-straight
driving constrains rotation weakly, and on nuScenes scene-0061 it preferred a
spurious roll. With all eight estimates, a frozen-input SLF refit accepted 1
of 5 cars (the other four missed the gain or residual gates; none was
relaxed).

**Where this leaves it.** WATO's camera orientations are likely off by
2-3°, NN's most clearly, and that misassigns points during lifting. A fix
is a measured calibration supplied as data for this rig (with depth refit
and every downstream stage rerun), not a default: the fitted rotations don't
belong in generic code. The next measurement is independent static-scene
correspondences across each camera's field of view. The epipolar estimates
came from a local diagnostic that isn't in the repository;
`calibration_audit` is.

```bash
python -m wato_semantic_lifting.calibration_audit \
    sensors_2026_07_07_23_01_24_2_with_eidos_mcap --canonical-lidar lidar_cc \
    --cache <dir>/cache --out <dir> [--candidate <calibration.json>] \
    [--check-min-inside 0.85]
```
