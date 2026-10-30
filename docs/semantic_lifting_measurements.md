# semantic_lifting: measurements behind the defaults

The measurements behind semantic_lifting's defaults. The behaviour itself is in
the component README. Everything here is `may_30_ring_road_test_5_2` chunk 0000
(WATO rig: 8 rectified pano cameras at 1280×1024, fx ≈ 1030 px; `lidar_cc`
VLP-32C plus two VLP-16 corners at ~20 Hz; 1,884 valid sweeps, ~11.5k points
each; 956 masklets), sampling every 6th sweep in time order unless stated.

## Voting: class first

Masklets are per-camera and `global_object_id` stays null until `tracking`,
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
