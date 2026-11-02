# proposal_generation: measurements behind the defaults

The measurements behind proposal_generation's defaults and what its first run
produced. The behaviour itself is in the component README. Everything here is
`sensors_2026_07_07_23_01_24_2_with_eidos_mcap` (WATO rig without `lidar_nw`:
frames of one `lidar_cc` VLP-32C sweep plus one `lidar_ne` VLP-16 sweep; 8 pano
cameras), all 11 chunks, `proposal_generation.wato.yaml`, unless stated.

## Fusion: per-frame overlap is small

Gated Step F clusters against lifted instances in the same frame, by shared
LiDAR points (chunks 0000 / 0005):

| Gated clusters | 1,211 / 2,728 |
|---|---|
| sharing any point with a lifted instance | 571 / 1,642 |
| best match: shared points p10 / p50 / p90 | 1 / 8 / 112 · 2 / 14.5 / 124 |
| matched at ≥ 20 shared and ≥ 0.3 of the smaller | 158 / 689 |
| matched at ≥ 5 shared and ≥ 0.3 | 205 / 1,118 |
| matched at ≥ 3 shared and ≥ 0.3 | 224 / 1,221 |

semantic_lifting's masklet-merge rule (≥ 20 points, ≥ 0.3) is counted over a
whole chunk; per frame it drops 40% of the matches ≥ 5 keeps on chunk 0005.
`fusion.min_shared_points: 5`.

## Lifted boxes: far and sparse

A lifted instance has a median of 4–8 points in a frame (p90 88–98); 35–46%
have ≥ 10, 48–65% ≥ 5. `lifted.min_points: 5`.

Instances with ≥ 5 points sit a median 51 m / 34 m from the reference LiDAR
(chunks 0000 / 0005, every 4th frame). The share of an instance's points in
its largest group, by `cluster_radius_m`:

| Radius | Median share kept | Keeping ≥ 90% | ≥ 90%, beyond 20 m |
|---|---|---|---|
| 0.5 m | 0.50 / 0.60 | 0.30 / 0.21 | 0.01 / 0.11 |
| 0.75 m | 0.60 / 0.71 | 0.39 / 0.32 | 0.08 / 0.21 |
| 1.0 m | 0.80 / 0.80 | 0.44 / 0.39 | 0.13 / 0.29 |
| 1.5 m | 0.96 / 0.86 | 0.59 / 0.46 | 0.36 / 0.36 |
| 2.0 m | 1.00 / 0.88 | 0.68 / 0.48 | 0.51 / 0.39 |
| 3.0 m | 1.00 / 0.91 | 0.72 / 0.51 | 0.57 / 0.43 |

Below 1.5 m the group splits real cars along VLP-16 rings (~0.7 m apart at
20 m); past 2 m the gain flattens, so what is left out is mostly bleed.
`lifted.cluster_radius_m: 2.0`.

## First run

| Chunk | Frames | lidar_mos | classed | lifted | fused | pairs | pairs / lidar_mos |
|---|---|---|---|---|---|---|---|
| 0000 | 629 | 1,211 | 0.17 | 8,023 | 9,097 | 137 | 0.11 |
| 0001 | 666 | 2,688 | 0.40 | 7,386 | 9,036 | 1,038 | 0.39 |
| 0002 | 666 | 1,678 | 0.26 | 2,135 | 3,324 | 489 | 0.29 |
| 0003 | 670 | 1,415 | 0.38 | 3,235 | 4,094 | 556 | 0.39 |
| 0004 | 674 | 2,144 | 0.36 | 7,011 | 8,206 | 949 | 0.44 |
| 0005 | 674 | 2,728 | 0.37 | 9,479 | 11,237 | 970 | 0.36 |
| 0006 | 675 | 2,791 | 0.57 | 6,545 | 7,776 | 1,560 | 0.56 |
| 0007 | 658 | 2,151 | 0.55 | 5,780 | 6,735 | 1,196 | 0.56 |
| 0008 | 657 | 2,397 | 0.37 | 10,033 | 11,605 | 825 | 0.34 |
| 0009 | 674 | 3,655 | 0.38 | 18,933 | 21,204 | 1,384 | 0.38 |
| 0010 | 567 | 1,972 | 0.47 | 6,503 | 7,516 | 959 | 0.49 |

"Frames" counts frames with at least one proposal; "classed" is the share of
lidar_mos boxes a lifted vote gave a class.

Bag totals: 24,830 lidar_mos, 85,063 lifted, 99,830 fused (10,063 pairs).

| Rows | n | l / w / h median (m) |
|---|---|---|
| lidar_mos with a class | 10,014 | 3.80 / 1.61 / 1.55 |
| lidar_mos without | 14,816 | 3.32 / 1.34 / 1.46 |
| lifted car | 75,116 | 2.26 / 0.83 / 1.42 (p90 4.57 / 1.89 / 3.28; 15 points) |
| lifted pedestrian | 5,494 | 1.12 / 0.24 / 1.29 (8 points) |
| lifted truck | 2,669 | 2.55 / 0.74 / 1.57 |
| lifted bus | 1,480 | 6.54 / 1.30 / 4.33 |
| fused pairs | 10,063 | 3.96 / 1.68 / 1.59 |

- Pairs are car-sized; lifted-only car boxes are not: a box around the
  visible points of a far car (a median of 15) covers one side. Completing the
  shape is what SLF is for.
- Class-less lidar_mos boxes are car-sized too (3% under 0.8 m tall): Step F
  movers no camera masklet covers, kept by design.
- Lifted car boxes up to 3.28 m tall (p90) exceed any car; probably bleed
  within 2 m of the car (a pole, a wall) that the largest group kept. Not yet
  checked frame by frame.

Runtime: 8–21 s per chunk, 2 min for the bag. Viewers on chunk 0005: the 3D
page is 36 MB (every 5th frame), top-down PNGs 2.4 MB (68 frames), camera
overlays 125 MB (every 10th image of 8 cameras).

## Occlusion

Lifted boxes in cluttered scenes, every 5th frame of all 11 chunks (17,043
lifted boxes). Occlusion of a box is measured from `lidar_cc`: the share of
its angular footprint (0.4° × 1° bins with a return) whose nearest return is
more than 0.5 m in front of the box.

**Overlapping boxes are mostly a hidden object boxed on its occluder.** Of
2,406 pairs of fused boxes that overlap in a frame (3D IoU > 0.05 or centres
< 1.5 m), 1,544 (64%) are two lifted instances of one class from the same
camera. On five rendered at < 15 m, each was a car behind another car (two
masks in the image), with the hidden car's box on the front car. Over 873 such
pairs under 30 m, perception_2d's depth map puts the smaller instance's mask a
median 2.1 m behind the other's (p75 15.9 m); its LiDAR points sit at the
other's depth (median 0.0 m, p25–p75 −0.3–0.6 m). The hidden car has a median
of 7–9 points: the occluder blocks the roof LiDAR, and the occluder's edge
returns that project into the hidden car's mask pass the visibility test,
being the nearest surface there. The occlusion metric reads such boxes as
unoccluded (median 0 over all car boxes): they sit on the occluder.

**Sparsity.** On car boxes both sources agree on (fused pairs), the LiDAR
returns inside the box and the share semantic_lifting labelled:

| Range | Pairs | Returns (lidar_cc / lidar_ne) | Labelled share |
|---|---|---|---|
| 0–15 m | 594 | 226 / 3 | 0.68 |
| 15–30 m | 958 | 97 / 2 | 0.43 |
| 30–50 m | 450 | 35 / 0 | 0.33 |

The VLP-16 corner adds almost nothing (this bag has no `lidar_nw`).
Unoccluded lifted car boxes have a median of 119 points under 15 m and 9–13
beyond.

**Rejecting points far nearer than their own mask (not shipped).** Point
camera depth ÷ the median depth-map value over its mask: clean instances
(lifted half of a fused pair, nothing overlapping) p5/p50/p95 0.60 / 0.90 /
1.25; the hidden instances above 0.23 / 0.55 / 0.99. Share of instances whose
median point falls below 0.5 × mask depth, i.e. that such a test would empty:

| Mask depth | Clean | Hidden |
|---|---|---|
| 10–20 m | 0.00 (316) | 0.26 (27) |
| 20–30 m | 0.00 (261) | 0.28 (135) |
| 30–50 m | 0.12 (155) | 0.41 (254) |
| > 50 m | 0.62 (85) | 0.70 (144) |

The depth map's error past ~30 m (it overestimates far masks) makes the test
unsafe there, and under 30 m it removes about a quarter of the hidden boxes.
"Hidden" is itself judged with the depth map, so part of it is depth-map error.
