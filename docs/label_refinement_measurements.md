# label_refinement: measurements behind the defaults

The measurements behind label_refinement's defaults. The behaviour itself is in
the component README. Scored with `src/label_refinement/scripts/eval_refined.py`
against nuScenes ground truth: scene-0061 (ego driving) and scene-1100 (ego
stationary), tracks from tracking's defaults. Owned frames only, matched within
2 m at the annotated instants. "Tracked" is the track's box as proposed;
errors are medians on matched boxes, prediction − ground truth.

## Before and after

Tracks from the current defaults (proposal_generation's fragment filter
included). Phase 3 is the training-free refinement (`slf.enabled: false`);
SLF is the shipped default.

| scene-0061 | recall | precision | IoU | centre | Δl | Δw | Δh | Δz |
|---|---|---|---|---|---|---|---|---|
| tracked, all | 0.32 | 0.72 | 0.12 | 0.28 | −0.51 | −0.51 | −0.30 | −0.29 |
| Phase 3, all | 0.35 | 0.77 | 0.17 | 0.25 | −0.31 | −0.40 | −0.18 | −0.12 |
| SLF, all | 0.34 | 0.77 | 0.34 | 0.28 | −0.22 | −0.16 | −0.09 | −0.18 |
| tracked, 30-50 m | 0.08 | 0.31 | 0.13 | 0.65 | −2.11 | −0.93 | −0.22 | −0.46 |
| Phase 3, 30-50 m | 0.10 | 0.42 | 0.26 | 0.50 | −0.68 | −0.21 | −0.15 | −0.13 |
| SLF, 30-50 m | 0.10 | 0.42 | 0.38 | 0.27 | −1.11 | −0.27 | +0.07 | −0.33 |
| tracked, car | 0.52 | – | 0.19 | 0.80 | −0.61 | −0.36 | −0.28 | −0.28 |
| Phase 3, car | 0.64 | – | 0.34 | 0.61 | −0.07 | −0.22 | −0.06 | −0.08 |
| SLF, car | 0.64 | – | 0.35 | 0.49 | −0.07 | −0.22 | −0.06 | −0.06 |
| tracked, pedestrian | 0.25 | – | 0.07 | 0.18 | −0.37 | −0.52 | −0.34 | −0.24 |
| Phase 3, pedestrian | 0.25 | – | 0.11 | 0.17 | −0.32 | −0.44 | −0.34 | −0.23 |
| SLF, pedestrian | 0.25 | – | 0.31 | 0.18 | −0.15 | −0.10 | −0.09 | −0.15 |
| tracked, truck | 0.53 | – | 0.38 | 0.55 | −0.76 | −0.56 | −0.21 | −0.76 |
| Phase 3, truck | 0.60 | – | 0.26 | 0.43 | −0.22 | +0.25 | −0.10 | −0.09 |
| SLF, truck | 0.60 | – | 0.44 | 0.43 | −0.89 | −0.40 | −0.10 | −0.63 |

| scene-1100 | recall | precision | IoU | centre | Δl | Δw | Δh | Δz |
|---|---|---|---|---|---|---|---|---|
| tracked, all | 0.60 | 0.81 | 0.15 | 0.23 | −0.33 | −0.37 | −0.19 | −0.41 |
| Phase 3, all | 0.60 | 0.83 | 0.21 | 0.23 | −0.26 | −0.32 | −0.13 | −0.38 |
| SLF, all | 0.60 | 0.83 | 0.27 | 0.27 | +0.09 | −0.04 | −0.12 | −0.38 |
| tracked, car | 0.70 | – | 0.25 | 1.00 | −1.81 | −0.23 | −0.29 | −0.04 |
| Phase 3, car | 0.72 | – | 0.29 | 0.58 | −0.86 | −0.07 | +0.06 | +0.24 |
| SLF, car | 0.72 | – | 0.29 | 0.52 | −0.86 | −0.18 | −0.07 | +0.24 |
| tracked, pedestrian | 0.61 | – | 0.12 | 0.11 | −0.12 | −0.39 | −0.17 | −0.49 |
| Phase 3, pedestrian | 0.62 | – | 0.13 | 0.11 | +0.04 | −0.38 | −0.14 | −0.49 |
| SLF, pedestrian | 0.62 | – | 0.24 | 0.22 | +0.22 | +0.07 | −0.14 | −0.49 |

Phase 3 fixes cars' length (−0.6 / −1.8 m to −0.1 / −0.9 m). SLF then gives
pedestrians their width (Δw −0.44 → −0.10 m, IoU 0.11 → 0.31 on scene-0061,
0.13 → 0.24 on scene-1100) and trucks their extent (0.26 → 0.44), and
recentres far boxes (30-50 m centre error 0.50 → 0.27 m). Recall and precision
don't move: SLF reshapes boxes, it doesn't add or drop any. Cars barely change
on these scenes: most are near enough that Phase 3's box already matches their
masks, and the gate leaves those alone. 38 trucks' length error goes the wrong
way while their IoU rises: a few long trucks dominate the median.

## SLF

Per-track comparison: the boxes with and without the track's fit, against the
same ground-truth box at each labelled instant (the one nearest the Phase 3
box), mean IoU over the matched instants, every fit kept (no gate):

| class (both scenes) | tracks | Phase 3 | SLF, every fit | better / worse (±0.02) |
|---|---|---|---|---|
| pedestrian | 63 | 0.13 | 0.25 | 56 / 2 |
| car | 32 | 0.33 | 0.30 | 9 / 16 |
| truck | 17 | 0.31 | 0.34 | 7 / 8 |
| bus | 1 | 0.08 | 0.28 | 1 / 0 |
| bicycle, motorcycle | 8 | 0.21 | 0.24 | 2 / 3 |

With the shipped gate: pedestrians 0.13 → 0.25 (57 fits kept), vehicles
0.32 → 0.34 (9 kept), two-wheelers unchanged (1 kept).

- **The gate.** A vehicle whose Phase 3 box was one-sided (2.2 × 0.9 m,
  1.4 × 0.1 m) grows to car size and gains most (IoU 0.18 → 0.59,
  0.01 → 0.47); its silhouette residual falls by 0.3-0.6. One whose box was
  already full-size is mostly shrunk to masks drawn tight (height 2.1 → 1.7 m,
  width 2.1 → 1.5 m) and loses; its residual moves by under 0.1. Keeping a
  vehicle's fit only when the residual falls by ≥ 0.15 takes vehicles from
  0.323 to 0.344 (9 of 50 kept; 0.329 at a 0.25 m point gate, which dropped a
  truck that gained 0.19); 0.1 does the same with 12 kept.
- **Pedestrians keep every fit.** Their silhouette residual barely moves
  (median gain −0.03), yet 56 of 63 gain: the boxes grow from 0.48 × 0.22 m
  (a sliver of returns) to 0.74 × 0.64 m. Without the mask term (points and
  prior only; measured before the two placement fixes below) pedestrians
  reached 0.249 on scene-0061 but dropped to 0.109 on scene-1100 (0.118
  before), against 0.263 and 0.140 with it: the prior does most of the work
  on the driving scene, the masks keep the stationary one from going wrong.
- **A pedestrian's box keeps its observed centre and encloses its returns.**
  Grown from its near corner, as a vehicle's, it landed 0.28 m too far from
  the ego radially on scene-1100 and 0.19 m sideways (the observed box: 0.01
  and 0.05 m), and pedestrian IoU there stayed 0.13. Centred, then with
  returns only needing to be inside the box, it reaches 0.24 (centre error
  0.22 m): the returns are on the person's front, inside the ground truth's
  box, not on its faces.
- **Synthetic check** (`tests/test_slf.py`): a 4.5 × 1.8 × 1.5 m car seen by
  the LiDAR only on its rear face and by four cameras comes out 4.44 × 1.98 ×
  1.61 m, silhouette residual 0.54 → 0.09.
- **Runtime.** One track at a time, a fit took 0.64 s and building its views
  0.14 s (RTX 4090, 16 views of 48 × 48 rays, 32 samples per ray, 150 steps):
  the GPU sat at 15%, each step's cost being its launches. Fitting 32 tracks
  at once (`batch_tracks`), each on its own loss, gives the same boxes (same
  fits kept, same scores) in 48 s instead of 110 s on scene-0061 (133 fits)
  and 28 s instead of 67 s on scene-1100 (71); building the views is most of
  what is left.

## Heading

`eval_refined.py`'s `yaw`: |heading − ground truth's| mod π, median / p90 in
degrees, on the cached tracks of both scenes (of 2026-11-03); vehicles split
by the ground truth's motion state. "Fixed" is SLF with Phase 3's heading (the previous
behaviour), "fit" the heading correction (`fit_yaw_classes`), "+ gate" with
`heading_speed_z: 3` too (the shipped defaults).

| scene-0061 | GT boxes | tracked | fixed | fit | fit + gate |
|---|---|---|---|---|---|
| parked vehicles | 190 | 3.6 / 25.0 | 14.4 / 65.9 | 1.3 / 65.9 | 1.3 / 57.8 |
| moving vehicles | 89 | 22.3 / 86.4 | 3.2 / 14.2 | 3.4 / 12.9 | 3.4 / 12.9 |
| truck | 113 | 3.5 / 19.1 | 14.4 / 36.5 | 1.4 / 36.5 | 0.8 / 16.8 |
| car | 160 | 15.8 / 55.5 | 3.1 / 35.8 | 3.1 / 35.8 | 2.9 / 24.5 |

| scene-1100 | GT boxes | tracked | fixed | fit | fit + gate |
|---|---|---|---|---|---|
| parked vehicles | 76 | 4.7 / 33.3 | 2.4 / 2.4 | 3.2 / 3.2 | 3.2 / 3.2 |
| moving vehicles | 158 | 11.2 / 46.1 | 3.2 / 14.8 | 3.2 / 14.8 | 4.0 / 14.8 |
| car | 196 | 6.9 / 47.3 | 2.4 / 14.8 | 3.2 / 14.8 | 3.2 / 14.8 |

- **A fixed heading made the size absorb its error.** Scene-0061's parked
  truck (`0000-54`, 25 labelled instants) was boxed 14.4° off from its
  accumulated points; the fit turns it to 0.3°, truck IoU 0.44 → 0.50 and
  centre error 0.43 → 0.35 m. On the synthetic car of `tests/test_slf.py`, a
  10° heading error left fixed fits 4.06 × 2.39 m (silhouette residual
  0.126); fitted, the heading returns to 0.7° and the box to 4.47 × 1.97 m
  (0.088), and a correct start stays within 0.2°. Moving vehicles' centre
  error on scene-0061 fell 0.63 → 0.50 m. Scene-1100's one parked car moved
  2.4° → 3.2°.
- **One angle per track** (`yaw_knot_spacing_s: null`) did about as well:
  scene-0061 cars 3.9° and movers 3.6° against 3.1° and 3.4° with 2 s knots,
  scene-1100 identical.
- **Stepping and keeping the best step.** At Adam's fixed rate the heading
  and size kept moving: started at the correct heading, the last step was 4°
  off and its loss 13% above the best. Each track keeps its lowest-loss step,
  and which end of the box is nearer the ego stays Phase 3's (letting it
  flip as the heading crossed zero jumped the box sideways).
- **The gate.** Parked vehicles that tracking called moving took velocity
  headings 25-80° off: their box centres jitter as their visible faces
  change, path length 15-50× net displacement, local speeds 0.5-1.9 m/s.
  Requiring the speed to be ≥ 3 standard errors of the local fit's velocity
  cut scene-0061's truck p90 36.5 → 16.8° and car p90 35.8 → 24.5°, and cost
  scene-1100's moving vehicles 0.8° at the median (a car crawling at
  0.26 m/s median over 19 s). At 2 standard errors the tails didn't move.
- **What is left.** The parked-vehicle p90 is one parked car tracked as
  moving (`0000-154`) whose heading is 67° off: undirected, it is the mean of
  its denser frames' headings mod π, and those mix boxes whose length and
  width are swapped (a one-face view's minimum-area fit), which a mod-π mean
  can't undo. Its SLF fit isn't kept, and its error exceeds the 30° cap. A
  pedestrian's heading isn't fitted (its median error is 38-69°: its
  footprint is near-square).

These are 12-19 matched vehicles per scene.

**Where the heading errors came from** (diagnosed 2026-11-05, before the fit):

- *The minimum-area rectangle fits the observed envelope.* An L-shaped view
  of a 4.5 × 1.8 m car (corners (−2.25, −0.9), (2.25, −0.9), (−2.25, 0.9))
  fits at 0°; one more return at (2.45, −0.5), 0.45 m from the edge, turns
  it to −16.6°. MODEST's edge-closeness criterion (its released code, 0.1°
  steps) keeps 0° there.
- *But the criterion wasn't the parked truck's problem; its points were.* On
  `0000-54`'s accumulated cloud (25,172 points over 264 frames, after the
  region, voxel and largest-group selection) the minimum-area fit is 14.43°
  off, MODEST closeness 14.42°, LISO's 5° closeness 13.32°, and minimum area
  on the cloud cropped to the ground-truth box (15,517 points) 1.12°: other
  structure connected to the truck survives the largest-group filter. The
  silhouette fit corrects it without touching the points.
- *A velocity heading follows the visible centre.* A car moving straight at
  1 m/s whose observed box centre also drifts 0.3 m/s sideways (its visible
  faces changing) gets a 16.7° heading from `refine_moving`'s local fit; a
  longer window can't remove a systematic drift.
- On the cached WATO bag, all 44,700 directed rows (of 62,234 owned
  car / truck / bus rows) are velocity headings; WATO has no heading ground
  truth.

Not done: better parked support (lifted instances' own points rather than
everything in the region), closeness seeds, LISO-style trajectory smoothing
before the velocity heading. Sources and the papers' actual algorithms:
`docs/research/box_orientation_reference_methods.md`.

## Choices

- **Small classes take the median size and aren't aggregated.** With every
  class at the 90th percentile and every parked track aggregated, pedestrian
  boxes came out too long (Δl +0.76 m on scene-0061, +1.34 m on scene-1100,
  IoU 0.07 → 0.16 and 0.12 → 0.05, centre error 0.11 → 0.69 m on scene-1100):
  a standing person still moves, a small object's region collects its
  neighbours, and its high-percentile frames are the ones with most bleed.
- **The parked region ignores stray boxes.** Enclosing every box of a track,
  one parked truck's region took in its neighbour and came out 7.2 × 7.1 m
  (ground truth 6.5 × 2.5; truck IoU 0.17). Only boxes within 1 m of the
  median centre now shape the region: truck IoU 0.26. Still below the tracked
  boxes' 0.38, on 3-4 trucks; recall 0.53 → 0.61.

## WATO

All 11 chunks, `label_refinement.wato.yaml`, on tracks linked across chunks:
2,669 tracks (427 spanning two or more chunks), 81,577 boxes; 330 parked
vehicles boxed from their accumulated points (482 before linking, when a
parked car across a seam was two tracks with half the points each). SLF fit
2,423 tracks in 742 s (0.3 s each, building the views most of it) and kept
374. No ground truth yet.

| WATO, SLF | fits | kept | silhouette residual, kept / not | kept l / w / h, Phase 3 → SLF |
|---|---|---|---|---|
| car | 2,040 | 184 | 0.53 / 0.73 | 3.04 / 1.16 / 1.69 → 4.34 / 1.56 / 2.24 m |
| pedestrian | 229 | 179 | 0.66 / 0.85 | 1.17 / 0.32 / 1.52 → 0.90 / 0.72 / 1.78 m |
| truck | 108 | 7 | 0.44 / 0.69 | 4.04 / 2.26 / 2.60 → 6.88 / 1.61 / 2.24 m |
| bus | 32 | 4 | 0.46 / 0.67 | 3.48 / 1.65 / 5.72 → 11.5 / 1.47 / 5.34 m |

The pedestrians come out person-sized. The cars come out car-long and -wide
but 2.24 m tall at the median, too tall for most cars, and WATO's car boxes
explain their masks far worse than nuScenes' (residual 0.73 before the fit,
about 0.45 on nuScenes): the camera and the LiDAR don't agree on this rig.
Of a car's returns inside its observed box, projected into its own masklet's
image at the image's pose (150-400 views per bag):

| | returns inside the mask | returns' centroid − mask's, share of its width |
|---|---|---|
| nuScenes scene-0061 | 0.85 | −0.01 |
| nuScenes scene-1100 | 0.96 | 0.00 |
| WATO | 0.70 | +0.08 |

| WATO camera | views | inside | offset | range |
|---|---|---|---|---|
| PANO_NN | 130 | 0.73 | +0.11 | 9 m |
| PANO_SS | 122 | 0.77 | +0.05 | 9 m |
| PANO_NE | 50 | 0.62 | −0.08 | 53 m |
| PANO_SW | 36 | 0.83 | +0.05 | 27 m |
| PANO_EE | 27 | 0.56 | +0.14 | 82 m |
| PANO_NW | 27 | 0.61 | +0.10 | 16 m |
| PANO_WW | 7 | 0.64 | +0.17 | 26 m |

The shift differs by camera, sign included, which points at each camera's
extrinsic (or its image timestamps) rather than at SLF; it costs semantic
lifting too. Until it is fixed, WATO's SLF car boxes are suspect; the WATO
labels will say by how much.
