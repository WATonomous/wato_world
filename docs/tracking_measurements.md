# tracking: measurements behind the defaults

The measurements behind tracking's defaults. The behaviour itself is in the
component README. Scored with `src/tracking/scripts/eval_tracks.py` against
nuScenes ground truth (`gt_boxes.parquet`, imported from the converted bags):
scene-0061 (ego driving; 42 objects with a motion state) and scene-1100 (ego
stationary; 25), each one chunk, proposals from `proposal_generation.yaml`.
Matching is within 2 m at the annotated instants. WATO has no ground truth yet.

## Association

A track is *true* when most of its matched instants hit one object, *false*
when most hit nothing. Purity is the share of a track's matches on its main
object; "switched" counts tracks under 0.9.

| scene-0061 | frame R / P | true / false tracks | tracks per object (in one) | switched |
|---|---|---|---|---|
| Chen's gates (IoU ≥ 0.05, volume ratio ≥ 0.3) | 0.30 / 0.74 | 91 / 58 | 2.59 (0.33) | 0.02 |
| no volume gate | 0.31 / 0.72 | 70 / 54 | 2.32 (0.37) | 0.15 |
| no shape gates | 0.32 / 0.71 | 47 / 45 | 1.50 (0.69) | 0.09 |
| **shared lifted instance opens the shape gates** | 0.32 / 0.72 | 61 / 46 | 1.79 (0.45) | 0.06 |
| same, instance bonus 3 | 0.32 / 0.72 | 64 / 46 | 1.83 (0.45) | 0.04 |

| scene-1100 | frame R / P | true / false | tracks per object (in one) | switched |
|---|---|---|---|---|
| Chen's gates | 0.58 / 0.74 | 56 / 28 | 2.64 (0.32) | 0.03 |
| no shape gates | 0.60 / 0.73 | 32 / 21 | 1.58 (0.62) | 0.11 |
| **shared instance opens the shape gates** | 0.60 / 0.73 | 38 / 21 | 1.68 (0.52) | 0.00 |
| same, instance bonus 3 | 0.60 / 0.73 | 41 / 21 | 1.80 (0.44) | 0.00 |

Chen's IoU and volume gates split tracks of one-sided and thin boxes (a 0.3 m
shift between two pedestrian slivers is IoU 0); dropping them halves the
fragmentation but lets tracks jump between neighbours. Letting SAM2's identity
(a shared lifted instance) open them keeps most of the gain without the jumps.
"True / false" counts shrink as fragments merge; compare false counts.

## Motion state

Per object, its dominant track's state against the ground truth's (3DAL's
rule over the object's life: parked iff < 1 m net and never > 1 m/s), with the
shared-instance association above. Objects whose track has < 7 frames are
"unknown" under every rule and left out (1 of 67).

| Rule (moving iff …) | scene-0061 | scene-1100 | both |
|---|---|---|---|
| MS3D: net ≥ 1 m (pedestrian 2 m) or centre variance ≥ 0.1 m² | 28 / 41 | 22 / 25 | 0.76 |
| net ≥ 0.5 m at ≥ 0.5 m/s | 35 / 41 | 19 / 25 | 0.82 |
| net ≥ 1 m | 32 / 41 | 22 / 25 | 0.82 |
| **net ≥ 1 m, or ≥ 0.5 m at ≥ 0.5 m/s** | 32 / 41 | 22 / 25 | 0.82 |

- MS3D's variance term reads parked cars and trucks as moving: a one-sided
  box's centre shifts as its visible face changes (variance 0.15-0.26 m² on
  parked vehicles tracked for 5-6 s).
- Its 2 m pedestrian threshold misses walkers on fragments: 0.6 m in 0.4 s.
- The speed rule alone misses objects that crept (a car at a light, 1.26 m in
  19 s, 0.06 m/s), which the ground truth calls moving.
- The shipped rule ties for best on 66 objects and is the most even across the
  two scenes. Remaining errors on scene-0061: 6 parked objects read as moving,
  3 movers as parked.

## Defaults on all three bags

| | scene-0061 | scene-1100 | WATO (11 chunks, linked) |
|---|---|---|---|
| tracks kept / short weak dropped | 142 / 104 | 73 / 49 | 2,669 / 2,906 |
| moving / parked / unknown | 55 / 56 / 31 | 37 / 25 / 11 | 1,409 / 458 / 802 |
| frame recall / precision | 0.32 / 0.72 | 0.60 / 0.81 | — |
| fused proposals' recall / precision | 0.32 / 0.71 | 0.60 / 0.81 | — |

Half of WATO's tracks read as moving. Whether that is the traffic or one-sided
far boxes drifting more than 1 m needs the WATO track tags. WATO also drops
421 tracks only the non-owning chunk saw. Its tracks are short: the owned span
is 0.7 s at the median and 3.7 s at p90, linked or not, so on WATO
fragmentation inside a chunk matters more than the seams.

## Linking across chunks

### What a shared span looks like in both chunks

The WATO bag, 10 seams, 6 s shared span each. 22 shared sweeps per seam
sampled: every one has the same point count and identical world points in
both chunks (max |Δxyz| 0), so data matched across chunks by
`(lidar_id, reference_timestamp_ns)` is the same data, point for point.

The proposals on those sweeps mostly agree. Per source, the share of chunk k's
rows at shared instants with a same-source row of chunk k+1 within 5 cm
(identical) / within 1 m:

| source | rows | ≤ 5 cm | ≤ 1 m | per-seam range ≤ 1 m |
|---|---|---|---|---|
| lifted | 11,941 | 0.76 | 0.94 | 0.85-0.99 |
| lidar_mos | 3,603 | 0.41 | 0.62 | 0.24-0.86 |
| fused | 14,026 | 0.66 | 0.88 | 0.69-0.91 |

Step F differs most: it runs on each chunk's own accumulated evidence, so its
tracks, motion scores and gates differ near a window edge. Of the 520 kept
per-chunk tracks with frames in a span, 464 (0.89) agree with a track of the
next chunk at ≥ half the shorter one's span instants (425 at ≥ 0.8); 44 share
no instant with any.

### Linking

A seam break here is a track whose owned frames end within 0.5 s before a seam
and one whose owned frames start within 0.5 s after it, ≤ 1.5 m apart: one
object, two ids (no ground truth needed).

| WATO | crossings kept in one track | breaks |
|---|---|---|
| per-chunk tracks | 0 | 117 |
| linked, cost 1 − share of the shorter track's instants | 114 | 10 |
| **linked, most instants agreed on** | 124 | 5 |

The first cost let a short fragment that agreed at all of its few instants
take the link from the long track of the same object (a bus agreeing at 99 of
100 instants stayed unlinked). Of the 5 breaks left, three are a second track
one chunk kept on the same object at the same time, one is a break inside a
chunk just before the seam, and one is a 2-frame fragment, under the
3-instant floor. Linking made 427 tracks span chunks; 8 now last over
30 s (none could before), and 401 parked vehicles are boxed from one cloud
instead of 482 from half clouds (`docs/label_refinement_measurements.md`).

On nuScenes (one chunk each) nothing changes: every number above the WATO
column is the same before and after.

### A seam against ground truth

scene-0061 ingested again as two chunks (`ingest.yaml` with `chunk_seconds:
12`, `--bag-id NuScenes_v1_0_mini_scene_0061_2chunk`; shared span 8-14 s, seam
at 11 s) and run through every stage with the default configs, against the
same scene as one chunk:

| scene-0061 | one chunk | two chunks, linked |
|---|---|---|
| frame recall / precision | 0.32 / 0.72 | 0.32 / 0.73 |
| true / false tracks | 61 / 46 | 65 / 43 |
| tracks per object (in one) | 1.79 (0.45) | 1.83 (0.40) |
| objects on both sides of 11 s kept in one track | 4 / 13 | 5 / 13 |
| parked / moving accuracy | 0.78 | 0.85 |
| refined IoU, all (car) | 0.17 (0.34) | 0.18 (0.37) |

Linked, the seam is no worse than the same instant inside one chunk. The
continuity there is low either way: objects fragment inside a chunk too
(1.8 tracks per object). The motion-state gain is 3 parked objects on 42, not
a measured effect of linking.

The midpoint rule, checked on the same span: each chunk's own track boxes,
owned or not, scored per half. Each chunk is about as good near its data
edge as where it owns the instants (recall 0.41 / 0.40 before 11 s, 0.50 /
0.50 after; IoU 0.11-0.13 throughout). Edge effects are small on nuScenes; the
rule's job here is labelling each instant once.

### Duplicate tracks

The second track on one object is a precision cost too: at the labelled
instants, 59 of scene-1100's 138 false-positive track frames (0.43) sat within
2 m of an object another track already matched, 5 of 133 on scene-0061. Most
were lifted slivers (0.2 × 0.1 m "cars" beside a car's box): a small mask
beside a big one claiming a few of its edge returns. proposal_generation now
drops those fragments (`docs/proposal_generation_measurements.md`,
"Fragments"): 8 duplicates left on scene-1100, 5 on scene-0061, mostly
pedestrian boxes 0.5-1.3 m apart, which may also be two people walking
together.
