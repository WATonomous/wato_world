# tracking: measurements behind the defaults

The measurements behind tracking's defaults. The behaviour itself is in the
component README. Scored with `src/tracking/scripts/eval_tracks.py` against
nuScenes ground truth (`gt_boxes.parquet`, imported from the converted bags):
scene-0061 (ego driving; 42 objects with a motion state) and scene-1100 (ego
stationary; 25), each one chunk, proposals from `proposal_generation.yaml`.
Matching is within 2 m at the annotated instants (37 and 38 of them, 2 Hz).
WATO has no ground truth yet; it is scored by the instance-split proxy.

## Association, duplicates and re-linking

The methods and their sources: `docs/research/tracking_improvements.md`.
Scored with TrackEval's association metrics (`eval_tracks.py` "association",
matched within 2 m): AssRe falls when one object is split over several
tracks, AssPr when one track covers several objects; IDSW counts an object
changing track. On WATO: tracks kept, and the lifted instances' mean tracks
each (SAM2's identity as weak ground truth; it misses splits where the new box
carries a new instance, and by construction favours the instance bonus).
Every change alone on the previous defaults (Chen's gates, 2 s coast), then
stacked:

| | 0061 AssRe / AssPr | IDF1 | IDSW | tracks per object | 1100 AssRe / AssPr | IDF1 | IDSW | WATO tracks | per instance |
|---|---|---|---|---|---|---|---|---|---|
| previous defaults | 0.42 / 0.96 | 0.36 | 33 | 1.79 | 0.69 / 0.98 | 0.61 | 15 | 2,757 | 2.39 |
| circle NMS, every CenterPoint head | 0.42 / 0.96 | 0.36 | 28 | 1.69 | 0.68 / 0.97 | 0.62 | 13 | 2,584 | 2.36 |
| GIoU cost | 0.48 / 0.93 | 0.40 | 22 | 1.50 | 0.74 / 0.98 | 0.64 | 10 | 2,039 | 2.24 |
| never terminate | 0.42 / 0.96 | 0.36 | 33 | 1.79 | 0.70 / 0.98 | 0.62 | 14 | 2,732 | 2.42 |
| STWO re-linking | 0.46 / 0.93 | 0.39 | 26 | 1.57 | 0.74 / 0.98 | 0.64 | 11 | 2,372 | 2.31 |
| GIoU + never terminate | 0.48 / 0.93 | 0.40 | 22 | 1.50 | 0.78 / 0.98 | 0.66 | 9 | 1,982 | 2.26 |
| + circle NMS, vehicle heads | 0.48 / 0.92 | 0.40 | 18 | 1.48 | 0.77 / 0.97 | 0.66 | 9 | 1,815 | 2.21 |
| **+ STWO (shipped)** | 0.49 / 0.93 | 0.41 | 17 | 1.43 | 0.77 / 0.97 | 0.66 | 9 | 1,746 | 2.21 |

Shipped against previous: HOTA 0.34 → 0.37 and 0.60 → 0.63; false tracks
46 → 35 and 19 → 11; frame recall unchanged (0.32, 0.60), precision
0.72 → 0.74 and 0.81 → 0.82; parked / moving accuracy 0.78 → 0.80 and 0.88
both. On WATO the median owned span is 1.5 s (0.7 s before), p90 8.2 s
(3.6 s), and 16 tracks last over 30 s (9). Tracking the 11 WATO chunks takes
80 s instead of 30 s, on CPU.

- **GIoU** (SimpleTrack's −0.5 for every class) is the largest single gain: a
  one-sided sliver beside its track's whole box has IoU 0 but a GIoU that
  still ranks it. It costs 0.03 AssPr on scene-0061 (tracks that now cover a
  neighbour at times).
- **Never terminating** does nothing alone: a coasting track's drifted
  prediction still fails Chen's IoU and volume gates. With GIoU it raises
  scene-1100's AssRe 0.74 → 0.78 (ego stationary: objects occluded and seen
  again near where they were).
- **Circle NMS.** CenterPoint's pedestrian head (0.42 m) costs scene-1100
  frame recall 0.60 → 0.58: it removes real pedestrians walking side by side.
  Its truck and bus heads cut scene-0061's switches (22 → 18). Shipped with
  the car, truck and bus heads only, it suppresses 205 of 5,500 fused boxes on
  scene-0061, 71 of 4,867 on scene-1100, and 5,504 of 90,800 (6%) on WATO,
  without changing frame recall. The motorcycle / bicycle head is left out
  untested (too few boxes).
- **STWO** on the previous tracker recovers about half of GIoU's gain. On top
  of GIoU and never terminating, few sequential splits are left for it: it
  fuses 15 of 173 tracks on scene-0061, 5 of 79 on scene-1100, and 2-55 per
  WATO chunk. Its metric (BEV IoU, 3D IoU or GIoU), class gate and
  shared-instance admission change nothing measurable on the two scenes.
- **The instance bonus** on the GIoU cost: 0 or 1 score the same on the
  ground truth (AssRe 0.49 / 0.48, 0.74 / 0.75); kept at 1.0.

### What is left

For each ground-truth object in several tracks, its tracks in time order:
pairs alive at once (concurrent) or one after the other (sequential, by the
gap between them).

| | 0061 split objects | concurrent | sequential (≤ 1 s / 1-2 s / > 2 s) | 1100 split objects | concurrent | sequential |
|---|---|---|---|---|---|---|
| previous defaults | 23 / 42 | 12 | 21 (10 / 5 / 6) | 11 / 25 | 8 | 8 (5 / 0 / 3) |
| GIoU + never terminate + NMS | 13 / 42 | 14 | 6 (2 / 0 / 4) | 11 / 25 | 14 | 1 |
| shipped | 13 / 42 | 13 | 5 (1 / 0 / 4) | 11 / 25 | 15 | 0 |

Sequential splits are gone but for gaps over 2 s, beyond STWO's two 1 s
horizons. The concurrent pairs are long-lived (the shorter track 52 frames at
the median on scene-0061, 71 on scene-1100), 2.2-2.6 m apart at the median
with IoU 0, mostly pedestrians and motorcycles: largely neighbours in a
group, which the eval's 2 m match (all fused scores tie) assigns to one
object at some instants and the other at others. Only the closest tenth
(≤ 0.5 m apart) look like one object's duplicate tracks.

## Chen's gates (cost `chen`)

Measured with a 2 s coast, before GIoU became the default. Step F still uses
these gates.

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
shared-instance association above (Chen's gates; with the shipped
association, 33 / 41 and 22 / 25). Objects whose track has < 7 frames are
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
| tracks kept / short weak dropped | 98 / 60 | 52 / 22 | 1,746 / 1,187 |
| moving / parked / unknown | 48 / 39 / 11 | 29 / 17 / 6 | 1,097 / 351 / 298 |
| frame recall / precision | 0.32 / 0.74 | 0.60 / 0.82 | — |
| fused proposals' recall / precision | 0.32 / 0.71 | 0.60 / 0.81 | — |

63% of WATO's kept tracks read as moving (53% under the previous
association, whose shorter tracks were more often "unknown"). Whether that is
the traffic or one-sided far boxes drifting more than 1 m needs the WATO track
tags. WATO also drops 174 tracks only the non-owning chunk saw. Its tracks are
short: the owned span is 1.5 s at the median and 8.2 s at p90, linked or not,
so on WATO fragmentation inside a chunk matters more than the seams.

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

Measured with Chen's gates and a 2 s coast.

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

Measured with Chen's gates and a 2 s coast.

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

## Appearance

2026-11-12. Both nuScenes GT scenes and scene-0061 as two chunks, on the
2026-11-07 `*_new` arms' masks and proposals (isolated roots, scripts and
results: `data/artifacts/eval_runs/2026-11-12_appearance/`). Two small
scenes, one of them at night (scene-1100): not WATO accuracy.

### Which descriptor

How well a descriptor tells one object's masklets (another camera, or a new
masklet after SAM2 lost it) from those of same-class objects that come
within 3 m of it. AUC over masklet pairs labelled through nuScenes GT (106
same / 258 different pairs on scene-0061, 59 / 51 on scene-1100), 16 images
per masklet, DINOv2 ViT-L/14 throughout
(`descriptor_variants/`):

| Descriptor | Compared by | scene-0061 | scene-1100 |
|---|---|---|---|
| CLS of the bbox crop, Resize(224) + CenterCrop(224) (before) | mean per masklet (before) | 0.65 | 0.66 |
| same | closest pair | 0.73 | 0.78 |
| CLS of the crop resized to a 224 square | mean per masklet | 0.72 | 0.71 |
| same | closest pair | 0.83 | 0.89 |
| patch tokens under the mask, 224 square (FFA; now) | mean per masklet | 0.80 | 0.70 |
| same | closest pair (now) | 0.90 | 0.87 |
| FFA at 336 (the papers' size) | closest pair | 0.90 | 0.89 |
| CLS of the crop with the background zeroed, 336 | closest pair | 0.81 | 0.80 |

- Keeping every image's descriptor and comparing by the closest pair is
  worth more than any change of descriptor (+0.08 to +0.18).
- The centre crop cost about 0.1: it cuts the ends off a wide or tall object.
- FFA and the square crop's CLS are level over the two scenes (FFA's paper
  finds the same split on its own data). FFA is kept because the mask picks
  its tokens: two overlapping masks in one bounding box get different
  descriptors, and the CLS of the crop cannot tell them apart.
- 224 and 336 are level; 224 costs 2.25× fewer tokens.
- Across cameras the descriptors are much weaker at night: against every
  same-class object of the scene, cross-camera pairs score 0.96 on
  scene-0061 and 0.70 on scene-1100, same-camera pairs 1.00 and 0.89.

### The threshold

θ_emb calibrated from the bag's own pairs (`appearance.calibrate`), against
the same equal-error rule on GT-labelled pairs (16 images per masklet, class
groups): 0.135 against 0.151 on scene-0061, 0.153 against 0.150 on
scene-1100. The pipeline's own labels put the threshold where ground truth
does. As tracking runs it (every image, up to 100 per masklet, same class
only):

| | scene-0061 | scene-1100 | scene-0061, two chunks |
|---|---|---|---|
| θ_emb | 0.116 | 0.126 | 0.114 |
| same-object / different-object pairs | 80 / 1,003 | 29 / 251 | — |
| pairs on the wrong side, each kind | 19% / 18% | 31% / 31% | — |

At that threshold a fifth (scene-0061) to a third (scene-1100, night) of
different objects sharing an image look alike, which is why appearance only
counts inside the distance gate.

### What it changes

Current tracker (GIoU cost, never-terminating tracks, circle NMS, STWO),
appearance off against on in each place:

| scene-0061 | off | re-link (default when on) | association | both |
|---|---|---|---|---|
| IDSW | 25 | 25 | 26 | 25 |
| IDF1 | 0.411 | 0.412 | 0.404 | 0.407 |
| AssRe / AssPr | 0.481 / 0.925 | 0.483 / 0.920 | 0.469 / 0.917 | 0.470 / 0.915 |
| tracks per object (in one) | 1.51 (0.65) | 1.51 (0.65) | 1.47 (0.65) | 1.44 (0.67) |
| tracks switching objects | 0.086 | 0.105 | 0.089 | 0.091 |

| scene-1100 | off | re-link | association | both |
|---|---|---|---|---|
| IDSW | 7 | 6 | 15 | 15 |
| IDF1 | 0.667 | 0.675 | 0.667 | 0.667 |
| AssRe / AssPr | 0.780 / 0.974 | 0.791 / 0.974 | 0.781 / 0.953 | 0.781 / 0.953 |
| tracks per object (in one) | 1.42 (0.62) | 1.38 (0.65) | 1.58 (0.58) | 1.58 (0.58) |
| tracks switching objects | 0.061 | 0.062 | 0.121 | 0.121 |

| scene-0061, two chunks | off | re-link | association | both |
|---|---|---|---|---|
| IDSW | 25 | 24 | 24 | 23 |
| IDF1 | 0.415 | 0.416 | 0.413 | 0.418 |
| AssRe / AssPr | 0.485 / 0.924 | 0.485 / 0.924 | 0.491 / 0.919 | 0.492 / 0.915 |
| seam crossings kept in one track | 11 / 13 | 11 / 13 | 11 / 13 | 11 / 13 |

Frame recall and precision don't move (appearance changes identities, not
boxes). In the re-link it is within noise of off; in the per-frame
association it doubles scene-1100's switches, by handing a track a
look-alike neighbour inside the 2 m gate. Hence off by default, and
association off when on.

### Why it changes so little

Each ID switch of the appearance-off run, by whether the old and the new
track ever exist at once, and what their masklets say (`switch_diag.py`):

| | scene-0061 | scene-1100 |
|---|---|---|
| ID switches | 25 | 7 |
| between two tracks alive at once | 19 | 7 |
| … that share a masklet / look alike / look different | 6 / 6 / 7 | 2 / 5 / 0 |
| one track after the other | 6 | 0 |
| … that share a masklet / look alike / look different | 2 / 2 / 2 | — |

Re-identification joins a track that ended to one that starts later. Three
quarters of scene-0061's switches and all of scene-1100's are instead
between two tracks running side by side on one object, which neither the
association nor the re-link may fuse (their lives overlap). In 19 of those
26 the two tracks share a masklet or look alike: the evidence that they are
one object is there, but the step that would use it (fusing concurrent
duplicate tracks) does not exist. The four sequential switches appearance
or a shared masklet could join have gaps of 1.3-2.9 s and 1.6-3.4 m, outside
the re-link's 1 s horizon or its 2 m gate.
