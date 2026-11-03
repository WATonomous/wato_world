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

| | scene-0061 | scene-1100 | WATO (11 chunks) |
|---|---|---|---|
| tracks kept / short weak dropped | 142 / 107 | 76 / 48 | 3,154 / 3,268 |
| moving / parked / unknown | 55 / 56 / 31 | 37 / 28 / 11 | 1,675 / 546 / 933 |
| frame recall / precision | 0.32 / 0.72 | 0.60 / 0.73 | — |
| fused proposals' recall / precision | 0.33 / 0.70 | 0.60 / 0.72 | — |

Half of WATO's tracks read as moving. Whether that is the traffic or one-sided
far boxes drifting more than 1 m needs the WATO track tags.
