# label_refinement: measurements behind the defaults

The measurements behind label_refinement's defaults. The behaviour itself is in
the component README. Scored with `src/label_refinement/scripts/eval_refined.py`
against nuScenes ground truth: scene-0061 (ego driving) and scene-1100 (ego
stationary), tracks from tracking's defaults. Owned frames only, matched within
2 m at the annotated instants. "Tracked" is the track's box as proposed;
errors are medians on matched boxes, prediction − ground truth.

## Before and after

| scene-0061 | recall | precision | IoU | centre | Δl | Δw | Δh | Δz |
|---|---|---|---|---|---|---|---|---|
| tracked, all | 0.32 | 0.72 | 0.12 | 0.28 | −0.51 | −0.51 | −0.30 | −0.29 |
| refined, all | 0.34 | 0.77 | 0.17 | 0.28 | −0.31 | −0.40 | −0.18 | −0.11 |
| tracked, car | 0.52 | – | 0.19 | 0.80 | −0.61 | −0.36 | −0.28 | −0.28 |
| refined, car | 0.64 | – | 0.34 | 0.61 | −0.07 | −0.22 | −0.06 | −0.08 |
| tracked, pedestrian | 0.25 | – | 0.07 | 0.18 | −0.37 | −0.52 | −0.34 | −0.24 |
| refined, pedestrian | 0.25 | – | 0.11 | 0.18 | −0.34 | −0.44 | −0.33 | −0.22 |
| tracked, truck | 0.53 | – | 0.38 | 0.55 | −0.76 | −0.56 | −0.21 | −0.76 |
| refined, truck | 0.61 | – | 0.26 | 0.43 | −0.22 | +0.25 | −0.10 | −0.09 |

| scene-1100 | recall | precision | IoU | centre | Δl | Δw | Δh | Δz |
|---|---|---|---|---|---|---|---|---|
| tracked, all | 0.60 | 0.73 | 0.14 | 0.23 | −0.33 | −0.38 | −0.21 | −0.43 |
| refined, all | 0.60 | 0.74 | 0.16 | 0.28 | −0.26 | −0.35 | −0.13 | −0.38 |
| tracked, car | 0.70 | – | 0.24 | 1.00 | −1.81 | −0.28 | −0.35 | −0.04 |
| refined, car | 0.72 | – | 0.29 | 0.58 | −0.86 | −0.11 | 0.00 | +0.24 |
| tracked, pedestrian | 0.61 | – | 0.12 | 0.11 | −0.12 | −0.39 | −0.17 | −0.49 |
| refined, pedestrian | 0.62 | – | 0.13 | 0.13 | +0.04 | −0.38 | −0.14 | −0.49 |

Cars gain most: their length error goes from −0.6 / −1.8 m to −0.1 / −0.9 m.
Pedestrians stay sliver-thin (Δw ≈ −0.4 m): no frame sees their width, and
nothing here completes it.

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

All 11 chunks, `label_refinement.wato.yaml`: 3,154 tracks, 82,002 boxes; 482
parked vehicles boxed from their accumulated points. No ground truth yet.
