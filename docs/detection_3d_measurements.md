# detection_3d measurements

BEVFusion (the car's `resnet50_trt11_int8`, TensorRT 10.16, RTX 4090) run
offline by detection_3d. Measured 2026-11-09 on branch `bzheng/bevfusion`.

**Caveat on nuScenes.** Both nuScenes scenes this repo has, 0061 and 1100,
are in nuScenes' `train` split (nuscenes-devkit `splits.py`), which the
checkpoint was trained on. Their scores check the adapter and bound what the
model can do; they are not held-out accuracy. The held-out mini scenes are
0103 and 0916 (`mini_val`), not yet converted. WATO has no ground truth yet.

## The box conventions, against nuScenes ground truth

Scene-0061, 37 labelled instants, boxes of the six pipeline classes matched
class-agnostically within 2 m (`wato_common.eval.boxes`):

| Conversion | median dz | median 3D IoU | directed heading error |
|---|---|---|---|
| vendor z as the box bottom (`+ h/2`) | +0.77 m | 0.25 | 4.8° median, 6.6% > 90° |
| vendor z as the box centre (shipped) | −0.17 m | 0.40 | same |

dz tracked h/2 per class as the bottom (cars +0.75, trucks +1.3), and the
detector's "bottom" sat a median 1.08 m above Patchwork++ ground under cars:
the vendor z is the centre. Headings use front = `-rot - pi/2`
(mmdetection3d v0.x); 6.6% flipped boxes is the model's front/back confusion,
not a convention error (a wrong convention flips all of them).

## Scene-0061 detection quality (training scene: an upper bound)

Input frame `sensor` (LIDAR_TOP), 10 sweeps, nearest images within 200 ms; all
380 frames run.

| min score | recall | precision | centre err | dl / dw / dh | IoU |
|---|---|---|---|---|---|
| 0.1 | 0.92 | 0.84 | 0.18 m | −0.03 / +0.02 / −0.02 m | 0.40 |
| 0.3 | 0.77 | 1.00 | 0.18 m | −0.04 / +0.02 / −0.02 m | 0.42 |
| 0.5 | 0.52 | 1.00 | 0.16 m | −0.03 / +0.03 / −0.03 m | 0.45 |

By range at 0.1: 0.99 recall to 15 m, 0.96 at 15–30 m, 0.93 at 30–50 m, 0.37
beyond 50 m (the model's grid ends at 54 m). Per class at 0.1: car 0.89,
truck 0.88, pedestrian 0.93, motorcycle 0.83, bicycle 0.75, bus 2 of 6.

The rebuilt `nuscenes_lidar` frame (what WATO uses) scores the same on this
scene as LIDAR_TOP itself: recall 0.91 / precision 0.85 against 0.92 / 0.84.

Frames: with only frame_index's pairs (≤ 50 ms), 60 of 380 sweeps lacked an
image from some 12 Hz camera; the nearest image within the car's 200 ms
sync tolerance runs all 380.

Runtime: 25 s for a 380-frame chunk (~65 ms per frame, image decoding
included); engines build once in 84 s. Two runs on one frame differ by ~1 cm
(TensorRT int8 is not bit-exact).

## WATO: offline inputs against the car's (no ground truth)

`sensors_2026_07_07-23_01_24_2`, chunk 0003 (674 frames). Proxies only: boxes
scored ≥ 0.3 of the pipeline classes, outside the 2.5 m ego zone; "lifted
objects" are fused proposals with a lifted member and ≥ 20 points within
40 m, every 5th frame.

| Variant | boxes / frame | lifted objects with a box within 2 m | boxes near a lifted object | boxes with ≥ 5 points |
|---|---|---|---|---|
| A: shipped (nuscenes_lidar frame, 0.5 s of sweeps, image band) | 2.3 | 0.46 | 0.51 | 0.94 |
| B: A in `lidar_cc`'s own frame (the car's) | 1.8 | 0.40 | 0.55 | 0.92 |
| C: A with one sweep (the car's) | 1.1 | 0.32 | 0.68 | 0.96 |
| D: A plus `lidar_ne` points | 0.5 | 0.04 | 0.35 | 0.86 |

The training frame and the earlier sweeps each find more of the objects
lifting finds; the boxes that don't match a lifted object still hold LiDAR
points (lifting misses objects too, so they are not all false positives).
Adding the VLP-16 corner LiDAR collapses detection; it stays out.

Overlays on PANO_SS / PANO_NN: boxes sit on the vehicles; on PANO_NN they
project below them by the same offset as the pipeline's own boxes, WATO's known
front-camera calibration error rather than the detector's.

## Proposals and tracks, with and without the detector (nuScenes)

Both scenes re-run from perception_2d in a scratch artifact root (their
perception_2d / semantic_lifting outputs predated independent masks and
`lifted_candidates`), current code, default configs. Three proposal_generation
variants, then tracking and its evaluator; matched within 2 m, class-agnostic.
Training scenes: an upper bound, not held-out accuracy.

**Proposals** (fused rows; `eval_proposals.py`):

| Scene | Variant | recall | precision | IoU | dl / dw | 0-15 / 15-30 / 30-50 m recall |
|---|---|---|---|---|---|---|
| 0061 | base (detector off) | 0.34 | 0.71 | 0.11 | −0.51 / −0.51 | 0.72 / 0.38 / 0.08 |
| 0061 | detector reshapes matched objects (`add_unmatched: false`) | 0.38 | 0.78 | 0.46 | −0.02 / +0.01 | 0.78 / 0.41 / 0.14 |
| 0061 | + detector-only objects (`add_unmatched: true`) | 0.93 | 0.72 | 0.38 | −0.03 / +0.01 | 0.99 / 0.98 / 0.96 |
| 1100 | base | 0.62 | 0.81 | 0.13 | −0.34 / −0.35 | 0.86 / 0.43 / 0.05 |
| 1100 | reshapes matched | 0.69 | 0.90 | 0.43 | 0.00 / −0.02 | 0.86 / 0.58 / 0.08 |
| 1100 | + detector-only | 0.99 | 0.79 | 0.41 | −0.01 / −0.01 | 0.99 / 0.99 / 1.00 |

Taking the detector's box for objects the other sources found removes the
half-metre shortfall of point-fitted boxes and quadruples their IoU, and
raises precision (fewer misplaced centres). Detector-only objects then lift
recall to 0.93–0.99 at the base's precision.

**Tracks** (`eval_tracks.py`), with tracking's detector rules (detector score
≥ 0.5 strong; a detector-only box starts a track only at ≥ 0.6, ByteTrack's
birth rule):

| Scene | Variant | frame recall | frame precision | true / seen tracks | tracks per object | objects in one track | switched |
|---|---|---|---|---|---|---|---|
| 0061 | base | 0.33 | 0.74 | 53 / 98 | 1.51 | 0.63 | 0.08 |
| 0061 | reshapes matched | 0.38 | 0.81 | 52 / 91 | 1.33 | 0.74 | 0.07 |
| 0061 | + detector-only | 0.78 | 0.81 | 46 / 130 | 2.03 | 0.43 | 0.28 |
| 1100 | base | 0.62 | 0.82 | 30 / 44 | 1.35 | 0.69 | 0.00 |
| 1100 | reshapes matched | 0.69 | 0.92 | 34 / 42 | 1.63 | 0.44 | 0.09 |
| 1100 | + detector-only | 0.97 | 0.85 | 28 / 45 | 1.70 | 0.59 | 0.08 |

Without the birth rule, detector-only boxes down to score 0.08 each started a
track: 224 tracks on scene-0061 (69 true), 2.55 per object, 27% of objects in
one track. With it, frame-level recall and precision both beat the base, but
objects split across more tracks and identities switch more (0061 is 70%
pedestrians, which the circle NMS deliberately doesn't suppress). Reshaping
matched objects alone helps scene-0061's continuity and hurts scene-1100's.

`fusion_detector.enabled` stays off by default: the gains are measured on
scenes the model was trained on, and the fragmentation is unresolved.
Turning it on is one config line per profile.
