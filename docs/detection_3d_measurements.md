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

Pending: base / detector reshaping matched objects only / plus unmatched
detector boxes, through `eval_proposals.py` and `eval_tracks.py`.
