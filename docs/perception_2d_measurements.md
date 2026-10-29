# perception_2d: measurements behind the defaults

The measurements and past fixes behind perception_2d's defaults. The behaviour
itself is in the component README. "The debug clip" is 181 frames of one WATO
pano camera at 1280×1024; timings are on an RTX 5070 Ti.

## Detector backends

- YOLO26-x takes ~30 ms per keyframe at `imgsz: 1280`.
- Ultralytics treats a numpy image as BGR. Passing the pipeline's RGB frames
  straight through measurably changed detections on a WATO frame (a fire hydrant
  and a stop sign were lost, every score shifted), so `detect()` converts.
- COCO semantics, not bugs: the campus shuttle comes out `truck`, and a cyclist
  is a `person` (→ pedestrian) plus a separate `bicycle`.
- Ego-vehicle false positive: PANO_NN (front) sees the hood along the bottom
  ~15 px (x ≈ 450–850 at 1280×1024). YOLO scores it `car` at 0.26–0.35 and SAM2
  tracks it for the rest of the clip (a 161-frame masklet on the debug clip).
  GroundingDINO did not fire on it.

## SAM2 track ending

When an object leaves the image or is covered, this SAM2 build keeps its id
alive and within 0–30 frames usually reports it on a different, similar object
nearby. On the debug clip every such resume was wrong: one pedestrian's masklet
walked out of view and reappeared on a car, three other pedestrians and a parked
car; an oncoming car's masklet jumped onto a parked car in the frame it left the
image. That is why tracks end (`drop_lost_after_frames`, `duplicate_mask_iou`).

The cost: a genuinely occluded object comes back as a new masklet, and an
object the detector never finds is no longer covered by a hijacked track (5
frames of a dark figure GroundingDINO never detected were previously masked
under another pedestrian's id as `motorcycle`).

## Speed (debug clip, full tracking pass)

| setting | time | peak VRAM | masklets |
|---|---|---|---|
| float32, never end tracks | 283 s | 7.3 GB | 25, four hopping between objects |
| bfloat16, never end tracks | 63 s | 7.3 GB | 26 |
| bfloat16 + current defaults | 27 s | 4.6 GB | 26, none hopping or duplicated |

Runs are bit-for-bit repeatable. bfloat16 moves masks no more than feeding SAM2
the source JPEG instead of a re-encode does (per-frame IoU vs the float32 path
0.981 vs 0.980 mean). SAM2's `vos_optimized=True` (torch.compile) is unusable
with this per-object tracker: 224 s of compile, then a CUDA-graphs buffer-reuse
error in memory attention.

## One entry per image (2026-10-15)

Before 2026-10-15, `frame_index` rows were fed straight through: each WATO image
went through Depth Anything ~5× (the last row's single sweep anchored the
surviving file), SAM2 held ~5× the frames in host RAM (chunk 0001 of
`recording_20260309_211754_0_with_eidos_trimmed` was SIGKILLed at 1,154 frames
per camera for ~238 images), and `frames_present` repeated each `camera_seq`
~5×. The frame-counted defaults were then 10 / 5 / 5 rows
(`redetect_every_k` / `drop_lost_after_frames` / `embeddings.every_k_frames`);
2 / 1 / 1 images keeps that behaviour. On that bag's chunk 0000 (parked in the
garage), 2 / 1 / 1 gave 63 masklets over 6,647 masked images against the old
60 over 6,545, while 10 / 5 / 5 as images found 49 over 6,018 (10 fewer
pedestrians). Outputs written before then aren't comparable.

## Depth fit (2026-10-11)

The original fit modelled depth as an affine of disparity, which is the wrong
shape (the true slope is negative and hyperbolic). Those files carry
`affine_a` / `affine_b` instead of `disp_scale` / `disp_shift`, and the reader
rejects them with `ArtifactFormatError`.

## Model registry (2026-09-21)

Florence-2 was referenced by `discovery.model` but missing from the registry
until 2026-09-21, so it was never pre-fetched and enabling discovery on a clean
host failed at load time.
