# perception_2d — YOLO26 / GroundingDINO + SAM2 + Depth Anything V2 + DINOv2

**Status**: implemented (`src/perception_2d/`).
**Supersedes**: the SAM3 + Florence-2 concept-everything plan. That design was
pivoted away from — SAM3's text-prompted "segment every region matching a
concept" produced a mask explosion (multiple synonymous detections per physical
object) and its long-video tracker needed upstream monkeypatches to bound VRAM.
The current design uses a *detector* (one box per object) as the box source for
the **SAM2** video predictor, which both segments and tracks. Florence-2 is
retained only as an optional open-vocabulary discovery backend.

This component follows UniLiPs' camera-side methodology — class discovery, then
promptable segmentation, then dense depth aligned to LiDAR — and prepares the 2D
artifacts that feed the downstream `semantic_lifting` component.

---

## Architecture — two VRAM-disjoint passes per chunk

The two heavy GPU model sets (Depth Anything V2; the detector + SAM2 + DINOv2)
are **never co-resident**. Each chunk runs as two passes so peak VRAM is roughly
one model set, which is what lets it run on small-VRAM cards (~6 GB). The cost is
a second image decode (each pass loads the frames once).

```
                         ┌──────────────── PASS 1: depth (DA-V2 only) ─────────────────┐
per camera, per frame    │                                                             │
  ── undistorted image ──┼─► Depth Anything V2 ──► relative disparity                  │
                         │            │                                                │
  static LiDAR points ───┼─► project into camera ──► (d_lidar, disp) anchor pairs      │
                         │            │                                                │
                         │            ▼                                                │
                         │   RANSAC fit  1/d_lidar ≈ scale·disp + shift                │
                         │            │  (fallback to the last good fit if degenerate) │
                         │            ▼                                                │
                         │   metric depth ──► depth_2d/<cam>/<frame>.npz               │
                         └─────────────────────────────────────────────────────────────┘
                                      DA-V2 unloaded, VRAM freed
                         ┌──────────── PASS 2: tracking (detector + SAM2 + DINOv2) ────┐
per camera stream        │                                                             │
  class vocabulary ──────┼─► fixed taxonomy   OR   Florence-2 open-vocab discovery     │
                         │            │                                                │
                         │            ▼                                                │
  keyframes ─────────────┼─► YOLO26 or GroundingDINO ──► class-labeled boxes           │
                         │            │                                                │
                         │            ▼                                                │
                         │   SAM2 video predictor: box → mask, propagate across stream │
                         │            │  (IoU merge of re-detections; end lost/jumped  │
                         │            │   tracks; pixel ownership; OOM windows; bf16)  │
                         │            ▼                                                │
                         │   Masklets (persistent ids) + DINOv2 embedding every k      │
                         │            │                                                │
                         │            ▼                                                │
                         │   detections_2d.parquet · tracklets_2d.parquet · masks_2d/  │
                         └─────────────────────────────────────────────────────────────┘
```

Model loading is **fail-loud**: a missing detector / SAM2 / depth model raises
rather than emitting degraded output, and the SAM2 + detector imports are checked
*before* the depth pass so a missing install fails fast instead of after the
whole depth pass has run.

---

## Class vocabulary — `discovery.backend`

The detector is text-prompted, so it needs a class vocabulary. Two sources,
selected by `discovery.backend`:

- **`fixed`** (default) — the closed-set taxonomy from `discovery.fixed_classes`
  (or `prompts.yaml`, or the hardcoded fallback when neither is present). One
  prompt phrase per class. Deterministic and fast; the right default for a known
  AV taxonomy (car, truck, bus, motorcycle, bicycle, pedestrian, traffic cone,
  barrier).
- **`florence2`** — open-vocabulary. Florence-2 (`<DENSE_REGION_CAPTION>`) runs
  on every `sample_every_k`-th frame of a camera stream; phrases scoring above
  `min_confidence` are pooled, deduped, and canonicalised through the
  `prompts.yaml` synonym map into one concept set for that camera, then fed to
  the detector. Use this when the taxonomy is genuinely open / long-tail.

Either way the output is a `(text_prompt, canonical_class)` concept list per
camera. The detector — not SAM — is what bounds the object count: one box per
object means one masklet per object (no synonym explosion).

---

## PASS 2 — detector + SAM2 tracking (per camera)

1. Point SAM2 at a temp dir of sequentially-named JPEGs (its video loader's
   input): symlinks to ingest's own files when they are JPEG (CompressedImage
   bytes are written verbatim), re-encoded at q95 otherwise. `init_state` is
   called with `offload_video_to_cpu=True` so the frame stack lives in host RAM
   and streams to the GPU per frame — essential for long panoramic clips — and
   `async_loading_frames=True` so decoding overlaps tracking.
2. At each keyframe (every `redetect_every_k` frames), first remove the tracks
   that **ended** since the last keyframe (below) from the session
   (`remove_object`). Then run **GroundingDINO** → class-labeled boxes,
   highest score first. A box whose mask-bbox IoU with an already-tracked
   object is ≥ `iou_match_threshold` is dropped as the existing object, and so
   is one that repeats a box already accepted at this keyframe (the detector's
   NMS is per class, so one shuttle comes back as both `truck` and `bus`);
   genuinely new boxes get a fresh `obj_id` via `add_new_points_or_box`. This
   is how objects entering mid-clip are introduced without duplicating existing
   tracks. A keyframe with no live objects and no detections is skipped (SAM2
   refuses to propagate zero objects — previously that error abandoned the
   whole window, so an empty first frame lost every later object).
3. `propagate_in_video` forward to the next keyframe, recording per-object masks
   into `Masklet`s (persistent ids) — per-frame mask PNGs (written on a thread
   pool) plus a DINOv2 appearance embedding extracted every
   `embeddings.every_k_frames`, batched over the frame's objects. Each frame,
   first decide which tracks **end** (on SAM2's raw masks), then give any pixel
   several surviving masks claim to the one with the highest logit.

**Why tracks end — SAM2 latches lost ids onto other objects.** SAM2 does not let
go of an object it has lost (it left the image, or got covered): within 0–30
frames it tends to report the lost object's id on a *different*, similar object
nearby — usually one that already has its own track, since per-object inference
has no notion of two objects owning the same pixels. Debugged on 181 frames of a
WATO pano camera, every such resume was wrong: a pedestrian's masklet walked out
of view at frame 61 and then reappeared on a car (77), on three other
pedestrians (110–122) and on a parked car (145–162) under one id; an oncoming
car's masklet jumped onto a parked car in the very frame it left the image, with
no empty frame between. So a track ends — no further masks recorded, removed
from the session at the next keyframe — when either:

- its object has had no mask for `drop_lost_after_frames` consecutive frames
  (default 5, ~0.4 s at 12 Hz; 0 = never), or
- its mask and another live track's reach mask IoU ≥ `duplicate_mask_iou`
  (default 0.7) in the same frame. Of the two, the one whose mask changed most
  since its own previous mask is the one that jumped and ends; a track prompted
  this frame counts as having jumped, so an existing track is never displaced.

The trade: an object that comes back after a real occlusion longer than
`drop_lost_after_frames` starts a new masklet when re-detected — the same
discontinuity as an OOM window boundary, re-linked downstream via DINOv2. On the
debug clip no track had a gap followed by a correct resume, and the masklet
count was unchanged (26). Ending tracks also bounds cost: the pinned SAM2 runs
memory attention and the memory encoder once *per object* per frame and keeps
per-frame state for every object it holds, so without it per-frame cost and VRAM
grow with every object detected so far rather than with what is visible.

**Why contested pixels go to one object**: independent per-object inference
also lets one object's mask wrap around another — the shuttle's mask swallowed a
car half-hidden beside it — and a pixel in two masks would be lifted to 3D as
both. Ownership is decided only after ending tracks: done first, it would split
a jumped track's stolen object with its rightful track instead of revealing the
jump (the two masks never overlap, so `duplicate_mask_iou` can't fire).

**Precision**: SAM2 calls (`init_state`, `add_new_points_or_box`,
`propagate_in_video`, `remove_object`) run under CUDA autocast at
`tracking.autocast_dtype` — `bfloat16` by default, which is how upstream runs
SAM2 and what lets its attention use flash kernels. The detector and DINOv2
run outside it, in float32.

**OOM fallback**: the whole clip is tried in one SAM2 session. On a CUDA OOM
mid-propagation the camera is retried in windows of `sub_clip_frames` (fresh
session each); a window that itself OOMs is recursively halved down to
`min_sub_clip_frames` before being skipped. Object ids reset between windows —
the same discontinuity as between chunks, which the downstream `tracking`
component re-links via the DINOv2 embeddings. Clips that fit never touch this
path.

**Detector backend** (`detection.backend`): **`yolo`** (default) — Ultralytics
YOLO26-x, COCO weights, closed-set; `detection.yolo.class_map` maps COCO classes
onto the taxonomy and drops the rest, so traffic cones and barriers (absent
from COCO) are never detected. Frames are converted RGB → BGR (Ultralytics'
numpy convention), the pinned `.pt` is loaded by path, and ultralytics runs with
`YOLO_OFFLINE=1`. Chosen over GroundingDINO for the fixed taxonomy because
GroundingDINO's class scores depend on which other phrases share its prompt
(on WATO data the shuttle came out `truck`, dark pedestrians `motorcycle`, and
fire hydrants `traffic cone`), and YOLO is ~7× faster per keyframe. `yolo` can't
be paired with `discovery.backend: florence2`; open-vocabulary work uses
**`grounding_dino`**: GroundingDINO via HuggingFace Transformers
(`AutoModelForZeroShotObjectDetection`, default `IDEA-Research/grounding-dino-base`),
which needs no CUDA custom-op compile (unlike the standalone `groundingdino`
package). Use a `-tiny` checkpoint for a lighter/faster model.

**Segmenter backend**: SAM2.1 via Meta's official `sam2` package,
`build_sam2_video_predictor(config_file, ckpt_path)` pointed at a *local*
checkpoint (`/data/models/sam2.1_hiera_large.pt`). The hydra `config_file` ships
inside the `sam2` package, so only the `.pt` is fetched. The container runs
`HF_HUB_OFFLINE=1`, so a loose `.pt` path is used rather than
`from_pretrained`.

---

## PASS 1 — depth branch (DA-V2 + LiDAR inverse-depth fit)

Produces a metric depth map per frame for `semantic_lifting`'s occlusion test
and for downstream Segment-Lift-Fit.

- **DA-V2 inference**: Depth Anything V2 (`depth-anything-v2-large`) on the full
  image → relative **disparity** (affine-invariant inverse depth: larger =
  closer, ~0 for sky — what DA-V2's relative checkpoints output, despite the
  name). Frames are pushed through the backbone
  `depth.batch_size` at a time; the LiDAR alignment that follows each batch stays
  strictly sequential, so the fallback-window state is identical to per-frame
  processing.
- **Why relative + LiDAR-anchored, not metric**: DA-V2's metric checkpoint has
  large absolute error on AV scenes; the relative variant has better ordinal
  consistency and aligns cleanly to LiDAR via a scale + shift fit in
  inverse-depth space. Trust LiDAR for scale, DA for density.
- **Static-only anchors** (`use_static_anchors_only`, non-optional): cameras
  (~12 Hz) and LiDARs (~20 Hz) are unsynchronized. A dynamic LiDAR point
  projects to where the object *was* at LiDAR time, not where it is in the
  current image, which wrecks the fit. Only points outside
  `lidar_preprocessing`'s `dynamic_mask` (the precision artifact) are used as
  anchors — never the Step F `motion_proposals`, which are recall-oriented and
  false-positive tolerant. (`~dynamic_mask` also admits ambiguous,
  under-evidenced and near-ego points; static-voxel membership would be the
  stricter anchor set — a known follow-up.)
- **Camera-time pose**: the static points are world-frame (each deskewed at
  its own time), so they are projected with the ego pose when the *image* was
  taken — looked up at the frame's `camera_timestamp_ns` through
  `wato_common.pose_lookup` — not `frame_index.world_T_ego`, which is the
  LiDAR sweep's pose. The two are up to ~44 ms apart on the WATO rig and
  ~120 ms on nuScenes; using the sweep's pose misplaced a point 30 m away by
  15–17 cm median and up to 0.9–1.3 m. A frame whose time falls in an
  untrusted stretch of the pose stream gets no anchors.
- **Sky filter**: DA-V2 emits enormous unreliable values for sky; the top
  `sky_mask_top_fraction` of the image is masked before fitting.
- **RANSAC inverse-depth fit**: solve `1/d_lidar ≈ scale·disp + shift` over the
  matched pairs (`ransac_n_iter`, `ransac_inlier_threshold_m`,
  `min_lidar_anchors`), then `depth_m = 1/(scale·disp + shift)`, clamped to
  250 m; pixels whose fitted inverse depth is ≤ 0 get 0 ("no measurement").
  Hypotheses are 2-point samples in inverse-depth space, but inliers are judged
  by the residual in **metres** after inverting, and the final least-squares
  refit weights each inlier by `d_lidar²` so it approximately minimises the
  metric error. `scale` must be positive (disparity rises as depth falls).
  On a degenerate fit, fall back to the most recent successful `(scale, shift)`
  among the last `fallback_window` frames; if none exists yet, write nothing
  (`fit_status == 2`) and downstream treats the missing artifact as
  "unavailable."
- **Why not `d_lidar ≈ a·disp + b`**: that was the original model and it is
  wrong for disparity — the true relation is hyperbolic with a *negative*
  slope, and the fit rejected every `a ≤ 0`, so frames either failed or took
  a positive slope fitted to noise. On real DA-V2 output for a synthetic ground
  plane (7–77 m anchors), the unconstrained depth-affine least-squares slope
  came out negative with 8.3 m RMSE; the inverse-depth fit kept 344/390 anchors
  as inliers at 0.12 m RMSE. Files written by the old fit used the field names
  `affine_a`/`affine_b`; the reader rejects them (`ArtifactFormatError`), so
  re-run perception_2d on any chunk produced before the fix.

DA-V2 is built inside this pass and **unloaded** in the pass's `finally` block,
freeing its VRAM before PASS 2 builds the detector / SAM2.

---

## DINOv2 appearance embeddings

DINOv2 (`embeddings.model`, default `dinov2_vitl14`) embeds a tight crop of each
masklet every `embeddings.every_k_frames` frames; the embedding is persisted
alongside the masklet (`dino_feature.npy` in the masklet's mask directory). It is
**not** used for tracking here — it is the feature the downstream `tracking`
component uses for cross-camera / cross-chunk re-identification.

---

## Cross-camera identity is deferred downstream

perception_2d's deliverable is **per-camera** masklets + masks + metric depth +
DINOv2 embeddings. Resolving the same physical object across the 12 cameras is a
3D/4D problem that belongs in the downstream `tracking` component (gated by the
DINOv2 features persisted here). `global_object_id` is left **null** for
`tracking` to populate. (The earlier SAM3 plan did cross-camera merge inside
perception_2d; that has been removed.)

---

## Output artifacts

```
data/artifacts/raw/<bag_id>/chunks/<chunk_id>/
├── detections_2d.parquet                         (MASKLET_SCHEMA, one row / masklet)
├── tracklets_2d.parquet                          (MASKLET_SCHEMA, same rows)
├── masks_2d/<masklet_id>/<camera_seq:06d>.png    (per-frame binary mask PNGs)
│                     └ dino_feature.npy          (DINOv2 embedding for the masklet)
└── depth_2d/<cam_id>/<camera_seq:06d>.npz        (metric depth + fit metadata)
```

Written and read through `wato_common.artifacts.masklets` and
`wato_common.artifacts.depth`, which own these formats.

```
```

`detections_2d.parquet` and `tracklets_2d.parquet` carry the same masklet rows —
perception_2d does not separate per-frame detections from tracks; a masklet *is*
the tracked detection.

### `detections_2d.parquet` / `tracklets_2d.parquet` (`MASKLET_SCHEMA`)

| Field | Type | Description |
|---|---|---|
| `masklet_id` | str | Stable id within the chunk/camera |
| `bag_id`, `chunk_id`, `cam_id` | str | Provenance |
| `cls` | str | Canonical taxonomy class |
| `score` | float | Masklet confidence |
| `frames_present` | str | JSON list[int] of `camera_seq` where the mask exists |
| `mask_path` | str | Directory of per-frame mask PNGs |
| `dino_feature_path` | str? | DINOv2 embedding `.npy` (null if not extracted) |
| `global_object_id` | str? | **null** — populated downstream by `tracking` |
| `raw_phrase` | str | Raw detector label before canonicalisation |
| `det_score` | float | Detector (× SAM2 mask) confidence |
| `discovery_score` | float | Detector / Florence-2 discovery confidence |
| `centroid_depth_m` | float | Not populated yet (always 0.0) |
| `tracker_backend` | str | `"sam2"` |

### `depth_2d/<cam>/<frame>.npz`

| Field | Dtype | Shape | Description |
|---|---|---|---|
| `depth_m` | `depth.output_dtype` (float16) | (H, W) | Metric depth in metres; 0 = no measurement |
| `disp_scale` | float32 | scalar | `1/depth_m = disp_scale·disparity + disp_shift` |
| `disp_shift` | float32 | scalar | (disparity = DA-V2's raw output) |
| `n_anchors` | int32 | scalar | (d_lidar, disparity) pairs available |
| `n_inliers` | int32 | scalar | RANSAC inliers |
| `rmse_inliers_m` | float32 | scalar | Metric RMSE over the inliers |
| `fit_status` | int32 | scalar | 0=ok · 1=fallback to a recent fit · 2=failed |

A `fit_status == 2` frame writes **no** npz; downstream treats a missing depth
artifact identically to a written `fit_status == 2`.

---

## Configuration, running, file layout

See `src/perception_2d/README.md` and `src/perception_2d/config/perception_2d.yaml`
(schema and field docs: `config.py`).

---

## Risks and things to validate

- **DA-V2 inference cost**: DA-V2-Large is ~0.5 s/image; 12 cameras × ~100
  frames is significant per chunk. `depth.batch_size` amortises this when the
  card has headroom.
- **GroundingDINO recall on AV viewpoints**: zero-shot detection can miss small
  / distant objects. `redetect_every_k` and `box_threshold` are the main knobs;
  LiDAR-dynamic-point cross-modal prompting (see `sam4d_guidance.md`) is a future
  recovery path.
- **SAM2 tracker quality on long sequences**: chunks can be 360+ frames per
  camera. The OOM windowing keeps it from dying, but very long tracks may
  fragment — re-linked downstream via DINOv2.
- **Cross-camera merge** is *not* validated here — it is the downstream
  `tracking` component's responsibility.
```
