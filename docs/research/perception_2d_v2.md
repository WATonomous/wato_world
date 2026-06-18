# perception_2d — GroundingDINO + SAM2 + Depth Anything V2 + DINOv2

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
  ── undistorted image ──┼─► Depth Anything V2 ──► relative depth                      │
                         │            │                                                │
  static LiDAR points ───┼─► project into camera ──► (d_lidar, d_da) anchor pairs      │
                         │            │                                                │
                         │            ▼                                                │
                         │   RANSAC affine fit  d_lidar ≈ a·d_da + b                   │
                         │            │  (fallback to recent (a,b) on degenerate fit)  │
                         │            ▼                                                │
                         │   metric depth ──► depth_2d/<cam>/<frame>.npz               │
                         └─────────────────────────────────────────────────────────────┘
                                      DA-V2 unloaded, VRAM freed
                         ┌──────────── PASS 2: tracking (detector + SAM2 + DINOv2) ────┐
per camera stream        │                                                             │
  class vocabulary ──────┼─► fixed taxonomy   OR   Florence-2 open-vocab discovery     │
                         │            │                                                │
                         │            ▼                                                │
  keyframes ─────────────┼─► GroundingDINO ──► class-labeled boxes                     │
                         │            │                                                │
                         │            ▼                                                │
                         │   SAM2 video predictor: box → mask, propagate across stream │
                         │            │  (IoU merge of re-detected boxes; OOM windowing)│
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

1. Write the camera's frames to a temp JPEG dir and `init_state` on it (SAM2's
   video loader takes a directory of sequentially-named JPEGs). `init_state` is
   called with `offload_video_to_cpu=True` so the frame stack lives in host RAM
   and streams to the GPU per frame — essential for long panoramic clips.
2. At each keyframe (every `redetect_every_k` frames) run **GroundingDINO** →
   class-labeled boxes. A box whose mask-bbox IoU with an already-tracked object
   is ≥ `iou_match_threshold` is dropped as the existing object; genuinely new
   boxes get a fresh `obj_id` via `add_new_points_or_box`. This is how objects
   entering mid-clip are introduced without duplicating existing tracks.
3. `propagate_in_video` forward to the next keyframe, recording per-object masks
   into `Masklet`s (persistent ids) — per-frame mask PNGs plus a DINOv2
   appearance embedding extracted every `embeddings.every_k_frames`.

**OOM fallback**: the whole clip is tried in one SAM2 session. On a CUDA OOM
mid-propagation the camera is retried in windows of `sub_clip_frames` (fresh
session each); a window that itself OOMs is recursively halved down to
`min_sub_clip_frames` before being skipped. Object ids reset between windows —
the same discontinuity as between chunks, which the downstream `tracking`
component re-links via the DINOv2 embeddings. Clips that fit never touch this
path.

**Detector backend**: GroundingDINO via HuggingFace Transformers
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

## PASS 1 — depth branch (DA-V2 + LiDAR affine fit)

Produces a metric depth map per frame for `semantic_lifting`'s occlusion test
and for downstream Segment-Lift-Fit.

- **DA-V2 inference**: Depth Anything V2 (`depth-anything-v2-large`) on the full
  image → *relative* depth. Frames are pushed through the backbone
  `depth.batch_size` at a time; the LiDAR alignment that follows each batch stays
  strictly sequential, so the fallback-window state is identical to per-frame
  processing.
- **Why relative + LiDAR-anchored, not metric**: DA-V2's metric checkpoint has
  large absolute error on AV scenes; the relative variant has better ordinal
  consistency and aligns cleanly to LiDAR via an affine fit. Trust LiDAR for
  scale, DA for density.
- **Static-only anchors** (`use_static_anchors_only`, non-optional): cameras
  (~12 Hz) and LiDARs (~20 Hz) are unsynchronized. A dynamic LiDAR point
  projects to where the object *was* at LiDAR time, not where it is in the
  current image, which wrecks the affine fit. Only `lidar_preprocessing`'s
  static points are used as anchors.
- **Sky filter**: DA-V2 emits enormous unreliable values for sky; the top
  `sky_mask_top_fraction` of the image is masked before fitting.
- **RANSAC affine fit**: solve `d_lidar ≈ a·d_da + b` over the matched pairs
  (`ransac_n_iter`, `ransac_inlier_threshold_m`, `min_lidar_anchors`). On a
  degenerate fit, fall back to the median `(a, b)` of the last
  `fallback_window` successful frames; if no successful fit exists yet, write
  nothing (`fit_status == 2`) and downstream treats the missing artifact as
  "relative-only / unavailable."

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
data/artifacts/raw/<bag_id>/perception_2d/<version>/<chunk_id>/
├── detections_2d.parquet                         (MASKLET_SCHEMA, one row / masklet)
├── tracklets_2d.parquet                          (MASKLET_SCHEMA, same rows)
├── masks_2d/<cam_id>/<masklet_id>/<frame_seq>.png (per-frame binary mask PNGs)
│                                  └ dino_feature.npy (DINOv2 embedding for the masklet)
└── depth_2d/<cam_id>/<frame_seq>.npz             (metric depth + fit metadata)
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
| `centroid_depth_m` | float | Metric depth at the mask centroid |
| `tracker_backend` | str | `"sam2"` |

### `depth_2d/<cam>/<frame>.npz`

| Field | Dtype | Shape | Description |
|---|---|---|---|
| `depth_m` | `depth.output_dtype` (float16) | (H, W) | Metric depth in metres |
| `affine_a` | float32 | scalar | DA→metric scale factor |
| `affine_b` | float32 | scalar | DA→metric offset |
| `n_anchors` | int32 | scalar | (d_lidar, d_da) pairs available |
| `n_inliers` | int32 | scalar | RANSAC inliers |
| `rmse_inliers_m` | float32 | scalar | Fit quality |
| `fit_status` | int32 | scalar | 0=ok · 1=fallback to recent (a,b) · 2=failed |

A `fit_status == 2` frame writes **no** npz; downstream treats a missing depth
artifact identically to a written `fit_status == 2`.

---

## Configuration (`src/perception_2d/config/perception_2d.yaml`)

```yaml
discovery:
  backend: "fixed"                 # "fixed" | "florence2"
  model: "microsoft/Florence-2-large-ft"
  task: "<DENSE_REGION_CAPTION>"
  min_confidence: 0.3
  sample_every_k: 10               # florence2 only: discover on every k-th frame
  fixed_classes: ["car", "truck", "bus", "motorcycle", "bicycle",
                  "pedestrian", "traffic cone", "barrier"]

detection:
  model: "IDEA-Research/grounding-dino-base"
  box_threshold: 0.35
  text_threshold: 0.25
  nms_iou: 0.5
  redetect_every_k: 10             # re-detect to admit objects entering mid-clip

segmentation:
  checkpoint: "/data/models/sam2.1_hiera_large.pt"
  config: "configs/sam2.1/sam2.1_hiera_l.yaml"   # bundled in the sam2 package

tracking:
  iou_match_threshold: 0.5         # re-detected box ≥ this IoU = existing object
  offload_video_to_cpu: true       # frame stack in host RAM (long-clip safe)
  sub_clip_frames: 150             # OOM fallback window
  min_sub_clip_frames: 16

depth:
  enabled: true
  model: "depth-anything-v2-large"
  batch_size: 4                    # frames through DA-V2 backbone per GPU batch
  min_lidar_anchors: 30
  ransac_n_iter: 200
  ransac_inlier_threshold_m: 0.5
  use_static_anchors_only: true    # NON-OPTIONAL — see depth branch
  fallback_window: 5
  sky_mask_top_fraction: 0.3
  output_dtype: "float16"

embeddings:
  model: "dinov2_vitl14"
  every_k_frames: 5

prompts_path: "/config/prompts.yaml"
```

---

## Running it

**1. Fetch the model weights on the host** (before launching the container), into
`${MODELS_ROOT}` (bind-mounted read-only at `/data/models`):

```bash
python3 src/perception_2d/scripts/fetch_models.py        # → ./data/models
# or a custom location:
MODELS_ROOT=/srv/wato_models python3 src/perception_2d/scripts/fetch_models.py
# skip a model (e.g. when depth.enabled: false):
python3 src/perception_2d/scripts/fetch_models.py --skip depth_anything_v2
```

This downloads: GroundingDINO + DA-V2 into the HF cache (`/data/models/hf`),
the SAM2.1 checkpoint as a loose `.pt` (`/data/models/sam2.1_hiera_large.pt`),
and DINOv2 into the torch.hub cache (`/data/models/torch_hub`). The container is
configured with `HF_HOME` / `TORCH_HOME` pointed at those caches and runs
`HF_HUB_OFFLINE=1`.

**2. Run the component on a bag** (ingest + lidar_preprocessing must have run
first — perception_2d reads `frame_index`, calibration, and static LiDAR):

```bash
./watod run perception_2d run --bag <bag_id>            # all chunks
./watod run perception_2d run --bag <bag_id> --chunk <chunk_id>
./watod run perception_2d run --bag <bag_id> --force    # ignore existing outputs
```

Chunks whose `detections_2d.parquet` + `tracklets_2d.parquet` already exist are
skipped unless `--force`. Within a chunk, each camera's masklets are checkpointed
to a per-camera resume pickle, so an aborted run resumes per camera rather than
restarting the chunk; `--force` also discards those caches.

**3. Inspect the depth output** with the interactive viewer (needs a forwarded
`DISPLAY`, or run on the host):

```bash
./watod run perception_2d viz --bag <bag_id> [--chunk <chunk_id>] [--cam <cam_id>]
```

See `src/perception_2d/README.md` for the viewer controls.

---

## File layout (`src/perception_2d/src/wato_perception_2d/`)

```
cli.py                     click entrypoint: `run` + `viz`
config.py                  pydantic config schemas + prompts.yaml synonym map
pipeline.py                two-pass orchestrator (depth pass, then tracking pass)
io.py                      frame_index / calibration / static-LiDAR loaders + caches
viz.py                     interactive depth-vs-image viewer
models/
  depth.py                 Depth Anything V2 wrapper (batched infer, unload)
  detector.py              GroundingDINO wrapper (box source for SAM2)
  _sam2_runtime.py         shared SAM2 video-predictor loader / cache
  sam2_tracker.py          detector + SAM2 per-camera tracking → Masklets
  discovery.py             Florence-2 open-vocab discovery (optional backend)
  embeddings.py            DINOv2 appearance embeddings
fusion/
  depth_align.py           build_anchor_pairs / ransac_affine_fit / apply_affine
  masklet.py               Masklet dataclass
```

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