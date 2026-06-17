# perception_2d v2 — SAM3 + Florence-2 + Depth Anything V2

**Paper reference**: UniLiPs (light.princeton.edu/unilips)
**Supersedes**: perception_2d v1 plan (GroundingDINO + SAM2 + DEVA)

The v2 design replaces the GroundingDINO + SAM2 detector/segmenter pair with
Florence-2 + SAM3, adds Depth Anything V2 as a dense depth source aligned to
LiDAR, and prepares 2D outputs that feed a new `semantic_lifting` component.

The design follows UniLiPs' camera-side methodology — open-vocabulary class
discovery, then promptable segmentation, then occlusion-aware lifting into 3D
— but collapses UniLiPs' five-model ensemble (SAM2 + 3×OneFormer + BLIP + CLIP
+ CLIPSeg) into two models (Florence-2 + SAM3) by leveraging SAM3's built-in
open-vocabulary segmentation.

---

## What changed from v1

| Stage | v1 plan | v2 plan | Reason |
|---|---|---|---|
| Discovery | GroundingDINO, fixed vocab | Florence-2 dense-region caption | downstream taxonomy is open-vocab; plays UniLiPs BLIP-analog role |
| Segmentation | SAM2 prompted by boxes | SAM3 prompted by text | SAM3 collapses detection + classification + segmentation |
| Closed-vocab classification | implicit in detector | none — SAM3 is concept-aware | one fewer model class |
| Candidate re-ranking | n/a | n/a — SAM3 presence-token filters hallucinations | SAM3 architectural feature |
| Phrase deduplication | n/a | NEW — synonym coarsening + NMS | Florence-2 produces synonyms |
| Temporal tracking | DEVA | SAM 3.1 built-in tracker (DEVA available as fallback config) | one less model, shared vision encoder with detector |
| ReID | DINOv2 | DINOv2 unchanged | — |
| Depth | none | DA-V2 + LiDAR-anchored RANSAC affine | enables SLF L_depth and occlusion-aware lifting |
| Cross-cam merge depth | sparse LiDAR + 20 m hardcoded fallback | metric depth artifact | reliable for objects with sparse LiDAR coverage |
| Per-frame artifacts | masks, detections, tracklets | masks, detections, tracklets, **depth** | new input for SLF and semantic_lifting |

---

## Architecture

```
                    ┌─── image branch ─────────────────────────────────┐
per camera          │                                                   │
per frame           ▼                                                   │
  ─── undistorted image ─► Florence-2  ⟨DENSE_REGION_CAPTION⟩           │
                          │ noun phrases + rough boxes                  │
                          ▼                                              │
                       SAM3  (text prompted, per phrase)                │
                          │ masks + tight boxes + scores                │
                          ▼                                              │
                    phrase coarsening + NMS                             │
                          │ deduped per-instance masks                  │
                          ▼                                              │
                    DINOv2 ReID embeddings ────────────────────────────►│
                                                                         │
                    ┌─── depth branch (parallel) ─────────────────────►│
                    ▼                                                    │
            Depth Anything V2 → relative depth + gradient confidence    │
                    │                                                    │
static LiDAR pts ──► LiDAR projection → (d_lidar, d_da) anchor pairs    │
                    │                                                    │
                    ▼                                                    │
            RANSAC affine fit  d_lidar ≈ a·d_da + b                     │
                    │                                                    │
                    ▼                                                    │
            metric depth + lidar_coverage mask ──────────────────────►  │
                                                                         │
per chunk                                                                │
  ─────────────────► SAM 3.1 tracker (or DEVA fallback) ◄──────────────┤
                          │ temporally-associated tracklets per camera  │
                          ▼                                              │
                    cross-camera merge using metric depth ◄────────────┘
                          │ global_object_id per masklet
                          ▼
                    artifacts to disk
```

Two branches run in parallel per camera frame (image + depth). They converge
at the chunk-level aggregation step (SAM 3.1 tracker, cross-camera merge),
which produces the final artifacts. Downstream `semantic_lifting` consumes
both the masks and the depth.

---

## Per-frame processing — image branch

### Stage 1 — Florence-2 dense-region caption

**What it does**: takes the full image, returns a list of noun phrases each
tagged with a rough bounding box, via the `<DENSE_REGION_CAPTION>` task.

**Why this stage exists**: SAM3 needs text prompts. Without a discovery step
the only way to use SAM3 is to hardcode a class list — which defeats the
open-vocab choice. Florence-2 is the modern equivalent of UniLiPs' BLIP step
(unprompted open-vocab class proposal) but built specifically for the
caption-then-ground workflow we need.

**Why Florence-2 over BLIP-2 or larger VLMs**:
- MIT license (BLIP-2 is BSD, larger VLMs vary)
- 0.23B / 0.77B parameter sizes — fast enough to run per frame
- Native task tokens for dense region captioning, no prompt engineering
- Outputs structured noun-phrase lists, no JSON parsing of free-form text

**Output**: `list[tuple[phrase: str, rough_box: tuple[x1,y1,x2,y2]]]`

### Stage 2 — SAM3 grounded segmentation

**What it does**: for each unique phrase from Florence-2, prompt SAM3 with the
text → receive every mask matching that concept, with tight box and
confidence score.

**Why this replaces the UniLiPs fseg ensemble**: SAM3 was trained on 4M unique
concepts with the SA-Co dataset. It is itself an open-vocab concept-aware
segmenter. The UniLiPs SAM2 + 3×OneFormer + BLIP + CLIP + CLIPSeg chain is
the workaround for *not having* SAM3 — SAM3 was published in November 2025,
after UniLiPs was written.

**The presence token**: SAM3's per-mask score uses its presence-token
mechanism, designed to discriminate "concept actually in image" from "concept
forced into image because we prompted for it." This is what filters Florence-2
hallucinations and removes the need for CLIP re-ranking.

**Object-only filter**: prompts are filtered against `prompts.yaml` taxonomy
(cars, trucks, pedestrians, cyclists, traffic cones, barriers, etc.) before
SAM3 is called. Stuff classes (road, sidewalk, building, sky) are not in
the prompt set — they're not needed for SLF and add noise. Visibility for
occlusion testing comes from depth, not semantics, so this is fine.

**Output**: `list[(phrase, mask: H×W bool, tight_box, score)]` per image.

### Stage 3 — phrase coarsening + NMS

**What it does**: Florence-2 emits `"car"`, `"vehicle"`, `"sedan"` for the
same object; SAM3 faithfully returns overlapping masks for each. Two-pass
cleanup:

1. **Synonym coarsening**: cluster phrases by CLIP text-embedding cosine
   similarity (threshold ~0.85); within each cluster keep the highest-scored
   mask and discard the rest.
2. **3D-aware NMS**: once depth is available (depth branch completed in
   parallel), back-project mask centroids to world and NMS by 3D centroid
   proximity (1.0 m radius). Falls back to 2D IoU NMS (threshold 0.5) for
   frames where depth alignment failed.

**Why this matters**: without coarsening, the same physical car ends up with
3-5 detections, each with a different DINOv2 embedding crop, each potentially
starting its own track in the downstream tracker. Open-vocab is a feature,
not a free lunch.

**Output**: deduped `list[(canonical_class, mask, box, score)]` per image.

### Stage 4 — DINOv2 ReID embeddings

**Unchanged from v1**. Tight-crop each instance, push through DINOv2-ViT-L/14,
store the 1024-D embedding alongside the mask. Used by the tracking stage for
cross-frame and cross-chunk re-identification.

---

## Per-frame processing — depth branch

This branch runs in parallel with the image branch. Its only inputs are the
undistorted image and the static LiDAR points projected for this frame.

### Stage 5 — Depth Anything V2 inference

**What it does**: DA-V2-Large on the full image, outputs relative depth
(arbitrary scale) plus a gradient-derived confidence map.

**Why relative + LiDAR-anchored, not metric**: per the comparison made during
design, DA-V2's metric checkpoint on AV scenes has MAE of ~8 m at 0-80 m
range against UniLiPs' Table 2. The relative variant has better ordinal
consistency and aligns cleanly to LiDAR via affine fit. The decision was
made to trust LiDAR for scale and DA for density.

**Confidence derivation**: depth-gradient magnitude per pixel — flat regions
get high confidence, depth discontinuities get low confidence. This is a
cheap proxy. If Framing C (L_depth loss term) shows noisy gradients during
SLF integration, upgrade to test-time-augmentation variance.

**Output**: `(relative_depth: H×W float32, confidence: H×W float32)`.

### Stage 6 — LiDAR projection for anchor pairs

**What it does**: project the chunk's static LiDAR points into this camera
frame using `K`, `ego_T_cam`, and the per-frame `world_T_ego`. Records depth
in camera frame for each projected point.

**Critical: static-only filter**. Cameras run at ~12 Hz, LiDARs at ~20 Hz,
unsynchronized (NovAtel GNSS/INS only). A camera frame at `t_cam` has its
nearest LiDAR sweep at `t_cam ± up to ~25 ms`. A vehicle at 30 m/s moves
75 cm in 25 ms. If dynamic LiDAR points are used as anchors, those points
project to where the object *was* at LiDAR time, not where it is in the
current image — the (d_lidar, d_da) pair pulled from that pixel measures
two different objects, which wrecks the affine fit even with RANSAC.

Mitigation: use only static points for anchors. The `dynamic_masks/*.npy`
artifacts from `lidar_preprocessing` already mark this. This filter is
**non-optional** in the first version.

**Sky/upper-image filter**: DA-V2 produces enormous unreliable values for
sky pixels. Mask out the upper ~30% of the image (or pixels where DA depth
exceeds the 95th percentile of LiDAR depths) before fitting.

**Output**: `(d_lidar: M float32, d_da: M float32)` matched pairs.

### Stage 7 — RANSAC affine fit + metric depth

**What it does**: solve `d_lidar = a · d_da + b` over matched pairs using
RANSAC with inlier threshold 0.5 m. Apply to the full DA map → metric depth.
Also compute `lidar_coverage`: True where LiDAR within 5 px contributed to
the affine fit at that pixel, False otherwise.

**Why `lidar_coverage`**: SLF needs to know which pixels are LiDAR-trusted
vs. DA-filled. Framing A (fill LiDAR holes for L_lidar) uses this mask to
decide which pixels to add as DA pseudo-points.

**Degenerate-fit detection**: if the LiDAR depth range across anchors is
< 5 m (camera mostly seeing a wall), the affine is underdetermined. Fall
back to the median (a, b) of the last 5 successful frames in this chunk.
If no successful fit exists yet, write `fit_status = failed` and downstream
treats the depth map as relative-only.

**Output**: `(metric_depth: H×W float16, lidar_coverage: H×W bool, fit_params)`.

---

## Per-chunk aggregation

### Stage 8 — SAM 3.1 tracker

**What it does**: temporally propagate masks across frames within a single
camera stream. SAM 3.1's Object Multiplex update (March 2026) uses
shared-memory multi-object tracking — designed for the AV use case of
many simultaneous tracked objects.

**Why SAM 3.1 over DEVA**:
- Already loading SAM3 for the per-frame segmentation — one fewer dependency.
- Detector and tracker share a vision encoder, features stay consistent.
- Concept-aware (text-prompted), which matches the open-vocab discovery flow.
- DEVA's main strength (bidirectional propagation, decoupling from segmenter)
  doesn't materially help here — the downstream 3D Kalman tracker handles
  long-range ID consistency, and the segmenter is already SAM3 either way.

**DEVA fallback**: kept behind a config flag (`tracker.backend: "sam3" |
"deva"`). If SAM 3.1 tracking quality is bottlenecking pseudo-label quality
during early validation, swap is one config line.

**Output**: persistent `track_id` per masklet within each camera stream.

### Stage 9 — Cross-camera merge

**What it does**: assign a `global_object_id` to masklets visible from
multiple cameras. Logic per `cross_cam_merge.py`:

1. For each masklet, compute the mask centroid `(u, v)` in image space.
2. Look up `depth_2d[u, v]` (the new metric depth artifact).
3. Back-project to world frame using `K`, `ego_T_cam`, `world_T_ego`.
4. Cluster centroids across cameras by 3D proximity (radius
   `cross_camera_match_radius_m = 1.5`).
5. Assign one `global_object_id` per cluster.

**What changes from v1**: the `_estimate_depth_from_lidar` helper with its
20 m hardcoded fallback is replaced by `depth_2d[u, v]` lookup. Every pixel
has a real depth value (DA-filled where LiDAR was silent), so cross-camera
matches no longer fail catastrophically on objects with sparse LiDAR.

**Output**: `global_object_id` assigned to every masklet.

---

## Output artifacts

```
data/artifacts/raw/<bag_id>/perception_2d/v2/<chunk_id>/
├── detections_2d.parquet              (existing schema, class from canonical_class)
├── tracklets_2d.parquet               (existing schema)
├── masks_2d/<cam_id>/<frame_seq>/<det_id>.png       (existing, binary PNG)
└── depth_2d/<cam_id>/<frame_seq>.npz                (NEW)
```

### `depth_2d/<cam>/<frame>.npz` schema

| Field | Dtype | Shape | Description |
|---|---|---|---|
| `depth_m` | float16 | (H, W) | Metric depth in meters |
| `confidence` | float16 | (H, W) | [0, 1], DA gradient-derived |
| `lidar_coverage` | bool | (H, W) | True where LiDAR within 5 px contributed |
| `affine_a` | float32 | scalar | DA→metric scale factor |
| `affine_b` | float32 | scalar | DA→metric offset |
| `n_anchors` | int32 | scalar | Number of (d_lidar, d_da) pairs used |
| `n_inliers` | int32 | scalar | RANSAC inliers |
| `rmse_inliers_m` | float32 | scalar | Fit quality metric |
| `fit_status` | int8 | scalar | 0=ok, 1=fallback to prior frame, 2=failed |

Storage budget at 1920×1080: ~8 MB compressed per frame → ~5 GB per chunk
across 12 cameras × ~100 frames. If painful, drop `confidence` or quantize
to uint8.

### `detections_2d.parquet` schema additions

| Field | Type | Description |
|---|---|---|
| `canonical_class` | str | Post-coarsening canonical class name from `prompts.yaml` |
| `raw_phrase` | str | Original Florence-2 noun phrase (kept for debugging open-vocab gaps) |
| `sam3_score` | float32 | SAM3 presence-token score |
| `discovery_score` | float32 | Florence-2 confidence |
| `centroid_depth_m` | float32 | depth_2d at mask centroid, used for cross-cam merge |
| `tracker_backend` | str | "sam3" or "deva" |

---

## UniLiPs alignment

Stages of UniLiPs that map to perception_2d v2:

| UniLiPs concept | v2 implementation |
|---|---|
| BLIP open-vocab class proposal | Florence-2 dense-region caption |
| SAM2 mask generation | SAM3 (text-prompted, mask + class in one) |
| OneFormer ensemble classification | SAM3 (concept-aware by design) |
| CLIP re-ranking | SAM3 presence-token score |
| CLIPSeg per-pixel grounding | SAM3 mask output |
| Majority-vote class assignment per SAM mask | SAM3 returns one class per mask directly |

Stages of UniLiPs that belong to other components, **not** perception_2d:

| UniLiPs concept | Where it lives |
|---|---|
| Occlusion-aware semantic lifting (Eq. 1) | `semantic_lifting` component |
| Probabilistic label propagation (Algorithm 1) | future addition to `semantic_lifting` |
| Iterative Weighted Update (Eqs. 3-4) | future addition to `semantic_lifting` |
| Moving-object detection from map inconsistency | already done by `lidar_preprocessing/classify.py` |
| HDBSCAN box clustering | SLF in `proposal_generation` does this better |
| Adaptive Spherical Occlusion Culling | future, for densified depth output |

---

## Configuration schema

```yaml
# src/perception_2d/config/perception_2d.yaml
perception_2d:
  discovery:
    model: florence-2-large-ft        # or florence-2-large
    task: "<DENSE_REGION_CAPTION>"
    min_confidence: 0.3
  segmentation:
    model: sam3.1
    checkpoint: facebook/sam3.1
    min_score: 0.5
    object_only: true                 # filter prompts against prompts.yaml
  phrase_dedup:
    synonym_clip_threshold: 0.85
    nms_3d_radius_m: 1.0
    nms_2d_iou: 0.5
  tracker:
    backend: "sam3"                   # or "deva"
    deva_model: "deva-vit-l"          # only if backend=deva
  reid:
    model: dinov2_vitl14
    every_k_frames: 5
  depth:
    enabled: true
    model: depth-anything-v2-large
    min_lidar_anchors: 30
    ransac_n_iter: 200
    ransac_inlier_threshold_m: 0.5
    use_static_anchors_only: true     # NON-OPTIONAL; see Stage 6
    fallback_window: 5
    sky_mask_top_fraction: 0.3
    confidence_method: gradient_magnitude
    output_dtype: float16
  cross_cam:
    match_radius_m: 1.5
  upstream_versions:
    ingest: v1
    lidar_preprocessing: v1
```

---

## File layout

```
src/perception_2d/
├── config/
│   └── perception_2d.yaml
└── src/wato_perception_2d/
    ├── __init__.py
    ├── cli.py                       (update for v2 commands)
    ├── config.py                    (rewrite Pydantic schemas for v2)
    ├── io.py                        (add depth_2d artifact helpers)
    ├── pipeline.py                  (orchestration — rewrite for new stages)
    │
    ├── discovery.py        NEW      Florence-2 wrapper
    ├── sam3.py             NEW      SAM3 detector + tracker wrapper
    ├── phrase_dedup.py     NEW      Synonym coarsening + NMS
    ├── projection.py       NEW      LiDAR → image projection (planned in SAM4D doc)
    ├── depth.py            NEW      Depth Anything V2 wrapper
    ├── depth_align.py      NEW      RANSAC affine fit + apply
    ├── reid.py                      DINOv2 (extract from existing pipeline.py)
    ├── cross_cam_merge.py           (update to use depth_2d artifact)
    │
    ├── detector.py         REMOVE   GroundingDINO no longer used
    ├── segmenter.py        REMOVE   SAM2 no longer used
    └── tracker_2d.py       REMOVE   replaced by sam3.py (DEVA logic moves there if needed)
```

---

## Migration from v1

Steps in order; each can be validated independently before the next:

1. **Build `projection.py`** (prerequisite for everything else). Tests:
   projection round-trips against existing `_estimate_depth_from_lidar` for
   known pixels.
2. **Build `depth.py`** standalone — DA-V2 wrapper, no LiDAR integration yet.
   Test on single image, visualize relative depth.
3. **Build `depth_align.py`** with synthetic anchors. Test: known affine
   recoverable to within 0.5%.
4. **Build `discovery.py`** — Florence-2 wrapper. Test on one image, inspect
   noun phrases.
5. **Build `sam3.py`** — wrapper around SAM3 image + video. Test on one
   image, then one camera stream.
6. **Build `phrase_dedup.py`** — synonym clustering + NMS. Unit tests on
   synthetic phrase lists.
7. **Wire end-to-end in `pipeline.py`** for one chunk, one camera. Visualize
   masks + depth + alignment quality.
8. **Scale to 12 cameras**, validate cross-camera merge.
9. **Validate against held-out data** if any ground truth available.
10. **Update Dockerfile** to install Florence-2, SAM3, DA-V2 instead of
    GroundingDINO + SAM2 + DEVA.

---

## Risks and things to validate

- **DA-V2 license**: confirm Apache 2.0 before commercial use. V1 was more
  restrictive.
- **SAM3 license**: "SAM License" — verify exact terms for non-research use.
- **Florence-2 dense-region caption quality on AV scenes**: trained mostly on
  general photos, may underperform on highway/dashcam viewpoints. Validate
  early — if poor, fall back to RAM++ tags or a fixed taxonomy.
- **SAM 3.1 tracker quality on long sequences**: chunks can be 30s+ at 12 Hz,
  i.e. 360+ frames per camera stream. Object Multiplex was tested on shorter
  videos in the SAM 3.1 release notes.
- **Per-frame DA inference cost**: DA-V2-Large is ~0.5s/image on an A100.
  12 cameras × 100 frames × 0.5s = 600s of GPU time per chunk just for depth.
  May need to batch across cameras or downsample DA input resolution.
- **Cross-camera merge cluster radius**: 1.5 m may be too tight given depth
  noise. Tune empirically.

---

## Open questions deferred to implementation

- Should `cross_cam_merge.py` weight masklets by `sam3_score` when clustering?
- Should the SAM 3.1 tracker run forward-only or also backward across the
  chunk overlap window (similar to v1's DEVA bidirectional plan)?
- Is the `lidar_coverage` mask better computed at the affine-fit step
  (per-anchor) or post-hoc by re-projecting LiDAR after metric depth exists?
- How should we handle the case where Florence-2 emits a phrase not in
  `prompts.yaml` (long-tail discovery)? Drop it, route it to
  `open_vocab_discovery`, or keep with `canonical_class = "unknown"`?

---

## Summary of actionable steps

1. Define `DepthFrame` and updated `DetectionRow` schemas in
   `src/common/src/wato_common/schemas.py`.
2. Add `depth_2d_dir()` and `depth_2d_path()` helpers to
   `src/common/src/wato_common/artifact_store.py`.
3. Build `projection.py` (prerequisite, planned in SAM4D guidance).
4. Build `depth.py` standalone wrapper around DA-V2.
5. Build `depth_align.py` with RANSAC affine fit.
6. Build `discovery.py` wrapper around Florence-2.
7. Build `sam3.py` wrapper around SAM3 detector + SAM 3.1 tracker.
8. Build `phrase_dedup.py` with CLIP-similarity coarsening + 2D/3D NMS.
9. Update `cross_cam_merge.py` to read `depth_2d` artifact instead of running
   its own LiDAR depth estimation.
10. Rewrite `pipeline.py` to orchestrate the new stages.
11. Remove `detector.py`, `segmenter.py`, `tracker_2d.py` (and their tests).
12. Update `docker/perception_2d.Dockerfile` with Florence-2, SAM3, DA-V2.
13. Update `config/pipeline.yaml` and create
    `src/perception_2d/config/perception_2d.yaml`.
