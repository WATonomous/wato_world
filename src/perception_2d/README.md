# perception_2d

2D perception pass (GroundingDINO + SAM2 + Depth Anything V2 + DINOv2
appearance embeddings).

Per camera, the detector (GroundingDINO) emits class-labeled boxes on keyframes
and SAM2's video predictor turns each box into a mask and tracks it across the
frame stream into masklets (persistent object ids) — one box per object, so each
physical object becomes a single masklet (no concept-everything explosion).
DINOv2 appearance embeddings are extracted every k frames (for downstream
re-identification — not used for tracking here). The class vocabulary comes from
one of two sources, selected by `discovery.backend`:

- `fixed` (default) — the closed-set taxonomy (`discovery.fixed_classes`,
  or prompts.yaml `primary_taxonomy` when empty).
- `florence2` — open-vocabulary noun phrases discovered per frame by Florence-2,
  pooled into one concept set per camera stream and fed to the detector.

Depth Anything V2 + a LiDAR-anchored fit produces a metric depth map per
frame. DA-V2's relative checkpoints output *disparity* (larger = closer), so the
fit is in inverse-depth space: `1/depth = scale·disparity + shift`, with RANSAC
inliers judged in metres. Model loading is fail-loud: a missing detector / SAM2 / Depth Anything
raises rather than emitting degraded placeholder output.

Cross-camera identity merging is **not** done here — it belongs to the
downstream `tracking` component (3D/4D association gated by the DINOv2 features
persisted on each masklet). `global_object_id` is left null for `tracking`.

For the full design (two-pass VRAM split, output schemas, config reference) see
[`docs/research/perception_2d_v2.md`](../../docs/research/perception_2d_v2.md).

## Running it

**1. Fetch model weights on the host** (before launching the container) into
`${MODELS_ROOT}`, bind-mounted read-only at `/data/models`:

```bash
python3 src/perception_2d/scripts/fetch_models.py        # → ./data/models
# custom location:
MODELS_ROOT=/srv/wato_models python3 src/perception_2d/scripts/fetch_models.py
# skip a model (e.g. when depth.enabled: false):
python3 src/perception_2d/scripts/fetch_models.py --skip depth_anything_v2
```

Pulls GroundingDINO, Depth Anything V2 and Florence-2 into the HF cache, the
SAM2.1 checkpoint as a loose `.pt`, and DINOv2 into the torch.hub cache. The
container points `HF_HOME` / `TORCH_HOME` at those caches and runs
`HF_HUB_OFFLINE=1`.

**Every model is pinned to an exact revision.** The registry lives at
[`model_registry.py`](src/wato_perception_2d/model_registry.py) and is the
single source of truth for both this fetcher and the in-container runtime
loaders, so a pinned fetch and an unpinned load cannot disagree. These four
models jointly decide every 2D label the pipeline emits; an unpinned HF `main`
or torch.hub default branch would silently change the labeler between runs.

Two consequences worth knowing:

- `torch.hub` caches by ref, so `embeddings.py` loads
  `facebookresearch/dinov2:<sha>`, not the bare repo. A bare reference looks
  for a differently-named cache directory, misses the pre-fetched weights, and
  tries to reach the network from a container whose `/data/models` is
  read-only.
- HF loaders (GroundingDINO, Florence-2, Depth Anything V2) pass
  `revision=hf_revision(repo_id)`. `fetch_models.py` downloads by SHA, and
  huggingface_hub writes no `refs/main` for a SHA fetch, so a loader asking for
  the default `main` can't resolve offline on a clean host. A model ID in the
  config that isn't in the registry now fails at load time on purpose; register
  it with a SHA first.
- Florence-2 is referenced by `discovery.model` in the config but was missing
  from the registry until 2026-09-21, so it was never pre-fetched — enabling
  discovery on a clean host failed at load time. It is registered now.

To upgrade a model: change its revision in `model_registry.py`, re-run
`fetch_models.py`, re-run the affected stages. The new revision is recorded in
every manifest written afterwards, so old and new labels stay distinguishable.

**2. Run on a bag** — `ingest` and `lidar_preprocessing` must have run first
(perception_2d reads `frame_index`, `poses.parquet`, calibration, and static
LiDAR points; each image's LiDAR anchors are projected with the ego pose at
that image's own timestamp, not the LiDAR sweep's):

```bash
./watod run perception_2d run --bag <bag_id>            # all chunks
./watod run perception_2d run --bag <bag_id> --chunk <chunk_id>
./watod run perception_2d run --bag <bag_id> --force    # ignore existing outputs
```

Chunks whose output parquets already exist are skipped unless `--force`. Each
camera's masklets are checkpointed to a per-camera resume cache, so an aborted
run resumes per camera; `--force` discards those caches too.

Outputs land in the chunk directory, `data/artifacts/raw/<bag_id>/chunks/<chunk_id>/`:
`detections_2d.parquet`, `tracklets_2d.parquet`, `masks_2d/`, `depth_2d/`.
The masklet tables and mask PNGs are written and read only through
`wato_common.artifacts.masklets`, which owns their encoding (`frames_present`
is a JSON string on disk, a tuple of ints once read). Consumers use its
`MaskletIndex`, never the paths. `depth_2d/<cam_id>/<camera_seq:06d>.npz` is
likewise owned by `wato_common.artifacts.depth` (`DepthFrame`, `FitStatus`):
a frame whose LiDAR-anchored fit failed is not written at all, so a missing
frame means "no usable depth", and a 0 pixel means no measurement there.

**Re-run depth produced before 2026-10-11.** The original fit modelled depth as
an affine of disparity, which is the wrong shape (the true slope is negative and
hyperbolic), so those `depth_2d/` maps are wrong. Their files carry
`affine_a`/`affine_b` instead of `disp_scale`/`disp_shift`, and the reader now
rejects them with `ArtifactFormatError` rather than serve them as metric;
`--force` regenerates them.

## Tracking: identity and speed

**SAM2 latches a lost object's id onto other objects.** When an object leaves
the image or is covered, this SAM2 build keeps its id alive and, within 0–30
frames, usually reports it on a different, similar object nearby — typically
one that already has its own track. On the debug clip (181 frames, one WATO
pano camera) every such resume was wrong: one pedestrian's masklet walked out of
view and then reappeared on a car, three other pedestrians and a parked car; an
oncoming car's masklet jumped onto a parked car in the very frame it left the
image. Three `tracking` settings handle it:

- `drop_lost_after_frames` (default 5, ~0.4 s at 12 Hz) — a track whose object
  has had no mask for that many consecutive frames ends: no further masks are
  recorded, and it is removed from the SAM2 session at the next keyframe. If the
  object returns and is re-detected it starts a new masklet (re-linked
  downstream via DINOv2). `0` = never.
- `duplicate_mask_iou` (default 0.7) — if two live tracks' masks reach this IoU
  in one frame, the one whose mask changed most since its previous frame jumped,
  and ends. Catches the jump with no empty frame in between.
- Contested pixels: after those decisions, a pixel in several masks goes to the
  object with the highest SAM2 logit, so a bus's mask can't swallow the car
  half-hidden beside it (that pixel would be lifted as both).

Same-keyframe detections are also deduplicated across classes, highest score
first — GroundingDINO's NMS is per class, so a shuttle comes back as both
`truck` and `bus`.

The cost of ending tracks: an object that is genuinely occluded for more than
`drop_lost_after_frames` comes back as a new masklet, and an object the detector
never finds is no longer covered by a hijacked track (on the debug clip, 5
frames of a dark figure in a field GroundingDINO never detected — previously
masked under another pedestrian's id with the label `motorcycle`).

**Speed.** This SAM2 build runs memory attention once per *object* per frame for
every object it holds, so ending tracks also bounds cost. `autocast_dtype`
(default `bfloat16`, upstream's own setting) runs SAM2 calls under CUDA
autocast; the detector and DINOv2 stay float32. Frames ingest wrote as JPEG are
symlinked into SAM2's frame dir instead of being re-encoded, mask PNGs are
written on a thread pool, and DINOv2 embeds all of a frame's objects in one
batch.

Measured on the debug clip (1280×1024), RTX 5070 Ti, full tracking pass
(detector + SAM2 + DINOv2 + mask writes):

| setting | time | peak VRAM | masklets |
|---|---|---|---|
| float32, never end tracks (old path) | 283 s | 7.3 GB | 25, four of them hopping between objects |
| bfloat16, never end tracks | 63 s | 7.3 GB | 26 |
| bfloat16 + current defaults | 27 s | 4.6 GB | 26, none hopping or duplicated |

Runs are bit-for-bit repeatable. bfloat16 alone moves masks no more than merely
feeding SAM2 the source JPEG instead of a re-encode does (per-frame IoU vs the
old path 0.981 vs 0.980 mean). SAM2's `vos_optimized=True` (torch.compile) was
tried and is not usable with this per-object tracker: 224 s of compile, then a
CUDA-graphs buffer-reuse error in memory attention.

## Visualizing depth

`viz` opens an interactive depth-vs-image viewer (the 2D analogue of
lidar_preprocessing's Open3D `watod viz`) for the `depth_2d` artifacts — to
eyeball what Depth Anything V2 + the LiDAR-anchored disparity fit produced:

```bash
./watod run perception_2d viz --bag <bag_id>
./watod run perception_2d viz --bag <bag_id> --chunk <chunk_id> --cam <cam_id>
```

- **frame slider** — scrub through one camera's stream
- **opacity slider** — blend RGB ↔ depth heatmap (0 = image, 1 = depth)
- **SPLIT toggle** — a draggable curtain: RGB on the left, depth on the right
- **◄ cam / cam ►** — switch camera when a chunk has more than one
- keyboard: `←/→` frame · `↑/↓` opacity · `space` split · `c` colormap · `q` quit

Depth is normalized per frame over its finite, positive pixels (2nd–98th
percentile); the colorbar reads true metres and the title shows the fit
status / inlier count / RMSE. Pixels with no valid depth (sky, fit holes)
fall through to the photo.

Needs DISPLAY (or WSLg) forwarded into the container — see the
`perception_2d_dev` service in `modules/docker-compose.dev.yaml`. On WSL2 you
can also just run it on the host, where matplotlib already has a GUI backend:

```bash
PYTHONPATH=src/common/src:src/perception_2d/src \
    python -m wato_perception_2d viz --bag <bag_id>
```

See `wato_world/README.md` and the architecture doc for context.
