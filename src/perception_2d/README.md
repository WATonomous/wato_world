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

## Tracking speed

SAM2 (this pinned build) runs memory attention once per *object* per frame and
never forgets an object, so cost tracks objects-ever-detected, not
objects-visible. Two `tracking` settings bound it:

- `drop_lost_after_frames` (default 30, ~2.5 s at 12 Hz) — an object with no
  mask for that long is removed from the session at the next keyframe. If it
  comes back and is re-detected it starts a new masklet (re-linked downstream
  via DINOv2). `0` = never drop.
- `autocast_dtype` (default `bfloat16`, upstream's own setting) — SAM2 calls
  run under CUDA autocast; the detector and DINOv2 stay float32.

Frames ingest wrote as JPEG are symlinked into SAM2's frame dir instead of being
re-encoded, mask PNGs are written on a thread pool, and DINOv2 embeds all of a
frame's objects in one batch.

Measured on 181 frames (1280×1024) of one WATO pano camera, RTX 5070 Ti, full
tracking pass (detector + SAM2 + DINOv2 + mask writes):

| setting | time | peak VRAM | masks vs. old float32 path (per-frame IoU, mean / median) |
|---|---|---|---|
| float32, never drop (old path) | 283 s | 7.3 GB | — |
| bfloat16, never drop | 63 s | 7.3 GB | 0.981 / 0.993 |
| bfloat16, drop after 30 (default) | 38 s | 4.8 GB | 0.976 / 0.993 |

Runs are bit-for-bit repeatable. bfloat16 moves masks no more than merely
feeding SAM2 the source JPEG instead of a re-encode does (0.980 / 0.996).
SAM2's `vos_optimized=True` (torch.compile) was tried and is not usable with
this per-object tracker: 224 s of compile, then a CUDA-graphs buffer-reuse error
in memory attention.

## Visualizing depth

`viz` opens an interactive depth-vs-image viewer (the 2D analogue of
lidar_preprocessing's Open3D `watod viz`) for the `depth_2d` artifacts — to
eyeball what Depth Anything V2 + the LiDAR-anchored affine fit produced:

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
