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

Depth Anything V2 + a LiDAR-anchored affine fit produces a metric depth map per
frame. Model loading is fail-loud: a missing detector / SAM2 / Depth Anything
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

Pulls GroundingDINO + Depth Anything V2 into the HF cache, the SAM2.1 checkpoint
as a loose `.pt`, and DINOv2 into the torch.hub cache. The container points
`HF_HOME` / `TORCH_HOME` at those caches and runs `HF_HUB_OFFLINE=1`.

**2. Run on a bag** — `ingest` and `lidar_preprocessing` must have run first
(perception_2d reads `frame_index`, calibration, and static LiDAR points):

```bash
./watod run perception_2d run --bag <bag_id>            # all chunks
./watod run perception_2d run --bag <bag_id> --chunk <chunk_id>
./watod run perception_2d run --bag <bag_id> --force    # ignore existing outputs
```

Chunks whose output parquets already exist are skipped unless `--force`. Each
camera's masklets are checkpointed to a per-camera resume cache, so an aborted
run resumes per camera; `--force` discards those caches too.

Outputs land at `data/artifacts/raw/<bag_id>/perception_2d/<version>/<chunk_id>/`:
`detections_2d.parquet`, `tracklets_2d.parquet`, `masks_2d/`, `depth_2d/`.

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
