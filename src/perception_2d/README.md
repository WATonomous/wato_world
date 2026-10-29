# perception_2d

Per camera: a detector boxes keyframes, SAM2's video predictor tracks each box
into a masklet (one per physical object), DINOv2 embeds each masklet, and Depth
Anything V2 gives a metric depth map per image, fitted to static LiDAR.
Cross-camera identity is left to the `tracking` component (`global_object_id`
stays null).

## Running

Fetch the weights on the host first, into `${MODELS_ROOT}` (default
`./data/models`, mounted read-only at `/data/models`):

```bash
python3 src/perception_2d/scripts/fetch_models.py [--skip depth_anything_v2]
```

Then, after ingest and lidar_preprocessing:

```bash
./watod run perception_2d --bag <bag_id> [--chunk <chunk_id>] [--force]
./watod run perception_2d viz --bag <bag_id> [--chunk <chunk_id> --cam <cam_id>]
./watod test perception_2d
```

A chunk whose output parquets exist is skipped. Each camera's masklets are
checkpointed, so an aborted run resumes per camera. `--force` discards those
caches and clears the chunk's `masks_2d/` and `depth_2d/` first (mask
directories are named by masklet id, which is new every run).

## How a chunk runs

Two passes, so the heavy models never share VRAM:

1. **Depth.** DA-V2 outputs relative *disparity*, so the fit is in inverse-depth
   space, `1/depth = scale·disparity + shift`, by RANSAC with inliers judged in
   metres. The anchors are the static (`~dynamic_mask`) world-frame points of
   every sweep the image matched, projected with the ego pose at the image's
   own `camera_timestamp_ns`. With too few anchors, a recent frame's fit is
   reused; a frame with no usable fit is not written.
2. **Tracking.** The detector runs every `redetect_every_k` images; boxes that
   match an existing track (mask IoU ≥ `iou_match_threshold`) or a better box
   at the same keyframe are dropped, and the rest prompt SAM2. SAM2 latches a
   lost object's id onto a similar nearby object within a few images, so a
   track **ends** when its object has had no mask for `drop_lost_after_frames`
   images, or its mask reaches `duplicate_mask_iou` with another live track's
   (the one whose mask changed most ends). A pixel claimed by several masks goes
   to the highest logit. On CUDA OOM the camera is retried in windows of
   `sub_clip_frames`. SAM2 runs under `autocast_dtype` (bfloat16); the detector
   and DINOv2 stay float32.

**One entry per image.** `frame_index` repeats an image once per sweep it is
nearest to (~4.9× on WATO, ~1.7× on nuScenes), so both passes collapse rows by
`camera_seq` first (`pipeline._one_frame_per_image`). Every frame-counted
setting counts images.

Model loading is fail-loud: a missing detector, SAM2 or DA-V2 raises, and SAM2
and the detector are import-checked before the depth pass.

## Detector backends

`detection.backend` picks the box source; both return the same boxes:

- **`yolo`** (default): Ultralytics YOLO26-x, COCO weights, closed-set.
  `detection.yolo.class_map` maps COCO onto the taxonomy. COCO has no traffic
  cone or barrier, so **this backend never detects those** (logged at startup).
  It needs `discovery.backend: fixed`. AGPL-3.0.
- **`grounding_dino`**: open-vocabulary, text-prompted by the taxonomy or by
  Florence-2 discovery (`discovery.backend: florence2`). Its scores depend on
  which other words share the prompt.

`models/yolo_detector.py` guards three silent failures: Ultralytics reads numpy
as BGR (frames are RGB), weights load by pinned path (a bare name downloads
whatever upstream ships), and `YOLO_OFFLINE` / `YOLO_CONFIG_DIR` are set before
import. The pipeline refuses a `class_map` target that nearly matches a
taxonomy class (`pedestrain`, `traffic cone` vs `traffic_cone`).

## Outputs

In `data/artifacts/raw/<bag_id>/chunks/<chunk_id>/`:

| Artifact | Owner |
|---|---|
| `tracklets_2d.parquet`, `detections_2d.parquet`, `masks_2d/<masklet_id>/<camera_seq:06d>.png`, `masks_2d/<masklet_id>/dino_feature.npy` | `wato_common.artifacts.masklets` (consumers use `MaskletIndex`) |
| `depth_2d/<cam_id>/<camera_seq:06d>.npz` | `wato_common.artifacts.depth`: a missing file means no usable depth; a 0 pixel means no measurement |
| `manifest_perception_2d.json` | Provenance incl. pinned model revisions, hashed inputs |

Outputs from before 2026-10-15 (per-row processing) aren't comparable, and
depth from before 2026-10-11 is rejected by the reader; rerun with `--force`.

## Models

Every model is pinned in
[`model_registry.py`](src/wato_perception_2d/model_registry.py), which both
`fetch_models.py` and the runtime loaders read. HF loaders pass
`revision=hf_revision(repo_id)` (a SHA fetch writes no `refs/main`, so `main`
can't resolve offline), and DINOv2 loads `facebookresearch/dinov2:<sha>`
(torch.hub caches by ref). A model ID not in the registry fails at load time:
register it with a SHA first. To upgrade one, change its revision, re-run
`fetch_models.py` and the affected stages; manifests record the revision.

## Depth viewer

`viz` opens a matplotlib window per chunk: a frame slider, an RGB ↔ depth
opacity slider, a draggable split curtain, camera switching; keys `←/→` frame,
`↑/↓` opacity, `space` split, `c` colormap, `q` quit. Depth is normalised per
frame (2nd–98th percentile); the colorbar reads metres and the title shows fit
status, inliers and RMSE. Needs DISPLAY / WSLg in the container, or run it on
the host:
`PYTHONPATH=src/common/src:src/perception_2d/src python -m wato_perception_2d viz --bag <bag_id>`.

## Known limitations

- **Ego vehicle.** PANO_NN sees the car's hood; YOLO scores it `car` and SAM2
  tracks it. Nothing filters ego regions yet.
- An object occluded for longer than `drop_lost_after_frames` comes back as a
  new masklet; `tracking` must re-link it.

Measurements behind these defaults:
[docs/perception_2d_measurements.md](../../docs/perception_2d_measurements.md).
