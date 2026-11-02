# perception_2d

Per camera: a detector boxes keyframes, SAM2's video predictor tracks each box
into a masklet (one per physical object), DINOv2 embeds each masklet, and Depth
Anything V2 gives a metric depth map per image, fitted to static LiDAR.
Cross-camera identity is left to the `tracking` component.

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
./watod run perception_2d ego-mask --bag <bag_id> [--config <yaml>]   # see "Ego mask"
./watod run perception_2d compare --bag <bag_id> --baseline <root>    # see "Comparing two runs"
./watod test perception_2d
```

WATO bags (`ingest.wato*.yaml`) take the rig's config, which adds the car's
ego mask: `--config /ws/src/perception_2d/config/perception_2d.wato.yaml`.

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
2. **Tracking.** The detector runs every `redetect_every_k` images. A box
   matching a track (box IoU with its mask bbox ≥ `iou_match_threshold`)
   **supports** it; the rest, less repeats of a better box at the same
   keyframe, prompt SAM2. Then, per track:
   - **Keep-alive.** It rises by one per supported keyframe (up to
     `support_keyframes`) and falls by one per keyframe the mask covers at
     least `support_min_area` of the image with no supporting box; at zero the
     track ends. A large mask the detector doesn't see (the sky, the road)
     ends within a few keyframes; a distant car COCO YOLO boxed once keeps its
     track. With `birth_score` > 0, a weaker box starts a track tentatively,
     its masks held back until boxes support it at `confirm_keyframes`
     keyframes in a row (off by default: YOLO rarely re-boxes distant cars).
   - **Re-prompting.** A confident box (≥ `reprompt_score`) backing one track
     alone, whose mask bbox has drifted below `reprompt_below_iou` from it,
     re-prompts that track.
   - **Class.** The one its boxes' scores sum highest for; its score is their
     mean.
   - **Identity.** SAM2 latches a lost object's id onto a similar nearby
     object within a few images, so a track ends when its object has had no
     mask for `drop_lost_after_frames` images, or its mask reaches
     `duplicate_mask_iou` with another live track's (the one whose mask
     changed most ends).
   - **Pixels.** A pixel claimed by several masks goes to the highest logit;
     a track owning under `collapse_fraction` of its own mask for
     `collapse_after_frames` images has latched onto another track's object
     and ends. Each mask drops its specks (components under `island_fraction`
     of its largest), and specks a track wins in a contested region go to the
     runner-up.
   - **Containment.** A `contained_classes` mask at least
     `containment_fraction` inside a `container_classes` (vehicle) mask and
     ending above its bottom is the driver seen through the vehicle's
     windows: not recorded, its pixels left to the vehicle, ended after
     `containment_after_frames`. An object in front of a vehicle reaches the
     road below it and is kept.
   - **Ego vehicle.** A box or mask at least `ego_mask.max_fraction` inside
     the camera's `ego_mask` polygons (the rack, hood and antenna) never
     becomes or stays a track, and every mask loses those pixels.

   On CUDA OOM the camera is retried in windows of `sub_clip_frames`. SAM2
   runs under `autocast_dtype` (bfloat16); the detector and DINOv2 stay
   float32.

**One entry per image.** `frame_index` repeats an image once per sweep it is
nearest to (~4.9× on WATO, ~1.7× on nuScenes); both passes read it through
`wato_common.artifacts.frame_index.FrameIndex`, which hands out each image once
with every sweep it is paired with (depth anchors on all of them) and the pose
at the image's own time. Every frame-counted setting counts images.

Model loading and inference are fail-loud: a detector, SAM2, DINOv2 or DA-V2
failure stops the run instead of leaving a camera with fewer masklets or no
embeddings, both of which look like a finished chunk. SAM2 and the detector are
import-checked, and the SAM2, YOLO and DINOv2 weights load-checked on the CPU,
before each chunk's depth pass, so a missing checkpoint fails in seconds.

## Detector backends

`detection.backend` picks the box source; both return the same boxes:

- **`yolo`** (default): Ultralytics YOLO26-x, COCO weights, closed-set.
  `detection.yolo.class_map` maps COCO onto the taxonomy; a taxonomy class
  nothing maps to is never detected (logged at startup). It needs
  `discovery.backend: fixed`. AGPL-3.0.
- **`grounding_dino`**: open-vocabulary, text-prompted by the taxonomy or by
  Florence-2 discovery (`discovery.backend: florence2`). Its scores depend on
  which other words share the prompt.

The taxonomy is the six classes COCO covers: car, truck, bus, motorcycle,
bicycle, pedestrian. Traffic cones and barriers are left out: COCO has neither,
GroundingDINO's hits for them were mostly hydrants, poles, walls and the ego
rack, and a class listed but never detected would be labelled absent from every
image.

`models/yolo_detector.py` guards three silent failures: Ultralytics reads numpy
as BGR (frames are RGB), weights load by pinned path (a bare name downloads
whatever upstream ships), and `YOLO_OFFLINE` / `YOLO_CONFIG_DIR` are set before
import. The pipeline refuses a `class_map` target that nearly matches a
taxonomy class (`pedestrain`, `Motorcycle` vs `motorcycle`).

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

## Ego mask

`ego_mask` (per camera, pixel polygons in the rectified image) is the only
per-vehicle part of the config; `perception_2d.yaml` has none, and each rig
gets its own file (`perception_2d.wato.yaml`). `ego-mask` drafts and checks it
from a bag's perception_2d output, since the detector boxes the vehicle's body
and SAM2 holds it all chunk: a mask at the image border that doesn't move
while the vehicle drives. Into `raw/<bag_id>/ego_mask/` (or `--out`) it writes,
per camera, the median of images taken metres apart (the scene blurs, the body
stays sharp), an overlay of those body masks (red) and the config's polygons
(green), and `ego_mask.yaml`, a draft block from the body masks.

- **New vehicle:** run perception_2d on a driving bag without an ego mask, then
  `ego-mask`; check the overlays, add by hand what no masklet covers (the WATO
  antenna), and put the block in the rig's config.
- **Check:** after a run with the polygons, `ego-mask --config <rig yaml>`
  reports body masklets left outside them (a camera moved, a rack added).
  None means the polygons still hold. The polygons are tied to the image size
  (a mismatch raises) and to the rectification; nothing else detects drift.

## Comparing two runs

`compare` reviews what a change to the tracking rules removed, against a run
of the same bag from before the change. Both runs box the same images with the
same detector, so an object the change lost is a baseline mask that no current
mask matches (pixel IoU < 0.5), checked on every `--every`-th image (5).

```bash
./watod run perception_2d compare --bag <bag_id> --baseline /data/artifacts/<root> \
    [--chunk <chunk_id> ...] [--config <the current run's yaml>]
```

The baseline is another artifact root: the old code run with
`ARTIFACT_ROOT_URI` pointing at it, its chunks linking the current root's
ingest and lidar_preprocessing outputs (depth can stay off). Into
`raw/<bag_id>/compare/` (or `--out`) it writes:

- `review.csv`: the baseline masklets with unmatched masks, largest loss first;
- `sheets/`: pictures of them (baseline mask red, the current mask over it
  outlined green);
- `masklets.csv` and `summary.txt`.

Give each review row a verdict: `real` (an object the current run lost),
`junk` (not an object: ego body, sky, a driver), `covered` (still masked, by
another masklet or a trimmed mask) or `unsure`; masklets inside `--config`'s
ego mask are pre-marked junk. Run it again: the verdicts are kept, and the
summary gives per class the share of the baseline's real-object masks lost,
with unsure and unreviewed rows counted as real for an upper bound. It can't
see an object neither run masks.

## Depth viewer

`viz` opens a matplotlib window per chunk: a frame slider, an RGB ↔ depth
opacity slider, a draggable split curtain, camera switching; keys `←/→` frame,
`↑/↓` opacity, `space` split, `c` colormap, `q` quit. Depth is normalised per
frame (2nd–98th percentile); the colorbar reads metres and the title shows fit
status, inliers and RMSE. Needs DISPLAY / WSLg in the container, or run it on
the host:
`PYTHONPATH=src/common/src:src/perception_2d/src python -m wato_perception_2d viz --bag <bag_id>`.

## Known limitations

- **Ego vehicle.** The WATO ego mask is for one camera mounting, traced on
  `sensors_2026_07_07-23_01_24_2`; a remounted camera needs it redrawn (see
  "Ego mask"). Under the default `perception_2d.yaml` (no ego mask) the rack is
  tracked as a truck.
- An object occluded for longer than `drop_lost_after_frames` comes back as a
  new masklet; `tracking` must re-link it.

Measurements behind these defaults:
[docs/perception_2d_measurements.md](../../docs/perception_2d_measurements.md).
